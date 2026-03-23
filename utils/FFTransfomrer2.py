import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class SpectralFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, dropout=0.3):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(in_channels, self.freq_dim, 2, dtype=torch.float32) * 0.02)
        self.norm = nn.LayerNorm(in_channels)
        # 保持原有的 Dropout
        self.post_conv = nn.Sequential(
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, L = x.shape
        x_in = x.permute(0, 2, 1)
        x_in = self.norm(x_in)
        x_in = x_in.permute(0, 2, 1)
        x_in = x_in.float()
        with torch.cuda.amp.autocast(enabled=False):
            x_fft = torch.fft.rfft(x_in, n=L, dim=-1, norm='ortho')
            weight = torch.view_as_complex(self.complex_weight)
            if x_fft.shape[-1] != weight.shape[-1]:
                weight_real = F.interpolate(self.complex_weight[..., 0].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight_imag = F.interpolate(self.complex_weight[..., 1].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight = torch.complex(weight_real.squeeze(0), weight_imag.squeeze(0))
            out_fft = x_fft * weight.unsqueeze(0)
            out = torch.fft.irfft(out_fft, n=L, dim=-1, norm='ortho')
        return self.post_conv(out.to(x.dtype))

class HybridStem(nn.Module):
    def __init__(self, vocab_size, base_filters, seq_len, dropout=0.2): # 新增 dropout 参数
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(vocab_size, base_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout) # [新增] 防止对特定输入序列过拟合
        )
        self.fft_project = nn.Conv1d(vocab_size, base_filters, 1)
        self.fft_block = SpectralFeatureBlock(base_filters, base_filters, seq_len=seq_len, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters, 1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout) # [新增] 融合后的 Dropout
        )

    def forward(self, x):
        out_conv = self.conv_branch(x)
        out_fft_in = self.fft_project(x)
        out_fft = self.fft_block(out_fft_in)
        return self.fusion(torch.cat([out_conv, out_fft], dim=1))

class ResConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=0.2): # 新增 dropout 参数
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout), # [关键新增] 这里的 Dropout 对防止过拟合至关重要
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_c)
            )
            
    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))

class PositionalEncoding(nn.Module):
    """标准的 Sinusoidal Positional Encoding，比随机初始化更利于泛化"""
    def __init__(self, d_model, max_len=12000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Length, Channel]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class DeepGenomicTransUnet(nn.Module):
    def __init__(self, vocab_size=5, base_filters=64, input_len=3000, dropout_rate=0.3): 
        super().__init__()
        
        # 1. Stem
        self.stem = HybridStem(vocab_size, base_filters, seq_len=input_len, dropout=dropout_rate)
        
        # 长度计算
        l1 = input_len
        l2 = l1 // 2
        l3 = l2 // 2
        
        # 2. Encoder
        self.enc1 = ResConvBlock(base_filters, base_filters*2, stride=2, dropout=dropout_rate)
        self.fft_block1 = SpectralFeatureBlock(base_filters*2, base_filters*2, seq_len=l1//2, dropout=dropout_rate)
        
        self.enc2 = ResConvBlock(base_filters*2, base_filters*4, stride=2, dropout=dropout_rate)
        self.fft_block2 = SpectralFeatureBlock(base_filters*4, base_filters*4, seq_len=l2//2, dropout=dropout_rate)
        
        self.enc3 = ResConvBlock(base_filters*4, base_filters*8, stride=2, dropout=dropout_rate)
        self.fft_block3 = SpectralFeatureBlock(base_filters*8, base_filters*8, seq_len=l3//2, dropout=dropout_rate)
        
        # 3. Transformer Bottleneck
        self.trans_dim = base_filters * 8
        
        # [改进2] 减少层数: 6 -> 4。防止过拟合，迫使模型学习更鲁棒的特征。
        # [改进3] 提高 Transformer 内部 Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.trans_dim, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.3,           # 保持高 Dropout
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4) # 降到 4 层
        
        # [改进4] 使用标准的 Positional Encoding 替代随机初始化
        self.pos_enc = PositionalEncoding(self.trans_dim, max_len=input_len + 1000)
        
        # 4. Decoder
        self.up3 = nn.ConvTranspose1d(base_filters*8, base_filters*4, 2, 2)
        self.dec3 = ResConvBlock(base_filters*8, base_filters*4, dropout=dropout_rate)
        
        self.up2 = nn.ConvTranspose1d(base_filters*4, base_filters*2, 2, 2)
        self.dec2 = ResConvBlock(base_filters*4, base_filters*2, dropout=dropout_rate)
        
        self.up1 = nn.ConvTranspose1d(base_filters*2, base_filters, 2, 2)
        self.dec1 = ResConvBlock(base_filters + base_filters, base_filters, dropout=dropout_rate)
        
        self.final = nn.Conv1d(base_filters, 1, 1)

    def forward(self, x):
        # x: [Batch, 5, Length]
        
        # Encoder Flow
        c1 = self.stem(x)
        
        c2 = self.enc1(c1)
        c2 = c2 + self.fft_block1(c2) # Add residual
        
        c3 = self.enc2(c2)
        c3 = c3 + self.fft_block2(c3) # Add residual
        
        c4 = self.enc3(c3)
        c4 = c4 + self.fft_block3(c4) # Add residual

        # Transformer Flow
        # Conv1d output is [Batch, Channel, Length], Transformer needs [Batch, Length, Channel]
        b = c4.permute(0, 2, 1) 
        b = self.pos_enc(b)     # Add PE
        b = self.transformer(b)
        b = b.permute(0, 2, 1)  # Back to [Batch, Channel, Length]
        
        # Decoder Flow (Skip Connections)
        d3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        
        # Output [Batch, Length]
        return self.final(d1).squeeze(1)