import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class SpectralFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, dropout=0.3):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(in_channels, self.freq_dim, 2, dtype=torch.float32) * 0.02)
        self.norm = nn.LayerNorm(in_channels)
        self.post_conv = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Conv1d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        B, C, L = x.shape
        x_in = x.permute(0, 2, 1)
        x_in = self.norm(x_in)
        x_in = x_in.permute(0, 2, 1)
        # 显式转 float 避免 SHAP 兼容问题
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
    def __init__(self, vocab_size, base_filters, seq_len):
        super().__init__()
        # 修正：inplace=False 以支持 SHAP
        self.conv_branch = nn.Sequential(
            nn.Conv1d(vocab_size, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=False) 
        )
        self.fft_project = nn.Conv1d(vocab_size, base_filters, 1)
        self.fft_block = SpectralFeatureBlock(base_filters, base_filters, seq_len=seq_len)
        self.fusion = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters, 1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out_conv = self.conv_branch(x)
        out_fft_in = self.fft_project(x)
        out_fft = self.fft_block(out_fft_in)
        return self.fusion(torch.cat([out_conv, out_fft], dim=1))

class ResConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # 修正：inplace=False 以支持 SHAP
        self.body = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=False),
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

class DeepGenomicTransUnet(nn.Module):
    def __init__(self, vocab_size=5, base_filters=64, input_len=10240):
        super().__init__()
        self.stem = HybridStem(vocab_size, base_filters, seq_len=input_len)
        l1=input_len; l2=l1//2; l3=l2//2
        self.enc1 = ResConvBlock(base_filters, base_filters*2, stride=2)
        self.fft_block1 = SpectralFeatureBlock(base_filters*2, base_filters*2, seq_len=l1//2)
        self.enc2 = ResConvBlock(base_filters*2, base_filters*4, stride=2)
        self.fft_block2 = SpectralFeatureBlock(base_filters*4, base_filters*4, seq_len=l2//2)
        self.enc3 = ResConvBlock(base_filters*4, base_filters*8, stride=2)
        self.fft_block3 = SpectralFeatureBlock(base_filters*8, base_filters*8, seq_len=l3//2)
        self.trans_dim = base_filters * 8
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.trans_dim, nhead=8, dim_feedforward=2048, dropout=0.3, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos_enc = nn.Parameter(torch.randn(1, 12000, self.trans_dim) * 0.02)
        self.up3 = nn.ConvTranspose1d(base_filters*8, base_filters*4, 2, 2)
        self.dec3 = ResConvBlock(base_filters*8, base_filters*4)
        self.up2 = nn.ConvTranspose1d(base_filters*4, base_filters*2, 2, 2)
        self.dec2 = ResConvBlock(base_filters*4, base_filters*2)
        self.up1 = nn.ConvTranspose1d(base_filters*2, base_filters, 2, 2)
        self.dec1 = ResConvBlock(base_filters + base_filters, base_filters)
        self.final = nn.Conv1d(base_filters, 1, 1)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.enc1(c1)
        # 注意：这里也需要加上fft部分，参考之前的逻辑
        c2_fft = self.fft_block1(c2)
        c2 = c2 + c2_fft
        
        c3 = self.enc2(c2)
        c3_fft = self.fft_block2(c3)
        c3 = c3 + c3_fft
        
        c4 = self.enc3(c3)
        c4_fft = self.fft_block3(c4)
        c4 = c4 + c4_fft

        b = c4.permute(0, 2, 1)
        seq_len = b.size(1)
        if seq_len > self.pos_enc.size(1):
             pos_emb = F.interpolate(self.pos_enc.permute(0,2,1), size=seq_len).permute(0,2,1)
        else:
             pos_emb = self.pos_enc[:, :seq_len, :]
        b = b + pos_emb
        b = self.transformer(b)
        b = b.permute(0, 2, 1)
        
        d3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        
        return self.final(d1).squeeze(1)