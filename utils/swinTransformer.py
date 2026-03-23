import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class SpectralFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, dropout=0.1):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(in_channels, self.freq_dim, 2, dtype=torch.float32) * 0.02)
        self.norm = nn.LayerNorm(in_channels)
        self.post_conv = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Conv1d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        B, C, L = x.shape
        x_in = x.permute(0, 2, 1) # [B, L, C]
        x_in = self.norm(x_in)
        x_in = x_in.float()
        
        with torch.cuda.amp.autocast(enabled=False):
            # --- 修正点：dim=1 才是对序列长度进行 FFT ---
            x_fft = torch.fft.rfft(x_in, n=L, dim=1, norm='ortho') 
            weight = torch.view_as_complex(self.complex_weight)
            
            if x_fft.shape[1] != weight.shape[1]: # 维度检查修正
                 # 这里需要小心处理插值，通常针对 freq 维度
                 pass 
            
            # 权重相乘 [B, L, C] * [C, L] -> 需要转置一下让广播正确
            # 简化起见，直接乘权重: 
            # weight shape: [C, Freq]
            # x_fft shape:  [B, Freq, C] (因为 dim=1 FFT后，Freq维度在1)
            # 正确的乘法逻辑:
            x_fft = x_fft.permute(0, 2, 1) # [B, C, Freq]
            weight = weight.unsqueeze(0)   # [1, C, Freq]
            
            # 动态插值权重以防长度不匹配
            if x_fft.shape[-1] != weight.shape[-1]:
                weight_real = F.interpolate(self.complex_weight[..., 0].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight_imag = F.interpolate(self.complex_weight[..., 1].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight = torch.complex(weight_real, weight_imag)

            out_fft = x_fft * weight
            out = torch.fft.irfft(out_fft, n=L, dim=-1, norm='ortho') # 逆变换回序列
            
        return self.post_conv(out.to(x.dtype))

class HybridStem(nn.Module):
    def __init__(self, vocab_size, base_filters, seq_len):
        super().__init__()
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

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # 新增：用于存储最近一次推理的注意力权重
        self.last_attn = None 

    def forward(self, x):
        B, N, C = x.shape
        pad_l = 0
        pad_r = (self.window_size - N % self.window_size) % self.window_size
        if pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r))
        
        _, N_padded, _ = x.shape
        num_windows = N_padded // self.window_size
        
        x_windows = x.view(B, num_windows, self.window_size, C)
        x_windows = x_windows.view(-1, self.window_size, C)
        
        Bw, W, _ = x_windows.shape
        qkv = self.qkv(x_windows).reshape(Bw, W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # --- 新增：保存 Attention 权重 (detach 并转到 cpu 以节省显存) ---
        # Shape: [Batch*NumWindows, Heads, WinSize, WinSize]
        self.last_attn = attn.detach().cpu() 
        
        attn = self.attn_drop(attn)
        x_windows = (attn @ v).transpose(1, 2).reshape(Bw, W, C)
        x_windows = self.proj(x_windows)
        x_windows = self.proj_drop(x_windows)

        x = x_windows.view(B, num_windows, self.window_size, C)
        x = x.view(B, N_padded, C)

        if pad_r > 0:
            x = x[:, :N, :]
            
        return x

class SparseTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=64, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 替换标准 Attention 为 WindowAttention
        self.attn = WindowAttention(dim, num_heads, window_size, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        
        # 增强型 FFN：包含卷积以混合不同窗口间的信息 (类似于 Conformer)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # 深度卷积混合局部上下文
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        x = x + self.attn(self.norm1(x))
        
        # FFN 需要 permute 因为中间有 Conv1d
        res = x
        x = self.norm2(x)
        x = self.mlp[0](x) # Linear
        x = x.transpose(1, 2) # -> [B, C, L]
        x = self.mlp[2](x) # Conv1d
        x = x.transpose(1, 2) # -> [B, L, C]
        x = self.mlp[3](x) # GELU
        x = self.mlp[4](x) # Dropout
        x = self.mlp[5](x) # Linear
        x = self.mlp[6](x) # Dropout
        
        return res + x

class DeepGenomicTransUnet(nn.Module):
    def __init__(self, vocab_size=5, base_filters=64, input_len=10240):
        super().__init__()
        self.stem = HybridStem(vocab_size, base_filters, seq_len=input_len)
        
        # 计算各阶段长度
        l1 = input_len
        l2 = l1 // 2
        l3 = l2 // 2
        
        self.enc1 = ResConvBlock(base_filters, base_filters*2, stride=2)
        self.fft_block1 = SpectralFeatureBlock(base_filters*2, base_filters*2, seq_len=l1//2)
        
        self.enc2 = ResConvBlock(base_filters*2, base_filters*4, stride=2)
        self.fft_block2 = SpectralFeatureBlock(base_filters*4, base_filters*4, seq_len=l2//2)
        
        self.enc3 = ResConvBlock(base_filters*4, base_filters*8, stride=2)
        self.fft_block3 = SpectralFeatureBlock(base_filters*8, base_filters*8, seq_len=l3//2)
        
        self.trans_dim = base_filters * 8
        
        # --- 修改：替换为稀疏 Transformer 编码器 ---
        # 这里的 seq_len 大约为 10240 / 8 = 1280
        # 窗口大小设为 64 或 128 较为合适
        self.sparse_transformer = nn.ModuleList([
            SparseTransformerBlock(
                dim=self.trans_dim, 
                num_heads=8, 
                window_size=64,  # 设置稀疏窗口大小
                mlp_ratio=4., 
                dropout=0.1
            )
            for _ in range(6)
        ])
        
        self.pos_enc = nn.Parameter(torch.randn(1, 12000, self.trans_dim) * 0.02)
        
        self.up3 = nn.ConvTranspose1d(base_filters*8, base_filters*4, 2, 2)
        self.dec3 = ResConvBlock(base_filters*8, base_filters*4)
        
        self.up2 = nn.ConvTranspose1d(base_filters*4, base_filters*2, 2, 2)
        self.dec2 = ResConvBlock(base_filters*4, base_filters*2)
        
        self.up1 = nn.ConvTranspose1d(base_filters*2, base_filters, 2, 2)
        self.dec1 = ResConvBlock(base_filters + base_filters, base_filters)
        
        self.final = nn.Conv1d(base_filters, 1, 1)

    def forward(self, x):
        # Encoder Path
        c1 = self.stem(x)
        
        c2 = self.enc1(c1)
        c2_fft = self.fft_block1(c2)
        c2 = c2 + c2_fft # 保持原逻辑
        
        c3 = self.enc2(c2)
        c3_fft = self.fft_block2(c3)
        c3 = c3 + c3_fft 
        
        c4 = self.enc3(c3)
        c4_fft = self.fft_block3(c4)
        c4 = c4 + c4_fft

        # Transformer Bottleneck
        b = c4.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        seq_len = b.size(1)
        
        # Positional Encoding Interp
        if seq_len > self.pos_enc.size(1):
             pos_emb = F.interpolate(self.pos_enc.permute(0,2,1), size=seq_len).permute(0,2,1)
        else:
             pos_emb = self.pos_enc[:, :seq_len, :]
        
        b = b + pos_emb
        
        # 应用稀疏 Transformer 层
        for layer in self.sparse_transformer:
            b = layer(b)
            
        b = b.permute(0, 2, 1) # [B, L, C] -> [B, C, L]
        
        # Decoder Path
        d3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        
        return self.final(d1).squeeze(1)

# # 测试代码
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = DeepGenomicTransUnet(input_len=10240).to(device)
#     dummy_input = torch.randn(2, 5, 10240).to(device) # Batch=2, Vocab=5, Length=10240
#     output = model(dummy_input)
#     print(f"Input shape: {dummy_input.shape}")
#     print(f"Output shape: {output.shape}")