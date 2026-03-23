import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

# -----------------------------------------------------------------------------
# 优化组件 1: RoPE (旋转位置编码) - 替代绝对位置编码
# -----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=20000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        # 缓存 cos 和 sin [L, D/2] -> [1, 1, L, D] (为了 broadcasting)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [B, Heads, L, Head_Dim]
        if seq_len > self.cos_cached.shape[2]:
            # 如果推理长度超过缓存，动态重新计算（极少情况）
            return self._dynamic_forward(x, seq_len)
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

    def _dynamic_forward(self, x, seq_len):
        # 兜底逻辑
        dim = x.shape[-1]
        device = x.device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, Heads, L, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# -----------------------------------------------------------------------------
# 优化组件 2: BioSpectralGating (保持原逻辑，增强数值稳定性)
# -----------------------------------------------------------------------------
class BioSpectralGating(nn.Module):
    def __init__(self, channels, seq_len, dropout=0.1):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.channels = channels
        
        # 权重初始化
        self.complex_weight = nn.Parameter(
            torch.randn(channels, self.freq_dim, 2, dtype=torch.float32) * 0.02
        )
        
        # 生物学偏置注入：增强 1/3 频率 (3-nt periodicity)
        with torch.no_grad():
            period_3_idx = seq_len // 3
            # 扩大带宽范围，防止整除误差
            start = max(0, period_3_idx - 2)
            end = min(self.freq_dim, period_3_idx + 3)
            for i in range(start, end):
                self.complex_weight.data[:, i, 0] += 1.0 

        self.norm = nn.LayerNorm(channels)
        self.post_conv = nn.Sequential(
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Conv1d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, L = x.shape
        x_in = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # FFT
        x_fft = torch.fft.rfft(x_in.float(), n=L, dim=-1, norm='ortho') 
        
        # 动态权重插值 (处理变长输入)
        weight = torch.view_as_complex(self.complex_weight)
        if x_fft.shape[-1] != weight.shape[-1]:
            weight = torch.view_as_complex(F.interpolate(
                self.complex_weight.permute(2,0,1).unsqueeze(0), # [1, 2, C, F]
                size=x_fft.shape[-1], mode='linear', align_corners=False
            ).squeeze(0).permute(1,2,0).contiguous()) # back to [C, F, 2]

        # 频域门控
        out_fft = x_fft * weight.unsqueeze(0)
        out = torch.fft.irfft(out_fft, n=L, dim=-1, norm='ortho')
        
        return self.post_conv(out)

# -----------------------------------------------------------------------------
# 优化组件 3: DynamicTopKSparseAttention (支持动态稀疏度课程学习)
# -----------------------------------------------------------------------------
class DynamicTopKSparseAttention(nn.Module):
    def __init__(self, dim, num_heads, max_sparsity=0.85, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.max_sparsity = max_sparsity
        self.current_sparsity = 0.0 # 初始为 0 (全注意力)，防止冷启动失败

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # RoPE 模块
        self.rope = RotaryEmbedding(self.head_dim)

    def set_sparsity(self, ratio):
        """外部调用的方法，用于调整稀疏度"""
        self.current_sparsity = min(ratio, self.max_sparsity)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 应用 RoPE
        cos, sin = self.rope(v, seq_len=N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # --- 动态稀疏化逻辑 ---
        if self.current_sparsity > 0.05:
            k_val = max(int(N * (1 - self.current_sparsity)), 16) # 至少保留 16 个 token
            if k_val < N:
                topk_val, _ = torch.topk(attn, k_val, dim=-1)
                threshold = topk_val[..., -1].unsqueeze(-1)
                mask = attn < threshold
                # 使用 -inf 替换 -1e4 更加安全
                attn = attn.masked_fill(mask, torch.finfo(torch.float32).min)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SparseTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1, max_sparsity=0.85):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DynamicTopKSparseAttention(dim, num_heads, max_sparsity=max_sparsity, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Depthwise Conv 用于增强局部性
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x + self.attn(self.norm1(x))
        
        # MLP path (需要转置处理 Conv1d)
        res = x
        x = self.norm2(x)
        x = self.mlp[0](x) # Linear
        x = self.mlp[1](x) # GELU
        x = self.mlp[2](x) # Drop
        x = x.transpose(1, 2)
        x = self.mlp[3](x) # Conv1d
        x = x.transpose(1, 2)
        x = self.mlp[4](x) # GELU
        x = self.mlp[5](x) # Linear
        x = self.mlp[6](x) # Drop
        return res + x

# -----------------------------------------------------------------------------
# 优化组件 4: 基础块 (GroupNorm 替代 BatchNorm)
# -----------------------------------------------------------------------------
class HybridStem(nn.Module):
    def __init__(self, vocab_size, base_filters, seq_len):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(vocab_size, base_filters, kernel_size=7, padding=3),
            nn.GroupNorm(8, base_filters), # 使用 GroupNorm 以适应小 Batch
            nn.ReLU(inplace=True) 
        )
        self.fft_project = nn.Conv1d(vocab_size, base_filters, 1)
        self.fft_block = BioSpectralGating(base_filters, seq_len=seq_len)
        self.fusion = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters, 1),
            nn.GroupNorm(8, base_filters),
            nn.ReLU(inplace=True)
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
            nn.GroupNorm(8, out_c), # GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c)  # GroupNorm
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_c)
            )
    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))

# -----------------------------------------------------------------------------
# 最终模型: FFTransformer (Optimized)
# -----------------------------------------------------------------------------
class FFTransformer(nn.Module):
    def __init__(self, vocab_size=5, base_filters=64, input_len=10240):
        super().__init__()
        # 计算下采样后的长度
        l1 = input_len
        l2 = l1 // 2
        l3 = l2 // 2
        l4 = l3 // 2

        # 1. Stem
        self.stem = HybridStem(vocab_size, base_filters, seq_len=l1)
        
        # 2. Encoder (Downsampling)
        self.enc1 = ResConvBlock(base_filters, base_filters*2, stride=2)
        self.fft1 = BioSpectralGating(base_filters*2, seq_len=l2)
        
        self.enc2 = ResConvBlock(base_filters*2, base_filters*4, stride=2)
        self.fft2 = BioSpectralGating(base_filters*4, seq_len=l3)
        
        self.enc3 = ResConvBlock(base_filters*4, base_filters*8, stride=2)
        self.fft3 = BioSpectralGating(base_filters*8, seq_len=l4)
        
        self.trans_dim = base_filters * 8
        
        # 3. Bottleneck (Sparse Transformer with RoPE)
        self.sparse_transformer = nn.ModuleList([
            SparseTransformerBlock(
                dim=self.trans_dim, 
                num_heads=8, 
                dropout=0.2,
                max_sparsity=0.85
            )
            for _ in range(4) 
        ])
        
        # 4. Decoder (Upsampling)
        self.up3 = nn.ConvTranspose1d(base_filters*8, base_filters*4, 2, 2)
        self.dec3 = ResConvBlock(base_filters*8, base_filters*4) # cat(up3, c3) -> 4+4=8
        
        self.up2 = nn.ConvTranspose1d(base_filters*4, base_filters*2, 2, 2)
        self.dec2 = ResConvBlock(base_filters*4, base_filters*2) # cat(up2, c2) -> 2+2=4
        
        self.up1 = nn.ConvTranspose1d(base_filters*2, base_filters, 2, 2)
        self.dec1 = ResConvBlock(base_filters*2, base_filters)   # cat(up1, c1) -> 1+1=2
        
        self.final = nn.Conv1d(base_filters, 1, 1)

    def update_sparsity_level(self, epoch, max_epochs=50):
        """
        供训练循环调用的函数:
        前 10% 的 epoch 保持全注意力 (0 sparsity)
        之后线性增加到 0.85
        """
        warmup_epochs = max_epochs * 0.1
        if epoch < warmup_epochs:
            current = 0.0
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            current = progress * 0.85
        
        # 更新每一层的稀疏度
        for layer in self.sparse_transformer:
            layer.attn.set_sparsity(current)
        return current

    def forward(self, x):
        # x: [B, Vocab, L]
        
        # --- Encoder ---
        c1 = self.stem(x)
        
        c2 = self.enc1(c1)
        c2 = c2 + self.fft1(c2)
        
        c3 = self.enc2(c2)
        c3 = c3 + self.fft2(c3)
        
        c4 = self.enc3(c3)
        c4 = c4 + self.fft3(c4)
        
        # --- Transformer Bottleneck ---
        # 转换到 [B, L, C] 
        b = c4.permute(0, 2, 1) 
        
        # 注意：不再需要 self.pos_enc，RoPE 会在 attention 内部处理
        for layer in self.sparse_transformer:
            b = layer(b)
            
        b = b.permute(0, 2, 1) # 回到 [B, C, L]
        
        # --- Decoder (带自动对齐) ---
        u3 = self.up3(b)
        # 对齐 u3 和 c3 的长度 (如果输入是奇数，可能差 1)
        if u3.size(2) != c3.size(2):
            u3 = F.interpolate(u3, size=c3.size(2), mode='linear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))
        
        u2 = self.up2(d3)
        if u2.size(2) != c2.size(2):
            u2 = F.interpolate(u2, size=c2.size(2), mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))
        
        u1 = self.up1(d2)
        if u1.size(2) != c1.size(2):
            u1 = F.interpolate(u1, size=c1.size(2), mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))
        
        return self.final(d1).squeeze(1)

