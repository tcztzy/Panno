import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class TopNMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, top_n=1280, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_n = top_n
        self.head_dim = embed_dim // num_heads
        
        self.batch_first = True 
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, **kwargs):
        B, L, E = query.shape
        H = self.num_heads
        
        qkv = self.qkv_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, L, H, self.head_dim).transpose(1, 2) 
        k = k.view(B, L, H, self.head_dim).transpose(1, 2)
        v = v.view(B, L, H, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        
        if self.top_n < L:
            topk_vals, _ = torch.topk(scores, self.top_n, dim=-1)
            threshold = topk_vals[..., -1:] 
            mask = scores < threshold
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v) 
        out = out.transpose(1, 2).reshape(B, L, E)
        
        return self.out_proj(out), None

class TopNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, top_n=1280, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = TopNMultiheadAttention(d_model, nhead, top_n=top_n, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        # 严格对齐 PyTorch 官方 TransformerEncoderLayer 的 Xavier 策略
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, **kwargs):
        src2 = self.norm1(src)
        attn_out, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(attn_out)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class SpectralFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, dropout=0.1):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(in_channels, self.freq_dim, 2, dtype=torch.float32) * 0.02)
        self.norm = nn.LayerNorm(in_channels)
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
    def __init__(self, vocab_size, base_filters, seq_len, dropout=0.1):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(vocab_size, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout) 
        )
        self.fft_project = nn.Conv1d(vocab_size, base_filters, 1)
        self.fft_block = SpectralFeatureBlock(base_filters, base_filters, seq_len=seq_len, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters, 1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out_conv = self.conv_branch(x)
        out_fft_in = self.fft_project(x)
        out_fft = self.fft_block(out_fft_in)
        return self.fusion(torch.cat([out_conv, out_fft], dim=1))

class ResConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout), 
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c)
        )
        self.activation = nn.ReLU(inplace=False)
        self.dropout_final = nn.Dropout(dropout) 
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        out = self.body(x)
        out = self.dropout_final(out) 
        out += self.shortcut(x)
        return self.activation(out)

class Panno(nn.Module):
    def __init__(self, vocab_size=5, base_filters=64, input_len=10240, dropout_rate=0.2, top_n=1280):
        super().__init__()
        
        self.stem = HybridStem(vocab_size, base_filters, seq_len=input_len, dropout=dropout_rate)
        
        l1=input_len; l2=l1//2; l3=l2//2
        
        self.enc1 = ResConvBlock(base_filters, base_filters*2, stride=2, dropout=dropout_rate)
        self.fft_block1 = SpectralFeatureBlock(base_filters*2, base_filters*2, seq_len=l1//2, dropout=dropout_rate)
        
        self.enc2 = ResConvBlock(base_filters*2, base_filters*4, stride=2, dropout=dropout_rate)
        self.fft_block2 = SpectralFeatureBlock(base_filters*4, base_filters*4, seq_len=l2//2, dropout=dropout_rate)
        
        self.enc3 = ResConvBlock(base_filters*4, base_filters*8, stride=2, dropout=dropout_rate)
        self.fft_block3 = SpectralFeatureBlock(base_filters*8, base_filters*8, seq_len=l3//2, dropout=dropout_rate)
        
        self.trans_dim = base_filters * 8

        encoder_layer = TopNTransformerEncoderLayer(
            d_model=self.trans_dim, 
            nhead=8, 
            top_n=top_n, 
            dim_feedforward=2048, 
            dropout=dropout_rate
        )

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.trans_dim, 
        #     nhead=8, 
        #     dim_feedforward=2048, 
        #     dropout=dropout_rate, 
        #     activation='gelu', 
        #     batch_first=True, 
        #     norm_first=True
        # )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6) 
        self.pos_enc = nn.Parameter(torch.randn(1, 12000, self.trans_dim) * 0.02)
        
        self.up3 = nn.ConvTranspose1d(base_filters*8, base_filters*4, 2, 2)
        self.dec3 = ResConvBlock(base_filters*8, base_filters*4, dropout=dropout_rate)
        
        self.up2 = nn.ConvTranspose1d(base_filters*4, base_filters*2, 2, 2)
        self.dec2 = ResConvBlock(base_filters*4, base_filters*2, dropout=dropout_rate)
        
        self.up1 = nn.ConvTranspose1d(base_filters*2, base_filters, 2, 2)
        self.dec1 = ResConvBlock(base_filters + base_filters, base_filters, dropout=dropout_rate)

        self.final_dropout = nn.Dropout(dropout_rate) 
        self.final = nn.Conv1d(base_filters, 1, 1)

    def forward(self, x):
        c1 = self.stem(x)
        
        c2 = self.enc1(c1)
        c2_fft = self.fft_block1(c2)
        c2 = c2 + c2_fft
        
        c3 = self.enc2(c2)
        c3_fft = self.fft_block2(c3)
        c3 = c3 + c3_fft
        
        c4 = self.enc3(c3)
        c4_fft = self.fft_block3(c4)
        c4 = c4 + c4_fft

        b = c4.permute(0, 2, 1) # (B, L, C)
        seq_len = b.size(1)
        if seq_len > self.pos_enc.size(1):
             pos_emb = F.interpolate(self.pos_enc.permute(0,2,1), size=seq_len).permute(0,2,1)
        else:
             pos_emb = self.pos_enc[:, :seq_len, :]
        b = b + pos_emb
        
        # 你的 forward 没有任何改变，直接运行
        b = self.transformer(b)
        
        b = b.permute(0, 2, 1) # (B, C, L)
        
        d3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))

        out = self.final_dropout(d1) 
        return self.final(out).squeeze(1)

# # ==========================================
# # 快速测试脚本（你可以直接运行这整个文件试试）
# # ==========================================
# if __name__ == "__main__":
#     # 创建模型，默认参数
#     model = Panno(vocab_size=5, input_len=10240, top_n=64).cuda()
    
#     # 创建一个模拟输入张量：Batch=2, Vocab=5, Length=10240
#     dummy_input = torch.randn(2, 5, 10240).cuda()
    
#     # 前向传播测试
#     out = model(dummy_input)
    
#     print("模型运行成功！")
#     print(f"输入尺寸: {dummy_input.shape}")
#     print(f"输出尺寸: {out.shape}")