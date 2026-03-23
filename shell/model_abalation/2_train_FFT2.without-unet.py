import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import math
import os
import numpy as np
import torch.nn.functional as F
import torch.fft 
import sys 
import matplotlib.pyplot as plt

# ================= Configuration =================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    cudnn.benchmark = True
    print(f"[Info] CUDA Accelerated: {torch.cuda.get_device_name(0)}")

# ================= Tools =================

class HistoryLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_f1': [], 'val_rec': [], 
            'val_prec': [], 'val_mcc': []
        }
        with open(self.save_path, 'w') as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_F1\tVal_Recall\tVal_Precision\tVal_MCC\n")

    def log(self, epoch, train_loss, val_loss, val_f1, val_rec, val_prec, val_mcc):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_f1'].append(val_f1)
        self.history['val_rec'].append(val_rec)
        self.history['val_prec'].append(val_prec)
        self.history['val_mcc'].append(val_mcc)

        with open(self.save_path, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}\t{val_mcc:.4f}\n")

    def plot_curves(self, save_path):
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss', color='blue')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss', color='red')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Metrics
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['val_f1'], label='F1 Score', color='green')
        plt.plot(epochs, self.history['val_mcc'], label='MCC', color='orange', linestyle='--')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Info] Plot saved to {save_path}")

# --- 1. Dataset ---
class TensorDataset(Dataset):
    def __init__(self, pt_path):
        print(f"Loading data from {pt_path} into RAM...")
        self.data = torch.load(pt_path, weights_only=False)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item['input'], dtype=torch.long), torch.tensor(item['label'], dtype=torch.float32)

# --- 2. EarlyStopping ---
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pth', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'\n[EarlyStopping] Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0 

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'\n[EarlyStopping] Validation score improved ({self.val_score_max:.6f} --> {val_score:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score

# --- 3. Model Architecture (NO U-NET / NO POOLING) ---

class SpectralFeatureBlock(nn.Module):
    """
    FFT Block for Global Context without Downsampling
    """
    def __init__(self, in_channels, out_channels, seq_len, dropout=0.1):
        super().__init__()
        self.freq_dim = seq_len // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(in_channels, self.freq_dim, 2, dtype=torch.float32) * 0.02
        )
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

        with torch.cuda.amp.autocast(enabled=False):
            x_in = x_in.float()
            x_fft = torch.fft.rfft(x_in, n=L, dim=-1, norm='ortho')
            
            weight = torch.view_as_complex(self.complex_weight)
            # Dynamic resizing if input length differs slightly
            if x_fft.shape[-1] != weight.shape[-1]:
                weight_real = F.interpolate(self.complex_weight[..., 0].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight_imag = F.interpolate(self.complex_weight[..., 1].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight = torch.complex(weight_real.squeeze(0), weight_imag.squeeze(0))
            
            out_fft = x_fft * weight.unsqueeze(0)
            out = torch.fft.irfft(out_fft, n=L, dim=-1, norm='ortho')
        
        out = out.to(x.dtype)
        return self.post_conv(out)

class HybridStem(nn.Module):
    def __init__(self, vocab_size, embed_dim, base_filters, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.conv_branch = nn.Sequential(
            nn.Conv1d(embed_dim, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.fft_project = nn.Conv1d(embed_dim, base_filters, 1)
        self.fft_block = SpectralFeatureBlock(base_filters, base_filters, seq_len=seq_len)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters, 1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_emb = self.embedding(x).permute(0, 2, 1)
        out_conv = self.conv_branch(x_emb)
        out_fft_in = self.fft_project(x_emb)
        out_fft = self.fft_block(out_fft_in)
        combined = torch.cat([out_conv, out_fft], dim=1)
        return self.fusion(combined)

class ResConvBlock(nn.Module):
    """
    Dilated Residual Block
    """
    def __init__(self, in_c, out_c, dilation=1, dropout=0.1):
        super().__init__()
        # Padding must equal dilation to maintain sequence length (Same Padding)
        self.body = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_c)
        )
        
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))

class DeepGenomic_FFT_Transformer_Only(nn.Module):
    """
    消融实验模型：仅保留 FFT + Transformer + CNN 骨架。
    移除了 U-Net 的核心特征 —— Skip Connections (跳跃连接)。
    验证模型是否依赖高分辨率特征融合，还是依靠 FFT/Transformer 的全局能力。
    """
    def __init__(self, vocab_size=5, embed_dim=512, base_filters=64, input_len=10240):
        super().__init__()
        
        # === 1. Encoder (保持不变，为了提取特征) ===
        self.stem = HybridStem(vocab_size, embed_dim, base_filters, seq_len=input_len)
        
        l1 = input_len          
        l2 = l1 // 2            
        l3 = l2 // 2            
        
        self.enc1 = ResConvBlock(base_filters, base_filters*2, stride=2) 
        self.fft_block1 = SpectralFeatureBlock(base_filters*2, base_filters*2, seq_len=l1//2)
        
        self.enc2 = ResConvBlock(base_filters*2, base_filters*4, stride=2) 
        self.fft_block2 = SpectralFeatureBlock(base_filters*4, base_filters*4, seq_len=l2//2)
        
        self.enc3 = ResConvBlock(base_filters*4, base_filters*8, stride=2) 
        self.fft_block3 = SpectralFeatureBlock(base_filters*8, base_filters*8, seq_len=l3//2)
        
        # === 2. Bottleneck (Transformer 保持不变) ===
        self.trans_dim = base_filters * 8
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.trans_dim, nhead=8, dim_feedforward=2048, 
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6) 
        
        max_len = 12000 
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, self.trans_dim) * 0.02)
        
        # === 3. Decoder (修改部分：移除 Skip Connection 的通道拼接) ===
        
        # Layer 3 Upsample
        # 输入: base_filters*8 (Trans输出) -> 输出: base_filters*4
        self.up3 = nn.ConvTranspose1d(base_filters*8, base_filters*4, 2, 2)
        # 注意：这里输入不再是 *8 (因为没有 concat c3)，而是 *4
        self.dec3 = ResConvBlock(base_filters*4, base_filters*4) 
        
        # Layer 2 Upsample
        # 输入: base_filters*4 -> 输出: base_filters*2
        self.up2 = nn.ConvTranspose1d(base_filters*4, base_filters*2, 2, 2)
        # 注意：这里输入不再是 *4 (因为没有 concat c2)，而是 *2
        self.dec2 = ResConvBlock(base_filters*2, base_filters*2)
        
        # Layer 1 Upsample
        # 输入: base_filters*2 -> 输出: base_filters
        self.up1 = nn.ConvTranspose1d(base_filters*2, base_filters, 2, 2)
        # 注意：这里输入不再是 *2 (因为没有 concat c1)，而是 base_filters
        self.dec1 = ResConvBlock(base_filters, base_filters)
        
        self.final = nn.Conv1d(base_filters, 1, 1)

    def forward(self, x):
        # === Encoder Pass (依然计算，但不保存用于 Skip) ===
        c1 = self.stem(x)
        
        c2_cnn = self.enc1(c1)
        c2_fft = self.fft_block1(c2_cnn)
        c2 = c2_cnn + c2_fft
        
        c3_cnn = self.enc2(c2)
        c3_fft = self.fft_block2(c3_cnn)
        c3 = c3_cnn + c3_fft
        
        c4_cnn = self.enc3(c3)
        c4_fft = self.fft_block3(c4_cnn)
        c4 = c4_cnn + c4_fft
        
        # === Transformer Pass ===
        b = c4.permute(0, 2, 1)
        seq_len = b.size(1)
        if seq_len > self.pos_enc.size(1):
             pos_emb = F.interpolate(self.pos_enc.permute(0,2,1), size=seq_len).permute(0,2,1)
        else:
             pos_emb = self.pos_enc[:, :seq_len, :]
        b = b + pos_emb
        b = self.transformer(b)
        b = b.permute(0, 2, 1)
        
        # === Decoder Pass (直连，无 Concat) ===
        
        # Block 3
        d3_up = self.up3(b)
        # 原版: torch.cat([d3_up, c3], dim=1) -> 现在: d3_up
        d3 = self.dec3(d3_up) 
        
        # Block 2
        d2_up = self.up2(d3)
        # 原版: torch.cat([d2_up, c2], dim=1) -> 现在: d2_up
        d2 = self.dec2(d2_up)
        
        # Block 1
        d1_up = self.up1(d2)
        # 原版: torch.cat([d1_up, c1], dim=1) -> 现在: d1_up
        d1 = self.dec1(d1_up)
        
        return self.final(d1).squeeze(1)

# --- 4. Loss Function ---
class DiceFocalLoss(nn.Module):
    def __init__(self, pos_weight_val, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = torch.tensor([pos_weight_val]).to(DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.weight, reduction='none')
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        logits = torch.clamp(logits, min=-100, max=100) 
        probas = torch.sigmoid(logits)
        
        p_t = targets * probas + (1 - targets) * (1 - probas)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0) 
        focal_loss = (self.alpha * (1 - p_t)**self.gamma * bce_loss).mean()
        
        smooth = 1e-5
        probs_flat = probas.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        
        return 0.5 * focal_loss + 0.5 * dice_loss

class PatchEmbedding(nn.Module):
    """
    类似于 ViT 的 Patch Embedding，或者 NLP 的 Tokenizer。
    作用：用一个简单的卷积层，将原始 DNA 序列直接投影到高维空间，并压缩长度。
    彻底取代 U-Net 的多层 Encoder。
    """
    def __init__(self, vocab_size, embed_dim, patch_size=8):
        super().__init__()
        # 使用 stride=patch_size 实现无重叠切分，或者 stride < patch_size 实现重叠切分
        # 这里为了最大程度模拟“无卷积特征提取”，我们用大步长直接压缩
        self.proj = nn.Conv1d(vocab_size, embed_dim, kernel_size=patch_size+4, stride=patch_size, padding=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [Batch, Length] -> one_hot/embedding -> [Batch, Channel, Length]
        # 注意：这里我们直接处理 one-hot 或者简单的 embedding 后的 transpose
        # 为了兼容你之前的 Dataset (输出是 Long Tensor)，我们先做一个简单的 Embedding
        x = self.proj(x) # [B, Dim, L/patch_size]
        x = x.permute(0, 2, 1) # [B, L', Dim]
        x = self.norm(x)
        return x.permute(0, 2, 1) # [B, Dim, L']

class PureFFTransformer(nn.Module):
    """
    [Extreme Ablation] 纯 FFT + Transformer 架构
    1. 移除了 U-Net 的所有层级结构 (Encoder/Decoder/Skips)。
    2. 移除了 ResConvBlock (卷积特征提取)。
    3. 结构：Input -> PatchEmbed -> FFT -> Transformer -> LinearUpsample -> Output
    """
    def __init__(self, vocab_size=5, embed_dim=256, input_len=102400, patch_size=8):
        super().__init__()
        
        # 1. Patch Embedding (替代 Stem 和 Encoder)
        # 将 100kb 序列压缩 8 倍 -> 12800 长度 (显存极限)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 这里的 patch_proj 负责将 embedding 后的特征聚合压缩
        self.patch_proj = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        
        seq_len_compressed = input_len // patch_size
        
        # 2. FFT 模块 (放在 Transformer 之前，提取全局频域特征)
        self.fft_block = SpectralFeatureBlock(embed_dim, embed_dim, seq_len=seq_len_compressed)
        
        # 3. Transformer (核心处理单元)
        # 增加层数以弥补移除 CNN 的损失
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=1024, 
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6) 
        
        # 位置编码
        self.pos_enc = nn.Parameter(torch.randn(1, 15000, embed_dim) * 0.02)
        
        # 4. Upsampling Head (简单还原分辨率)
        # 使用转置卷积直接将 12800 长度放大回 102400
        self.upsample = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.final_head = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3), # 稍微平滑一下 Patch 边界效应
            nn.GELU(),
            nn.Conv1d(64, 1, 1)
        )

    def forward(self, x):
        # x: [Batch, Length]
        
        # === 1. Tokenization / Patching ===
        x = self.embedding(x).permute(0, 2, 1) # [B, C, L]
        x = self.patch_proj(x)                 # [B, C, L/8]
        
        # === 2. Spectral Feature (FFT) ===
        x_fft = self.fft_block(x)
        x = x + x_fft # 残差连接
        
        # === 3. Transformer ===
        x = x.permute(0, 2, 1) # [B, L/8, C]
        
        # Add Positional Encoding
        seq_len = x.size(1)
        if seq_len > self.pos_enc.size(1):
             pos_emb = F.interpolate(self.pos_enc.permute(0,2,1), size=seq_len).permute(0,2,1)
        else:
             pos_emb = self.pos_enc[:, :seq_len, :]
        x = x + pos_emb
        
        x = self.transformer(x)
        x = x.permute(0, 2, 1) # [B, C, L/8]
        
        # === 4. Upsample & Prediction ===
        x = self.upsample(x) # [B, C, L]
        out = self.final_head(x)
        
        return out.squeeze(1)

def compute_metrics(logits, targets):
    if torch.isnan(logits).any():
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    preds = (torch.sigmoid(logits) > 0.5).float().view(-1)
    targets = targets.view(-1)
    
    tp = (preds * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item() 
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / (denominator + 1e-8)
    
    return f1, rec, prec, acc, mcc

# --- 5. Main Training Loop ---
def main():
    # ================= PATHS =================
    TRAIN_PATH = "/data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/model_test/final_out/D_only/DataSet/MT_10kb/train_dataset.pt"
    VAL_PATH = "/data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/model_test/final_out/D_only/DataSet/MT_10kb/val_dataset.pt"
    BASE_DIR = "/data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/model_test/final_out/D_only/DataSet/MT_10kb"

    # ================= HYPERPARAMETERS =================
    BATCH_SIZE = 8            # Reduced batch size due to high resolution
    LEARNING_RATE = 3e-5      # Slightly lower LR for stability
    EPOCHS = 500              
    ACCUMULATION_STEPS = 4    # Increased accumulation to simulate larger batch
    POS_WEIGHT = 2       
    PATIENCE = 10             
    INPUT_SEQ_LEN = 102400

    loss_name = "DiceFocalLoss"
    seq_length_name = f'Nounet_{INPUT_SEQ_LEN//1000}kb'
    print(f"Config: {seq_length_name}, Loss: {loss_name}")

    MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"{seq_length_name}.pth")
    LOG_SAVE_PATH = os.path.join(BASE_DIR, f"{seq_length_name}_log.txt")
    PLOT_SAVE_PATH = os.path.join(BASE_DIR, f"{seq_length_name}_plot.png")

    

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader = DataLoader(
        TensorDataset(TRAIN_PATH), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(VAL_PATH), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    logger = HistoryLogger(LOG_SAVE_PATH)
    
    # [MODIFIED] Initialize No-Pooling Model
    vocab_size=5
    embed_dim=64
    base_filters=64 # Adjusted for VRAM safety
    
    print(f"---------> [Ablation] Pure FFT + Transformer  <---------")
    model = PureFFTransformer(vocab_size=vocab_size,
                              embed_dim=256,  # 稍微增大维度，因为没有多层 CNN 了
                              input_len=INPUT_SEQ_LEN,
                              patch_size=8).to(DEVICE)
    
    print(f"[Info] Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    criterion = DiceFocalLoss(pos_weight_val=POS_WEIGHT)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_SAVE_PATH, verbose=True)
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            current_lr = optimizer.param_groups[0]['lr']
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [LR={current_lr:.1e}]")
            
            valid_batch_count = 0

            for i, (x, y) in enumerate(loop):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                
                with autocast():
                    out = model(x)
                    loss = criterion(out, y)
                    loss = loss / ACCUMULATION_STEPS 
                
                if torch.isnan(loss):
                    print(f"\n[Error] NaN detected in loss at epoch {epoch}, batch {i}!")
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * ACCUMULATION_STEPS
                valid_batch_count += 1
                loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
                
            avg_train_loss = train_loss / valid_batch_count if valid_batch_count > 0 else 0.0

            # Validation
            model.eval()
            val_loss, val_f1, val_rec, val_prec, val_acc, val_mcc = 0, 0, 0, 0, 0, 0
            steps = 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                    with autocast():
                        out = model(x)
                        loss_val = criterion(out, y)
                        if torch.isnan(loss_val): continue
                        val_loss += loss_val.item()
                    
                    f1, r, p, a, m = compute_metrics(out.float(), y.float())
                    
                    val_f1 += f1
                    val_rec += r
                    val_prec += p
                    val_acc += a
                    val_mcc += m
                    steps += 1
            
            if steps == 0: steps = 1
            avg_val_loss = val_loss / steps
            avg_f1 = val_f1 / steps
            avg_rec = val_rec / steps
            avg_prec = val_prec / steps
            avg_acc = val_acc / steps
            avg_mcc = val_mcc / steps
            
            print(f"\n[Epoch {epoch+1} Report]")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Val F1: {avg_f1:.4f} | Recall: {avg_rec:.4f} | Precision: {avg_prec:.4f}")
            print(f"Val ACC: {avg_acc:.4f} | Val MCC: {avg_mcc:.4f}")
            
            logger.log(epoch+1, avg_train_loss, avg_val_loss, avg_f1, avg_rec, avg_prec, avg_mcc)
            
            scheduler.step(avg_f1)
            
            early_stopping(avg_f1, model)
            if early_stopping.early_stop:
                print(f"\n[Info] Early Stopping triggered! Best F1: {early_stopping.val_score_max:.4f}")
                break
    
    except KeyboardInterrupt:
        print("\n[Info] Training interrupted. Saving current logs and plots...")

    logger.plot_curves(PLOT_SAVE_PATH)
    print(f"\nTraining Finished. Best model saved at: {MODEL_SAVE_PATH}")
    print(f"Logs saved at: {LOG_SAVE_PATH}")
    print(f"Plots saved at: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()