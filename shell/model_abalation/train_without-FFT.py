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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    cudnn.benchmark = True
    print(f"[Info] CUDA Accelerated: {torch.cuda.get_device_name(0)}")


class HistoryLogger:
    """
    [Added] Tool to track metrics and save them to file/plots
    """
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
        """Draw and save training curves"""
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


class TensorDataset(Dataset):
    def __init__(self, pt_path):
        print(f"Loading data from {pt_path} into RAM...")
        self.data = torch.load(pt_path, weights_only=False)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item['input'], dtype=torch.long), torch.tensor(item['label'], dtype=torch.float32)

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

# --- 3. Model Architecture ---

class SpectralFeatureBlock(nn.Module):
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
            if x_fft.shape[-1] != weight.shape[-1]:
                weight_real = F.interpolate(self.complex_weight[..., 0].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight_imag = F.interpolate(self.complex_weight[..., 1].unsqueeze(0), size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight = torch.complex(weight_real.squeeze(0), weight_imag.squeeze(0))
            
            out_fft = x_fft * weight.unsqueeze(0)
            out = torch.fft.irfft(out_fft, n=L, dim=-1, norm='ortho')
        
        out = out.to(x.dtype)
        return self.post_conv(out)

class HybridStem_NoFFT(nn.Module):
    """
    消融版本的 Stem：移除了 FFT 分支，仅保留 Conv 分支。
    """
    def __init__(self, vocab_size, embed_dim, base_filters):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.conv_branch = nn.Sequential(
            nn.Conv1d(embed_dim, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv1d(base_filters, base_filters, 1), 
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_emb = self.embedding(x).permute(0, 2, 1)
        out_conv = self.conv_branch(x_emb)
        return self.fusion(out_conv)

class ResConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
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

class Panno_NoFFT(nn.Module):
    """
    消融实验模型：完全移除 SpectralFeatureBlock。
    用于验证频域特征提取对 sORF 预测的重要性。
    """
    def __init__(self, vocab_size=5, embed_dim=512, base_filters=64, input_len=10240):
        super().__init__()

        self.stem = HybridStem_NoFFT(vocab_size, embed_dim, base_filters)
        
        l1 = input_len          
        l2 = l1 // 2            
        l3 = l2 // 2            

        self.enc1 = ResConvBlock(base_filters, base_filters*2, stride=2) 

        self.enc2 = ResConvBlock(base_filters*2, base_filters*4, stride=2) 

        self.enc3 = ResConvBlock(base_filters*4, base_filters*8, stride=2) 

        self.trans_dim = base_filters * 8
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.trans_dim, nhead=8, dim_feedforward=2048, 
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6) 
        
        max_len = 12000 
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, self.trans_dim) * 0.02)
        
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
        
        c3 = self.enc2(c2)
        
        c4 = self.enc3(c3)
        
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

class FocalLoss(nn.Module):
    """
    适用于: 存在大量简单背景(Easy Negatives)，需要挖掘难例(Hard Mining)。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        pt = torch.exp(-bce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    阶段 4: Dice Loss
    适用于: 关注分割区域的整体重叠度，对不平衡极其鲁棒。
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        if targets.dim() > 1:
            probs = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        Tversky Loss 实现
        参数:
            alpha (float): False Positive (FP) 的权重，控制假阳性。
            beta (float): False Negative (FN) 的权重，控制假阴性 (漏报)。
            smooth (float): 平滑项，防止分母为 0。
        注意:
            alpha + beta 通常等于 1.0 (但不强制)。
            如果 beta > alpha，模型会倾向于提高 Recall (召回率)。
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        batch_size = targets.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        true_positives = (probs * targets).sum(dim=1)

        false_positives = (probs * (1 - targets)).sum(dim=1)

        false_negatives = ((1 - probs) * targets).sum(dim=1)

        numerator = true_positives
        denominator = true_positives + (self.alpha * false_positives) + (self.beta * false_negatives)
        
        tversky_index = (numerator + self.smooth) / (denominator + self.smooth)

        return 1 - tversky_index.mean()


class BoundaryAwareFocalLoss(nn.Module):
    def __init__(self, alpha=0.90, gamma=2.0, boundary_weight=20.0, internal_weight=5.0, codon_len=3):
        super(BoundaryAwareFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.internal_weight = internal_weight
        self.codon_len = codon_len
        print(f"[Loss Init] Alpha={alpha}, Gamma={gamma}, Boundary W={boundary_weight}, Internal W={internal_weight}")

    def generate_weight_map(self, targets):
        """生成权重图 (保持你的原有逻辑)"""

        weight_map = torch.ones_like(targets, dtype=torch.float32)

        weight_map = torch.where(targets > 0.5, 
                                 torch.tensor(self.internal_weight, device=targets.device), 
                                 weight_map)

        padded_targets = F.pad(targets, (1, 1), mode='constant', value=0)
        diff = padded_targets[:, 1:] - padded_targets[:, :-1]
        diff = diff[:, :-1] 

        starts = (diff == 1).float()
        ends = (diff == -1).float()

        start_mask = torch.zeros_like(targets)
        end_mask = torch.zeros_like(targets)
        
        for k in range(self.codon_len):

            s_s = torch.roll(starts, shifts=k, dims=1)
            s_s[:, :k] = 0
            start_mask = torch.max(start_mask, s_s)

            e_s = torch.roll(ends, shifts=-(k+1), dims=1)
            e_s[:, -(k+1):] = 0
            end_mask = torch.max(end_mask, e_s)
            
        boundary_mask = torch.max(start_mask, end_mask)

        weight_map = torch.where(boundary_mask > 0.5, 
                                 torch.tensor(self.boundary_weight, device=targets.device), 
                                 weight_map)
        
        return weight_map

    def forward(self, logits, targets):

        logits = logits.float() 
        targets = targets.float()

        if logits.dim() == 3: logits = logits.squeeze(1)
        if targets.dim() == 3: targets = targets.squeeze(1)

        logits = torch.clamp(logits, min=-100, max=100)

        with torch.no_grad():
            pixel_weights = self.generate_weight_map(targets)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        pt = torch.exp(-bce_loss)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0
            

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        weighted_loss = focal_loss * pixel_weights
        
        loss = weighted_loss.mean()
        
        return loss


class ComprehensiveBoundaryLoss(nn.Module):
    def __init__(self, 
                 # Focal / Boundary 参数
                 alpha=0.90, 
                 gamma=2.0, 
                 boundary_weight=20.0, 
                 internal_weight=5.0, 
                 codon_len=3,
                 # 组合权重参数
                 lambda_boundary=1.0,  # 边界感知 Focal 的权重
                 lambda_dice=0.5       # Dice Loss 的权重
                 ):
        """
        ComprehensiveBoundaryLoss: 综合了边界感知与全局形状优化的终极损失函数。
        
        参数详解:
            alpha (float): Focal Loss 正样本平衡参数。建议 0.90 (强力提升 Recall)。
            gamma (float): Focal Loss 聚焦参数。建议 2.0。
            boundary_weight (float): 边界区域(ATG/Stop)的惩罚倍数。建议 20.0。
            internal_weight (float): sORF 内部区域的惩罚倍数。建议 5.0。
            lambda_boundary (float): BoundaryAwareFocalLoss 在总 Loss 中的占比权重。
            lambda_dice (float): Dice Loss 在总 Loss 中的占比权重。
        """
        super(ComprehensiveBoundaryLoss, self).__init__()
        # 核心参数
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.internal_weight = internal_weight
        self.codon_len = codon_len
        
        # 组合权重
        self.lambda_boundary = lambda_boundary
        self.lambda_dice = lambda_dice
        
        print(f"[Loss Init] Comprehensive Mode: Boundary(x{lambda_boundary}) + Dice(x{lambda_dice})")
        print(f"            Details: Alpha={alpha}, BoundW={boundary_weight}, InternW={internal_weight}")

    def generate_weight_map(self, targets):
        """
        生成空间权重图: 
        - 背景: 1.0
        - 内部: internal_weight
        - 边界: boundary_weight
        """

        weight_map = torch.ones_like(targets, dtype=torch.float32)

        weight_map = torch.where(targets > 0.5, 
                                 torch.tensor(self.internal_weight, device=targets.device), 
                                 weight_map)

        padded_targets = F.pad(targets, (1, 1), mode='constant', value=0)
        diff = padded_targets[:, 1:] - padded_targets[:, :-1]
        diff = diff[:, :-1] 

        starts = (diff == 1).float() 
        ends = (diff == -1).float() 

        start_mask = torch.zeros_like(targets)
        end_mask = torch.zeros_like(targets)
        
        for k in range(self.codon_len):

            s_s = torch.roll(starts, shifts=k, dims=1)
            s_s[:, :k] = 0
            start_mask = torch.max(start_mask, s_s)

            e_s = torch.roll(ends, shifts=-(k+1), dims=1)
            e_s[:, -(k+1):] = 0
            end_mask = torch.max(end_mask, e_s)
            
        boundary_mask = torch.max(start_mask, end_mask)
        

        weight_map = torch.where(boundary_mask > 0.5, 
                                 torch.tensor(self.boundary_weight, device=targets.device), 
                                 weight_map)
        return weight_map

    def forward(self, logits, targets):
        """
        logits: 模型输出 [Batch, Length] (未经过 Sigmoid)
        targets: 真实标签 [Batch, Length]
        """

        logits = logits.float()   
        targets = targets.float()
        
        # 维度对齐
        if logits.dim() == 3: logits = logits.squeeze(1)
        if targets.dim() == 3: targets = targets.squeeze(1)

        logits = torch.clamp(logits, min=-100, max=100)

        with torch.no_grad():
            pixel_weights = self.generate_weight_map(targets)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # C. Alpha 平衡
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0
            
        focal_term = pixel_weights * alpha_t * (1 - pt) ** self.gamma * bce_loss
        boundary_loss = focal_term.mean()

        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        smooth = 1e-6
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score

        total_loss = self.lambda_boundary * boundary_loss + self.lambda_dice * dice_loss
        
        return total_loss
    

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

def main():

    TRAIN_PATH = "/data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/model_test/final_out/D_only/DataSet/MT_10kb/train_dataset.pt"
    VAL_PATH = "/data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/model_test/final_out/D_only/DataSet/MT_10kb/val_dataset.pt"

    BASE_DIR = "/data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/model_test/final_out/D_only/DataSet/MT_10kb"

    BATCH_SIZE = 8            
    LEARNING_RATE = 3e-5      
    EPOCHS = 500              
    ACCUMULATION_STEPS = 4  
    POS_WEIGHT = 2       
    PATIENCE = 10            
    INPUT_SEQ_LEN = 71680

    loss_name = "DiceFocalLoss"
    seq_length = f'{INPUT_SEQ_LEN//1000}kb'
    print(f"长度是{seq_length}")

    suffix =  "NoFFT"
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"{suffix}_{seq_length}_.pth")
    LOG_SAVE_PATH = os.path.join(BASE_DIR, f"{suffix}_{seq_length}_log.txt")
    PLOT_SAVE_PATH = os.path.join(BASE_DIR, f"{suffix}_{seq_length}_plot.png")

    

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
    
    # Initialize Logger
    logger = HistoryLogger(LOG_SAVE_PATH)
    
    vocab_size=5
    embed_dim=64
    base_filters=64
    model = Panno_NoFFT(vocab_size=vocab_size,
                                           embed_dim=embed_dim, 
                                           base_filters=base_filters,
                                           input_len=INPUT_SEQ_LEN).to(DEVICE)
    print(f"[Info] Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    
    print(f"=================现在使用的是  {loss_name}===============================================")
    
    criterion = DiceFocalLoss(pos_weight_val=POS_WEIGHT)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE)
    # criterion = FocalLoss(alpha=0.8, gamma=2.0)
    # criterion = DiceLoss()
    # criterion = TverskyLoss(alpha=0.3, beta=0.7) 
    #criterion = DiceFocalLoss(pos_weight_val=8)
    #criterion = DiceFocalLoss_optimize(alpha=0.8, gamma=2.0, weight_dice=0.4, weight_focal=0.7)
    #criterion = ComboLoss(alpha=0.2, beta=0.8, focal_gamma=2.0)
    
    # criterion = BoundaryAwareFocalLoss(alpha=0.90, gamma=2.0, boundary_weight=10.0, internal_weight=5.0)
    # criterion = ComprehensiveBoundaryLoss(
    #     alpha=0.60, 
    #     gamma=1.5, 
    #     boundary_weight=2, 
    #     internal_weight=1,
    #     lambda_boundary=1.0, 
    #     lambda_dice=0.7
    # ).to(DEVICE)

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
            
            # Print Report
            print(f"\n[Epoch {epoch+1} Report]")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Val F1: {avg_f1:.4f} | Recall: {avg_rec:.4f} | Precision: {avg_prec:.4f}")
            print(f"Val ACC: {avg_acc:.4f} | Val MCC: {avg_mcc:.4f}")
            
            # [Added] Log to file
            logger.log(epoch+1, avg_train_loss, avg_val_loss, avg_f1, avg_rec, avg_prec, avg_mcc)
            
            # Scheduler & Early Stopping
            scheduler.step(avg_f1)
            
            early_stopping(avg_f1, model)
            if early_stopping.early_stop:
                print(f"\n[Info] Early Stopping triggered! Best F1: {early_stopping.val_score_max:.4f}")
                break
    
    except KeyboardInterrupt:
        print("\n[Info] Training interrupted. Saving current logs and plots...")

    # [Added] Plot at the end
    logger.plot_curves(PLOT_SAVE_PATH)
    print(f"\nTraining Finished. Best model saved at: {MODEL_SAVE_PATH}")
    print(f"Logs saved at: {LOG_SAVE_PATH}")
    print(f"Plots saved at: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()