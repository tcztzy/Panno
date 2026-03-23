import torch
from utils.sparseTransformer import FFTransformer  
from utils.visualize import plot_attention_profile , plot_sparse_attention
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
import seaborn as sns
from sklearn.metrics import confusion_matrix

"top-k"
BASE_PATH = "/data/user_home/2023122004/FFTransformer/D"
TRAIN_PATH = f"{BASE_PATH}/train_dataset.pt"
VAL_PATH = f"{BASE_PATH}/val_dataset.pt"
MODEL_SAVE_PATH = f"{BASE_PATH}/top-k.pth"
CONFUSION_MATRIX_PATH = f"{BASE_PATH}/top-k.pdf" # 新增：混淆矩阵保存路径
ATTENTION_MAP_PATH = f"{BASE_PATH}/sparse_attention_map.pdf"

BATCH_SIZE = 16              
LEARNING_RATE = 1e-4      
EPOCHS = 500            
ACCUMULATION_STEPS = 4  
POS_WEIGHT = 2       
PATIENCE = 15             
INPUT_SEQ_LEN = 10240    
VOCAB_SIZE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    cudnn.benchmark = True
    print(f"[Info] CUDA Accelerated: {torch.cuda.get_device_name(0)}")

class TensorDataset(Dataset):
    def __init__(self, pt_path, vocab_size=5):
        print(f"Loading data from {pt_path} into RAM...")
        self.data = torch.load(pt_path, weights_only=False)
        self.vocab_size = vocab_size

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq_int = torch.tensor(item['input'], dtype=torch.long)
        seq_onehot = F.one_hot(seq_int, num_classes=self.vocab_size).permute(1, 0).float()
        return seq_onehot, torch.tensor(item['label'], dtype=torch.float32)

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

def plot_confusion_matrix(tp, tn, fp, fn, save_path):

    cm = np.array([[tn, fp], [fn, tp]])

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    cm_norm = cm / row_sums

    group_names = ['TN', 'FP', 'FN', 'TP']

    group_percentages = ["{0:.2%}".format(value) for value in cm_norm.flatten()]
    labels_norm = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels_norm = np.asarray(labels_norm).reshape(2,2)

    axis_labels = ["0", "1"]

    fig, ax = plt.subplots(figsize=(10, 8)) 

    sns.heatmap(cm_norm, annot=labels_norm, fmt='', cmap='PuBu', ax=ax, 
                xticklabels=axis_labels, yticklabels=axis_labels, 
                annot_kws={"size": 18, "weight": "bold"}, cbar=True)
    
    ax.set_title("Confused Matrix", fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    plt.tight_layout()

    plt.savefig(save_path, dpi=1000, bbox_inches='tight') 
    plt.close()
    print(f"   [Plot] Confusion Matrix saved to: {save_path}")

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

def compute_raw_counts(logits, targets):
    """
    计算 TP, TN, FP, FN 的原始数量
    """
    if torch.isnan(logits).any():
        return 0, 0, 0, 0
        
    preds = (torch.sigmoid(logits) > 0.5).float().view(-1)
    targets = targets.view(-1)
    
    tp = (preds * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item() 
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    return tp, tn, fp, fn

def calculate_advanced_metrics(tp, tn, fp, fn):
    """
    基于累计的统计量计算高级指标
    """
    smooth = 1e-8
    
    # Precision (查准率)
    prec = tp / (tp + fp + smooth)
    # Recall (查全率 / Sensitivity)
    rec = tp / (tp + fn + smooth)
    # Specificity (特异性 - 衡量对背景的识别能力)
    spec = tn / (tn + fp + smooth)
    # F1 Score
    f1 = 2 * (prec * rec) / (prec + rec + smooth)
    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn + smooth)
    # IoU (Intersection over Union) - 分割任务的核心指标
    iou = tp / (tp + fp + fn + smooth)
    # MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / (denominator + smooth)
    
    return f1, rec, prec, acc, mcc, iou, spec

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_ds = TensorDataset(TRAIN_PATH, vocab_size=VOCAB_SIZE)
    val_ds = TensorDataset(VAL_PATH, vocab_size=VOCAB_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    base_filters = 64
    model = FFTransformer(vocab_size=VOCAB_SIZE, base_filters=base_filters, input_len=INPUT_SEQ_LEN).to(DEVICE)
    print(f"[Info] Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    criterion = DiceFocalLoss(pos_weight_val=POS_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_SAVE_PATH, verbose=True)
    
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

        model.eval()
        val_loss = 0
        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
        steps = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast():
                    out = model(x)
                    loss_val = criterion(out, y)
                    if torch.isnan(loss_val): continue
                    val_loss += loss_val.item()

                tp, tn, fp, fn = compute_raw_counts(out.float(), y.float())
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                steps += 1
        
        if steps == 0: steps = 1
        avg_val_loss = val_loss / steps

        f1, rec, prec, acc, mcc, iou, spec = calculate_advanced_metrics(total_tp, total_tn, total_fp, total_fn)
        
        print(f"\n[Epoch {epoch+1}]")
        print(f"Loss: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")
        print("tp",total_tp,"tn",total_tn,"fp",total_tn,"fn",total_fn)
        print(f"Metrics: F1 {f1:.4f} | IoU {iou:.4f} | MCC {mcc:.4f}")
        print(f"Details: Recall {rec:.4f} | Precision {prec:.4f} | Specificity {spec:.4f} | Acc {acc:.4f}")

        scheduler.step(f1)
        early_stopping(f1, model)

        if early_stopping.best_score == f1 or early_stopping.early_stop:
            print(f"Saving Confusion Matrix to {CONFUSION_MATRIX_PATH}...")
            plot_confusion_matrix(total_tp, total_tn, total_fp, total_fn, CONFUSION_MATRIX_PATH)
            # # 2. 保存 Top-k 稀疏注意力热图 (2D Map)
            # # 这里的 stride=8 对应你 Enc1->Enc2->Enc3 的三次下采样 (2*2*2)
            # plot_sparse_attention(model, ATTENTION_MAP_PATH, stride=8)
            
            # # 3. 保存 基因组注意力分布曲线 (1D Profile)
            # # 保存为同名但不同后缀的文件，例如 sparse_attention_map_profile.png
            # profile_path = ATTENTION_MAP_PATH.replace(".pdf", "_profile.pdf")
            # plot_attention_profile(model, profile_path, stride=8)

        if early_stopping.early_stop:
            print(f"\n[Info] Early Stopping triggered! Best F1: {early_stopping.val_score_max:.4f}")
            break

    print(f"\nTraining Finished. Best model saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()