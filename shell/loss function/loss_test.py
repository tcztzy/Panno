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
import sys 
import matplotlib.pyplot as plt 

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)
from utils.Panno_model import Panno
from utils.Lossfuction import DiceFocalLoss,DiceFocalLoss_optimize,ComboLoss,FocalLoss,BoundaryAwareFocalLoss,ComprehensiveBoundaryLoss

loss_name = "ComprehensiveBoundaryLoss"
print(f'现在使用的是{loss_name}')

TRAIN_PATH = "/data/user_home/2023122004/FFTransformer/protein/0.5/MultiSpecies_2/train_dataset.pt"
VAL_PATH = "/data/user_home/2023122004/FFTransformer/protein/0.5/MultiSpecies_2/val_dataset.pt"
BASE_SAVE_PATH = f"/data/user_home/2023122004/FFTransformer/protein/0.5/MultiSpecies_2/muti{loss_name}"
MODEL_SAVE_PATH = f"{BASE_SAVE_PATH}.pth"

BATCH_SIZE = 64        
LEARNING_RATE = 3e-5       
EPOCHS = 500              
ACCUMULATION_STEPS = 2  
PATIENCE = 10         
INPUT_SEQ_LEN = 20480 
VOCAB_SIZE = 5           

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    cudnn.benchmark = True
    print(f"[Info] CUDA Accelerated: {torch.cuda.get_device_name(0)}") 


class MetricTracker:
    """用于累积和计算平均指标的工具类"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {
            'loss': 0.0, 'f1': 0.0, 'rec': 0.0, 
            'prec': 0.0, 'acc': 0.0, 'mcc': 0.0
        }
        self.count = 0
        
    def update(self, loss_val, logits, targets):
        """更新当前batch的指标"""
        self.count += 1
        self.metrics['loss'] += loss_val

        with torch.no_grad():
            f1, rec, prec, acc, mcc = compute_metrics(logits, targets)
            self.metrics['f1'] += f1
            self.metrics['rec'] += rec
            self.metrics['prec'] += prec
            self.metrics['acc'] += acc
            self.metrics['mcc'] += mcc
            
    def get_avg(self):
        """返回平均值字典"""
        if self.count == 0: return self.metrics
        return {k: v / self.count for k, v in self.metrics.items()}

class TensorDataset(Dataset):
    def __init__(self, pt_path, vocab_size=5, augment=False):
        """
        augment=True: 开启训练时的随机增强
        """
        print(f"Loading data from {pt_path} into RAM...")
        self.data = torch.load(pt_path, weights_only=False)
        self.vocab_size = vocab_size
        self.augment = augment

        self.complement_map = torch.tensor([0, 4, 3, 2, 1], dtype=torch.long)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq_int = torch.tensor(item['input'], dtype=torch.long)
        label = torch.tensor(item['label'], dtype=torch.float32)


        if self.augment and torch.rand(1).item() < 0.5:
            seq_int = torch.flip(seq_int, dims=[0])
            label = torch.flip(label, dims=[0])

            seq_int = self.complement_map[seq_int]

        seq_onehot = F.one_hot(seq_int, num_classes=self.vocab_size).permute(1, 0).float()
            
        return seq_onehot, label

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
                print(f' | EarlyStopping count: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0 

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f' | [Save] Validation score improved ({self.val_score_max:.4f} --> {val_score:.4f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score

def compute_metrics(logits, targets):
    if torch.isnan(logits).any():
        return 0.0, 0.0, 0.0, 0.0, 0.0

    preds = (torch.sigmoid(logits) > 0.3).float().view(-1)
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


def save_logs_to_txt(history, file_path):
    print(f"\n[Info] Saving training logs to {file_path}")
    with open(file_path, 'w') as f:

        header = f"{'Epoch':^5}\t{'Train_Loss':^10}\t{'Val_Loss':^10}\t{'Train_F1':^10}\t{'Val_F1':^10}\t{'Val_Rec':^10}\t{'Val_Prec':^10}\n"
        f.write(header)
        f.write("-" * 80 + "\n")

        for i in range(len(history['train_loss'])):
            line = (f"{i+1:^5}\t"
                    f"{history['train_loss'][i]:^10.4f}\t"
                    f"{history['val_loss'][i]:^10.4f}\t"
                    f"{history['train_f1'][i]:^10.4f}\t"
                    f"{history['val_f1'][i]:^10.4f}\t"
                    f"{history['val_rec'][i]:^10.4f}\t"
                    f"{history['val_prec'][i]:^10.4f}\n")
            f.write(line)

def plot_training_curves(history, save_path):
    print(f"[Info] Saving training plots to {save_path}")
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='red')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], label='Train F1', color='blue', linestyle='--')
    plt.plot(epochs, history['val_f1'], label='Val F1', color='red')
    plt.plot(epochs, history['val_rec'], label='Val Recall', color='green', alpha=0.6)
    plt.title('Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Load Data
    train_ds = TensorDataset(TRAIN_PATH, vocab_size=VOCAB_SIZE, augment=True) 
    val_ds = TensorDataset(VAL_PATH, vocab_size=VOCAB_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Model
    base_filters = 128
    model = Panno(vocab_size=VOCAB_SIZE,
                                  base_filters=base_filters,
                                    input_len=INPUT_SEQ_LEN,
                                    dropout_rate=0.2).to(DEVICE)
    print(f"[Info] Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    print("============================Lossfunction============================================================")
    # pos_weight = torch.tensor([1.0]).to(DEVICE) 
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE)

    # criterion = TverskyLoss(alpha=0.3, beta=0.7) 
    # criterion = DiceFocalLoss(pos_weight_val=2)
    #criterion = DiceFocalLoss_optimize(alpha=0.8, gamma=2.0, weight_dice=0.4, weight_focal=0.7)
    # criterion = FocalLoss(alpha=0.6, gamma=2.0)
    # criterion = BoundaryAwareFocalLoss(alpha=0.90, gamma=2.0, boundary_weight=10.0, internal_weight=5.0)
    criterion = ComprehensiveBoundaryLoss(
        alpha=0.60, 
        gamma=2, 
        boundary_weight=2, 
        internal_weight=1,
        lambda_boundary=1.0, 
        lambda_dice=0.7
    ).to(DEVICE)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2) 
    

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,     
        T_mult=2,    
        eta_min=1e-6
    )
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_SAVE_PATH, verbose=True)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'val_rec': [], 'val_prec': []
    }

    for epoch in range(EPOCHS):

        model.train()
        train_tracker = MetricTracker() 
        
        current_lr = optimizer.param_groups[0]['lr']
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)

        for i, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            
            with autocast():
                out = model(x)
                loss = criterion(out, y)
                loss_step = loss / ACCUMULATION_STEPS 
            
            if torch.isnan(loss):
                print(f"[Error] NaN loss detected!")
                continue

            scaler.scale(loss_step).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


            train_tracker.update(loss.item(), out.detach(), y.detach())

            loop.set_postfix(loss=loss.item(), rec=train_tracker.metrics['rec']/(train_tracker.count+1e-8))

        model.eval()
        val_tracker = MetricTracker()
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast():
                    out = model(x)
                    loss_val = criterion(out, y)
                val_tracker.update(loss_val.item(), out, y)


        train_avg = train_tracker.get_avg()
        val_avg = val_tracker.get_avg()

        history['train_loss'].append(train_avg['loss'])
        history['val_loss'].append(val_avg['loss'])
        history['train_f1'].append(train_avg['f1'])
        history['val_f1'].append(val_avg['f1'])
        history['val_rec'].append(val_avg['rec'])
        history['val_prec'].append(val_avg['prec'])

        # --- 打印对比行 ---
        # 格式：Epoch | LR | Type | Loss | F1 | Rec | Prec | MCC
        # 第一行：Train
        # 4. Training Loop
        print("\n" + "="*80)
        print(f"{'Epoch':^5} | {'LR':^7} | {'State':^5} | {'Loss':^7} | {'F1':^7} | {'Recall':^7} | {'Prec':^7} | {'MCC':^7}")
        print("="*80)
        print(f"{epoch+1:^5} | {current_lr:.1e} | {'TRAIN':^5} | {train_avg['loss']:.4f} | {train_avg['f1']:.4f} | {train_avg['rec']:.4f} | {train_avg['prec']:.4f} | {train_avg['mcc']:.4f}")
        # 第二行：Val (高亮显示)
        print(f"{'':^5} | {'':^7} | {'VAL':^5} | {val_avg['loss']:.4f} | {val_avg['f1']:.4f} | {val_avg['rec']:.4f} | {val_avg['prec']:.4f} | {val_avg['mcc']:.4f}")
        print("-" * 80)

        # --- Scheduler & Early Stopping ---
        # 使用 Validation F1 作为监控指标
        # scheduler.step(val_avg['f1'])
        # early_stopping(val_avg['f1'], model)
        scheduler.step(val_avg['mcc'])
        early_stopping(val_avg['mcc'], model) 
        
        if early_stopping.early_stop:
            print(f"\n[Info] Early Stopping Triggered. Best Val F1: {early_stopping.val_score_max:.4f}")
            break

    print(f"\nTraining Finished. Best model saved at: {MODEL_SAVE_PATH}")

    save_logs_to_txt(history, f"{BASE_SAVE_PATH}_log.txt")
    plot_training_curves(history, f"{BASE_SAVE_PATH}_plot.png")
    
    print(f"\nTraining Finished.")
    print(f"Model saved at: {MODEL_SAVE_PATH}")
    print(f"Logs saved at:  {BASE_SAVE_PATH}_log.txt")
    print(f"Plot saved at:  {BASE_SAVE_PATH}_plot.png")

if __name__ == "__main__":
    main()