import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= Dataset =================
class TensorDataset(Dataset):
    def __init__(self, pt_path, vocab_size=5, augment=False):
        print(f"[Info] Loading data from {pt_path} into RAM...")
        self.data = torch.load(pt_path, weights_only=False)
        self.vocab_size = vocab_size
        self.augment = augment
        self.complement_map = torch.tensor([0, 4, 3, 2, 1], dtype=torch.long)

    def __len__(self): 
        return len(self.data)

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

# ================= Metrics & Tracking =================
def compute_metrics(logits, targets, threshold=0.3):
    if torch.isnan(logits).any():
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    preds = (torch.sigmoid(logits) > threshold).float().view(-1)
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

class MetricTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = {'loss': 0.0, 'f1': 0.0, 'rec': 0.0, 'prec': 0.0, 'acc': 0.0, 'mcc': 0.0}
        self.count = 0
        
    def update(self, loss_val, logits, targets):
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
        if self.count == 0: return self.metrics
        return {k: v / self.count for k, v in self.metrics.items()}

# ================= Early Stopping =================
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
            if self.verbose: print(f' | EarlyStopping count: {self.counter}/{self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0 

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f' | [Save] Validation score improved ({self.val_score_max:.4f} --> {val_score:.4f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score

# ================= Plotting & Logging =================
def save_logs_to_txt(history, file_path):
    with open(file_path, 'w') as f:
        header = f"{'Epoch':^5}\t{'Train_Loss':^10}\t{'Val_Loss':^10}\t{'Train_F1':^10}\t{'Val_F1':^10}\t{'Val_Rec':^10}\t{'Val_Prec':^10}\n"
        f.write(header + "-" * 80 + "\n")
        for i in range(len(history['train_loss'])):
            line = (f"{i+1:^5}\t{history['train_loss'][i]:^10.4f}\t{history['val_loss'][i]:^10.4f}\t{history['train_f1'][i]:^10.4f}\t{history['val_f1'][i]:^10.4f}\t{history['val_rec'][i]:^10.4f}\t{history['val_prec'][i]:^10.4f}\n")
            f.write(line)

def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='red')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], label='Train F1', color='blue', linestyle='--')
    plt.plot(epochs, history['val_f1'], label='Val F1', color='red')
    plt.plot(epochs, history['val_rec'], label='Val Recall', color='green', alpha=0.6)
    plt.title('Metrics')
    plt.xlabel('Epochs'); plt.ylabel('Score'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
