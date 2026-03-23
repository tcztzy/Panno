import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.Panno_model import Panno
from utils.Lossfuction import ComprehensiveBoundaryLoss
from utils.train_utils import (
    TensorDataset, MetricTracker, EarlyStopping, 
    save_logs_to_txt, plot_training_curves
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Finetune the Panno Model")

    parser.add_argument("--train_path", type=str, required=True, help="Path to train_dataset.pt")
    parser.add_argument("--val_path", type=str, required=True, help="Path to val_dataset.pt")
    parser.add_argument("--save_prefix", type=str, required=True, help="Prefix for saving model, logs, and plots (e.g., outputs/muti_model)")
    parser.add_argument("--pretrained_path", type=str, default="", help="Path to .pth file. If provided, activates Fine-Tuning mode.")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (use 3e-5 for scratch, 1e-5 for finetune)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--vocab_size", type=int, default=5)
    parser.add_argument("--input_len", type=int, default=20480)
    parser.add_argument("--base_filters", type=int, default=128)

    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--bound_weight", type=float, default=2.0)
    parser.add_argument("--int_weight", type=float, default=1.0)
    
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in str(device) and torch.cuda.is_available():
        cudnn.benchmark = True
        print(f"[Info] CUDA Accelerated: {torch.cuda.get_device_name(0)}")

    print(f"\n{'='*30} STARTING PIPELINE {'='*30}")

    print("[Info] Loading Datasets...")
    train_ds = TensorDataset(args.train_path, vocab_size=args.vocab_size, augment=True) 
    val_ds = TensorDataset(args.val_path, vocab_size=args.vocab_size, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Panno(
        vocab_size=args.vocab_size,
        base_filters=args.base_filters,
        input_len=args.input_len,
        dropout_rate=args.dropout
    ).to(device)

    # === Fine-Tuning ===
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"[Info] Loading Pretrained Weights from: {args.pretrained_path}")
        print(f"[Info] MODE: FINE-TUNING")
        checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("[Info] MODE: TRAINING FROM SCRATCH")

    print(f"[Info] Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    criterion = ComprehensiveBoundaryLoss(
        alpha=args.alpha, 
        gamma=args.gamma, 
        boundary_weight=args.bound_weight, 
        internal_weight=args.int_weight,
        lambda_boundary=1.0, 
        lambda_dice=0.7
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2) 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    scaler = GradScaler()
    
    model_save_path = f"{args.save_prefix}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, path=model_save_path, verbose=True)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'val_rec': [], 'val_prec': []}

    for epoch in range(args.epochs):
        model.train()
        train_tracker = MetricTracker()
        current_lr = optimizer.param_groups[0]['lr']
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", leave=False)

        for i, (x, y) in enumerate(loop):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
                loss_step = loss / args.accumulation_steps
            
            if torch.isnan(loss):
                continue

            scaler.scale(loss_step).backward()
            
            if (i + 1) % args.accumulation_steps == 0:
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
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
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

        print(f"\nEp {epoch+1:^3} | LR: {current_lr:.1e} | TRAIN Loss: {train_avg['loss']:.4f} F1: {train_avg['f1']:.4f} | VAL Loss: {val_avg['loss']:.4f} F1: {val_avg['f1']:.4f} MCC: {val_avg['mcc']:.4f}")


        scheduler.step(val_avg['mcc'])
        early_stopping(val_avg['mcc'], model) 
        
        if early_stopping.early_stop:
            print(f"[Info] Early Stopping Triggered. Best Val Metric: {early_stopping.val_score_max:.4f}")
            break

    save_logs_to_txt(history, f"{args.save_prefix}_log.txt")
    plot_training_curves(history, f"{args.save_prefix}_plot.png")
    
    print(f"\nPipeline Finished.")
    print(f"Model saved at: {model_save_path}")
    print(f"Plots saved at: {args.save_prefix}_plot.png")

if __name__ == "__main__":
    main()