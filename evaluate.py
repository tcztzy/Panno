import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from utils.Panno_model import Panno
from utils.train_utils import TensorDataset  
from utils.eval_utils import (           
    plot_confusion_matrix, plot_roc_curve, 
    plot_pr_curve, plot_threshold_metrics
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Panno Model")

    parser.add_argument("--test_path", type=str, required=True, help="Path to test_dataset.pt")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save evaluation results and plots")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_len", type=int, default=20480)
    parser.add_argument("--vocab_size", type=int, default=5)
    parser.add_argument("--base_filters", type=int, default=128)
    
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"[Info] Loading Test Data from {args.test_path}...")
    test_ds = TensorDataset(args.test_path, vocab_size=args.vocab_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = Panno(
        vocab_size=args.vocab_size,
        base_filters=args.base_filters,
        input_len=args.input_len,
        dropout_rate=0.2
    ).to(device)
    
    print(f"[Info] Loading weights from {args.model_path}...")
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"[Error] Model path not found: {args.model_path}")

    model.eval()

    all_targets, all_probs = [], []
    print("[Info] Starting Inference...")
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(x)
                probs = torch.sigmoid(logits) 
            
            all_probs.extend(probs.view(-1).cpu().numpy())
            all_targets.extend(y.view(-1).cpu().numpy())

    y_true = np.array(all_targets)
    y_score = np.array(all_probs)
    print(f"[Info] Inference done. Total samples: {len(y_true)}")

    print("Calculating AUC, AP & Thresholds...")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    thresholds_to_test = np.linspace(0.01, 0.99, 100)
    prec_list, rec_list, f1_list = [], [], []
    best_f1, best_thresh = -1, 0.5
    
    for th in tqdm(thresholds_to_test, desc="Scanning Thresholds", leave=False):
        y_pred_tmp = (y_score >= th) 
        tp = np.sum((y_pred_tmp == 1) & (y_true == 1))
        fp = np.sum((y_pred_tmp == 1) & (y_true == 0))
        fn = np.sum((y_pred_tmp == 0) & (y_true == 1))
        
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        
        prec_list.append(p); rec_list.append(r); f1_list.append(f)
        if f > best_f1:
            best_f1, best_thresh = f, th

    y_pred = (y_score >= best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. Output Results
    report = (
        f"==================================================\n"
        f" EVALUATION REPORT (Threshold = {best_thresh:.3f})\n"
        f"==================================================\n"
        f"AUC-ROC    : {roc_auc:.4f}\n"
        f"AU-PRC (AP): {avg_precision:.4f}\n"
        f"Accuracy   : {acc:.4f}\n"
        f"Precision  : {prec:.4f}\n"
        f"Recall     : {rec:.4f}\n"
        f"F1 Score   : {f1:.4f}\n"
        f"MCC        : {mcc:.4f}\n"
        f"--------------------------------------------------\n"
        f"Confusion Matrix:\n{cm}\n"
        f"==================================================\n"
    )
    print(f"\n{report}")
    
    with open(os.path.join(args.save_dir, "eval_metrics.txt"), "w") as f:
        f.write(f"Model: {args.model_path}\n{report}")

    print(f"[Info] Saving plots to {args.save_dir}...")
    plot_confusion_matrix(cm, os.path.join(args.save_dir, "confusion_matrix.png"))
    plot_roc_curve(fpr, tpr, roc_auc, os.path.join(args.save_dir, "roc_curve.png"))
    plot_pr_curve(recall_curve, precision_curve, avg_precision, os.path.join(args.save_dir, "pr_curve.png"))
    plot_threshold_metrics(
        thresholds_to_test, prec_list, rec_list, f1_list, 
        best_thresh, best_f1, os.path.join(args.save_dir, "threshold_metrics.png")
    )
    print("[Done] Evaluation complete.")

if __name__ == "__main__":
    main()