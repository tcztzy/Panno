import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-sORF', 'sORF'], yticklabels=['Non-sORF', 'sORF'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_pr_curve(recall, precision, ap, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_threshold_metrics(thresholds, precisions, recalls, f1_scores, best_thresh, max_f1, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b--', label='Precision', linewidth=1.5)
    plt.plot(thresholds, recalls, 'g-', label='Recall', linewidth=1.5)
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2.5)
    plt.axvline(x=best_thresh, color='k', linestyle=':', linewidth=1.5, label=f'Best Threshold: {best_thresh:.3f}')
    plt.plot(best_thresh, max_f1, 'ro') 
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Decision Threshold'); plt.ylabel('Score')
    plt.title(f'Metrics vs. Decision Threshold (Max F1={max_f1:.4f})')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()