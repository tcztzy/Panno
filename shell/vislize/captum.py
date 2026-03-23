import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
from captum.attr import InputXGradient
from torch.utils.data import Dataset
from scipy.ndimage import convolve

from utils.FFTransfomrer import DeepGenomicTransUnet

MODEL_PATH = "/data/user_home/2023122004/FFTransformer/10kb/model/ComprehensiveBoundaryLoss.pth"
VAL_PATH = "/data/user_home/2023122004/FFTransformer/10kb/test_dataset.pt"
INPUT_SEQ_LEN = 10240
VOCAB_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/data/user_home/2023122004/FFTransformer/figure/gradient" 

class ModelWrapper_Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor, target_indices=None):
        logits = self.model(input_tensor)
        if target_indices is None:
            return logits.sum().unsqueeze(0)
        selected_logits = logits[:, target_indices]
        return selected_logits.sum().unsqueeze(0)

class TensorDataset(Dataset):
    def __init__(self, pt_path, vocab_size=5):
        self.data = torch.load(pt_path, weights_only=False)
        self.vocab_size = vocab_size
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        seq_int = torch.tensor(item['input'], dtype=torch.long)
        seq_onehot = F.one_hot(seq_int, num_classes=self.vocab_size).permute(1, 0).float()
        return seq_onehot, torch.tensor(item['label'], dtype=torch.float32)

def plot_paper_style(attribution, prediction, label, title=None, save_path_prefix=None, smooth_window=2):

    attr_raw = torch.sum(torch.abs(attribution), dim=1).squeeze().detach().cpu().numpy()
    
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        attr_smoothed = convolve(attr_raw, kernel, mode='nearest')
    else:
        attr_smoothed = attr_raw
    
    robust_max = np.percentile(attr_smoothed, 99.9)
    if robust_max == 0: robust_max = 1.0
    attr_norm = np.clip(attr_smoothed, 0, robust_max) / robust_max
    
    label_np = label.squeeze().detach().cpu().numpy()
    x_axis = np.arange(len(label_np))

    fig = plt.figure(figsize=(20, 7))

    gs = gridspec.GridSpec(2, 1, height_ratios=[0.15, 1], hspace=0.3)

    ax0 = plt.subplot(gs[0])
    
    LABEL_GREEN =  '#55A868'
    ax0.fill_between(x_axis, 0, 1, where=(label_np > 0.5), color=LABEL_GREEN, alpha=1, transform=ax0.get_xaxis_transform())
    
    ax0.set_ylabel('Label', fontsize=12, rotation=0, labelpad=30, va='center')#fontweight='bold', 
    ax0.set_yticks([]) 
    ax0.set_xlim(0, len(x_axis))

    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['bottom'].set_visible(True)

    ax0.tick_params(axis='x', direction='out')

    if title:
        ax0.set_title(title, fontsize=14,  )

    ax1 = plt.subplot(gs[1])
    
    SCIENCE_BLUE = '#4c72b0'
    
    ax1.plot(x_axis, attr_norm, color=SCIENCE_BLUE, linewidth=0.8, label='Gradient')
    ax1.fill_between(x_axis, 0, attr_norm, color=SCIENCE_BLUE, alpha=0.3)
    
    ax1.set_ylabel('Gradient', fontsize=12, )
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, len(x_axis))
    ax1.set_xlabel('Genomic Position (bp)', fontsize=12)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax1.grid(True, which='major', axis='y', linestyle=':', color='gray', alpha=0.4)

    if save_path_prefix:
        formats = ['pdf', 'svg', 'png']
        for fmt in formats:
            full_save_path = f"{save_path_prefix}.{fmt}"
            plt.savefig(full_save_path, dpi=1300, bbox_inches='tight',padinched=0.1)
            print(f"[Saved] {full_save_path}")
    
    
    plt.close()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("[Info] Loading Model...")
    base_model = DeepGenomicTransUnet(vocab_size=VOCAB_SIZE, base_filters=128, input_len=INPUT_SEQ_LEN).to(DEVICE)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    base_model.eval()
    
    interpretable_model = ModelWrapper_Logits(base_model).to(DEVICE)
    
    print("[Info] Loading Validation Data...")
    val_data = torch.load(VAL_PATH)
    
    MAX_SAMPLES = 50
    count = 0
    ixg = InputXGradient(interpretable_model)
    
    print(f"[Info] Starting visualization loop...")

    for i, item in enumerate(val_data):
        if count >= MAX_SAMPLES: break
        if count == 0:
            print(f"\n[DEBUG] Dictionary keys in your data: {list(item.keys())}")
        label = item['label']
        input_seq = item['input']
        
        if np.sum(label) == 0: continue
        true_indices = np.where(label == 1)[0]
        if len(true_indices) < 50: continue 

        seq_int = torch.tensor(input_seq, dtype=torch.long)
        seq_onehot = F.one_hot(seq_int, num_classes=VOCAB_SIZE).permute(1, 0).float().unsqueeze(0).to(DEVICE)
        seq_onehot.requires_grad = True

        with torch.no_grad():
            logits_for_plot = base_model(seq_onehot)
            probs = torch.sigmoid(logits_for_plot)

        target_indices_tensor = torch.tensor(true_indices, device=DEVICE)
        
        attributions = ixg.attribute(
            inputs=seq_onehot,
            additional_forward_args=(target_indices_tensor,)
        )
        
        chrom_name = item.get('chrom', 'Unknown')
        start_pos = item.get('start', 0)
        
        print(f"[Processing Sample {count+1}] Chr {chrom_name}, Pos {start_pos}, SEPs len {len(true_indices)}")
        
        plot_title = f"Chr{chrom_name}: {start_pos}-{start_pos+INPUT_SEQ_LEN}"
        save_name = os.path.join(SAVE_DIR, f"{count}_{chrom_name}_{start_pos}")
        
        plot_paper_style(attributions, probs, torch.tensor(label), title=plot_title, save_path_prefix=save_name)
        
        count += 1

if __name__ == "__main__":
    main()