import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import os
import sys

from captum.attr import IntegratedGradients, InputXGradient
from torch.utils.data import Dataset

from utils.FFTransfomrer import DeepGenomicTransUnet

MODEL_PATH = "/data/user_home/2023122004/FFTransformer/D2MT_10kb/model/ComprehensiveBoundaryLoss.pth"
VAL_PATH = "/data/user_home/2023122004/FFTransformer/D2MT_10kb/test_dataset.pt"
INPUT_SEQ_LEN = 10240
VOCAB_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_MAP = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}

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

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor, target_indices=None):
        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits) 
        
        if target_indices is None:
            return probs.sum().unsqueeze(0)
        
        selected_probs = probs[:, target_indices]
        return selected_probs.sum().unsqueeze(0)

def plot_seq_logo(attribution, sequence_tensor, start_idx, end_idx, sorf_start, sorf_end, title=None, save_path_prefix=None):
    """
    绘制 Sequence Logo (已隐藏 Y 轴)。
    """

    attr_window_tensor = attribution[:, start_idx:end_idx].clone()

    rel_start = sorf_start - start_idx
    rel_end = sorf_end - start_idx
    window_len = attr_window_tensor.shape[1]

    s_start = max(0, rel_start)
    s_end = min(window_len, rel_start + 3)
    if s_end > s_start:
        attr_window_tensor[:, s_start:s_end] = torch.abs(attr_window_tensor[:, s_start:s_end])

    e_start = max(0, rel_end - 2)
    e_end = min(window_len, rel_end + 1)
    if e_end > e_start:
        attr_window_tensor[:, e_start:e_end] = torch.abs(attr_window_tensor[:, e_start:e_end])

    print("[Visual] Applying scale factor 0.2 (divide by 5) to flanking 25bp regions...")

    up_scale_start = max(0, rel_start - 25)
    up_scale_end = rel_start
    if up_scale_end > up_scale_start:
        attr_window_tensor[:, up_scale_start:up_scale_end] /= 5.0
    down_scale_start = rel_end + 1
    down_scale_end = min(window_len, rel_end + 1 + 25)
    if down_scale_end > down_scale_start:
        attr_window_tensor[:, down_scale_start:down_scale_end] /= 5.0

    attr_window_np = attr_window_tensor.cpu().numpy().transpose()
    df = pd.DataFrame(attr_window_np[:, 1:], columns=['A', 'C', 'G', 'T'])

    dynamic_width = max(15, window_len * 0.15)
    fig, ax = plt.subplots(figsize=(dynamic_width, 3.5))
    
    color_scheme = {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}

    logo = logomaker.Logo(df,
                          ax=ax,
                          color_scheme=color_scheme,
                          stack_order='small_on_top', 
                          font_name='DejaVu Sans')

    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True, linewidth=1.2)
    ax.set_yticks([])             
    ax.set_ylabel("")    
    ax.spines['left'].set_visible(False) 
    # ax.get_yaxis().set_visible(False) 

    ax.axvspan(rel_start - 0.5, rel_end + 1, color='#DEF2FB', alpha=1, zorder=-1, label='SEPs Region')

    ax.set_xticks([])       
    ax.set_xticklabels([])   
    ax.set_xlabel("")   

    ax.tick_params(axis='x', which='both', bottom=False, top=False)


    # if title:
    #     plt.title(title, fontsize=16, fontweight='bold', pad=15)
    # else:
    #     plt.title(f"Sequence Importance", fontsize=14, pad=15)
    calc_fontsize = int(dynamic_width * 0.8) 
    final_fontsize = min(24, max(12, calc_fontsize)) 
    plt.legend(
        loc='upper right', 
        frameon=True, 
        facecolor='white', 
        framealpha=1,
        fontsize=final_fontsize,  
        markerscale=2.0        
    )
    plt.tight_layout()

    if save_path_prefix:
        formats = ['pdf', 'svg']
        for fmt in formats:
            full_save_path = f"{save_path_prefix}.{fmt}"
            plt.savefig(full_save_path, dpi=1300, bbox_inches='tight',padinched=0.1)
            print(f"[Saved] {full_save_path}")
    
    
    plt.close()

def main():
    save_dir = "/data/user_home/2023122004/FFTransformer/figure/logomaker"
    os.makedirs(save_dir, exist_ok=True)

    print("[Info] Loading Model...")
    base_model = DeepGenomicTransUnet(vocab_size=VOCAB_SIZE, base_filters=128, input_len=INPUT_SEQ_LEN).to(DEVICE)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    base_model.eval()
    
    interpretable_model = ModelWrapper(base_model).to(DEVICE)

    print("[Info] Loading Validation Data...")

    val_data = torch.load(VAL_PATH)

    MAX_SAMPLES = 1   
    METHOD = "GradInput" 
    
    print(f"[Info] Searching for top {MAX_SAMPLES} HIGH-CONFIDENCE sORF samples...")
    print(f"[Info] Running interpretation method: {METHOD}...")
    
    count = 0

    if METHOD == "IG":
        ig = IntegratedGradients(interpretable_model)
    elif METHOD == "GradInput":
        ixg = InputXGradient(interpretable_model)
    
    for i, item in enumerate(val_data):
        if count >= MAX_SAMPLES:
            print(f"[Info] Done. Generated {count} images.")
            break
            
        label = item['label']
        input_seq = item['input']
   
        if np.sum(label) == 0: continue
            
        true_indices = np.where(label == 1)[0]
        start, end = true_indices[0], true_indices[-1]
        length = end - start

        if not (10 <= length <= 303): continue

        if start < 2000 or end > 8000: continue

        with torch.no_grad():
            seq_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
            seq_oh = F.one_hot(seq_tensor, num_classes=VOCAB_SIZE).permute(0, 2, 1).float()
            probs = torch.sigmoid(base_model(seq_oh)).squeeze(0).cpu().numpy()
            sorf_pred_prob = np.mean(probs[start:end])
            
            if sorf_pred_prob < 0.95: 
                continue
        count += 1
        chrom_name = item.get('chr', item.get('chrom', item.get('seqname', 'Unknown_Chr')))
        window_genomic_start = item.get('start', item.get('window_start', 0))

        abs_start = window_genomic_start + start
        abs_end = window_genomic_start + end

        if chrom_name == 'Unknown_Chr' and window_genomic_start == 0:

            plot_title = f"sORF Prediction (Relative Pos: {start}-{end})"
        else:
            plot_title = f"Chromosome {chrom_name}, Region: {abs_start}-{abs_end}"

        print(f"\n[Sample {count}/{MAX_SAMPLES}] Index {i}, Pos: {start}-{end}, Prob: {sorf_pred_prob:.4f}")

        seq_int = torch.tensor(input_seq, dtype=torch.long)
        seq_onehot = F.one_hot(seq_int, num_classes=VOCAB_SIZE).permute(1, 0).float().unsqueeze(0).to(DEVICE)
        seq_onehot.requires_grad = True

        target_indices_tensor = torch.arange(start, end + 1).to(DEVICE)
        if METHOD == "IG":
            baseline = torch.zeros_like(seq_onehot).to(DEVICE)
            attributions, delta = ig.attribute(
                inputs=seq_onehot,
                baselines=baseline,
                additional_forward_args=(target_indices_tensor,),
                n_steps=50,
                return_convergence_delta=True
            )
        elif METHOD == "GradInput":
            attributions = ixg.attribute(
                inputs=seq_onehot,
                additional_forward_args=(target_indices_tensor,)
            )

        analyze_start = max(0, start - 50)
        analyze_end = min(INPUT_SEQ_LEN, end + 50)
        
        attr_squeeze = attributions.squeeze(0).detach()
        seq_squeeze = seq_onehot.squeeze(0).detach()

        file_name = f"sample_{count}_{chrom_name}_{abs_start}_{abs_end}"
        full_path = os.path.join(save_dir, file_name)

        plot_seq_logo(
            attr_squeeze,
            seq_squeeze,
            start_idx=analyze_start,
            end_idx=analyze_end,
            sorf_start=start,
            sorf_end=end,
            title = plot_title,
            save_path_prefix=full_path
        )

if __name__ == "__main__":
    main()