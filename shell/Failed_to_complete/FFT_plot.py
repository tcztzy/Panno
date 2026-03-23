import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import random
import os

# ================= 配置 =================
FASTA_PATH = "/data/user_home/2023122004/FFTransformer/data/MtSSPdb/MS-support/Known_transcript.fa"
SAVE_PATH = "./evidence_population_MS_support.png"
SAMPLE_LIMIT = 5000 

# ================= 工具函数 =================
def read_fasta(file_path, limit=0):
    sequences = []
    current_seq = []
    
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
                if limit > 0 and len(sequences) >= limit:
                    break
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
            
    print(f"Loaded {len(sequences)} sequences from {os.path.basename(file_path)}")
    return sequences

def dna_to_numerical(seq):
    seq = seq.upper()
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq)))
    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[mapping[char], i] = 1
    return one_hot

def compute_normalized_power(seq, target_len=150):
    """
    计算单条序列的功率谱，并插值到统一长度以便叠加
    target_len: 频率轴的分辨率
    """
    L = len(seq)
    if L < 10: return None, None # 太短忽略

    # 1. FFT
    signal = dna_to_numerical(seq)
    spectrum_energy = np.zeros(L)
    for i in range(4):
        f = fft(signal[i])
        spectrum_energy += np.abs(f)**2
    
    # 取半频 (Nyquist)
    freqs = np.fft.fftfreq(L)
    mask = freqs > 0
    valid_freqs = freqs[mask]
    valid_power = spectrum_energy[mask]
    
    # 2. 归一化 (除以总能量)
    if np.sum(valid_power) > 0:
        valid_power = valid_power / np.sum(valid_power)

    standard_x = np.linspace(0, 0.5, target_len)
    interp_power = np.interp(standard_x, valid_freqs, valid_power)
    
    return standard_x, interp_power

def plot_population_stats(sequences):
    """核心：群体叠加分析"""
    print("[Processing] Computing spectra for real and shuffled populations...")
    
    accumulated_real = None
    accumulated_shuffled = None
    count = 0

    BIN_NUM = 200 
    
    for seq in sequences:

        x_axis, p_real = compute_normalized_power(seq, target_len=BIN_NUM)
        if p_real is None: continue

        seq_list = list(seq)
        random.shuffle(seq_list)
        shuffled_seq = "".join(seq_list)
        _, p_shuf = compute_normalized_power(shuffled_seq, target_len=BIN_NUM)

        if accumulated_real is None:
            accumulated_real = np.zeros_like(p_real)
            accumulated_shuffled = np.zeros_like(p_shuf)
            
        accumulated_real += p_real
        accumulated_shuffled += p_shuf
        count += 1
        
        if count % 1000 == 0:
            print(f"   Processed {count} sequences...")

    avg_real = accumulated_real / count
    avg_shuf = accumulated_shuffled / count

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    plt.plot(x_axis, avg_shuf, color='gray', lw=1.5, alpha=0.8, label='Shuffled Control (Background)')
    plt.fill_between(x_axis, 0, avg_shuf, color='gray', alpha=0.1)

    plt.plot(x_axis, avg_real, color='#D62728', lw=2.5, label='MS-Supported sORFs')

    idx_1_3 = (np.abs(x_axis - 1/3)).argmin()
    signal_val = avg_real[idx_1_3]
    noise_val = avg_shuf[idx_1_3]
    snr = signal_val / noise_val

    plt.axvline(x=1/3, color='black', linestyle='--', alpha=0.6)
    plt.annotate(f'3-nt Periodicity Peak\n(SNR = {snr:.2f})', 
                 xy=(1/3, signal_val), 
                 xytext=(0.35, signal_val + 0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, fontweight='bold')

    plt.title(f"Population-Level Spectral Analysis (N={count})\nData Source: MS-Supported Transcripts", fontsize=14)
    plt.xlabel("Frequency")
    plt.ylabel("Average Normalized Spectral Power")
    plt.legend(loc='upper right')
    plt.xlim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"[Done] Plot saved to {SAVE_PATH}")
    print(f"Observed SNR at f=1/3: {snr:.4f}")

if __name__ == "__main__":

    seqs = read_fasta(FASTA_PATH, limit=SAMPLE_LIMIT)
    
    if len(seqs) > 0:
        plot_population_stats(seqs)
    else:
        print("No sequences loaded. Check path.")