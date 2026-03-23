import torch
import numpy as np
import torch.nn.functional as F
import pickle
import os

_config_path = os.path.join(os.path.dirname(__file__), '../model/vocab_config.bin')
with open(_config_path, 'rb') as f:
    _rules = pickle.load(f)

VOCAB_MAP = _rules['v_map']
START_CODONS = _rules['s_codons']
STOP_CODONS = _rules['e_codons']

def seq_to_tensor(seq_str, vocab_size=5):
    indices = [VOCAB_MAP.get(b, 0) for b in seq_str]
    seq_int = torch.tensor(indices, dtype=torch.long)
    seq_onehot = F.one_hot(seq_int, num_classes=vocab_size).permute(1, 0).float()
    return seq_onehot

def predict_chromosome(config, model, chrom_seq, device):
    """对单条染色体进行滑动窗口预测"""
    seq_len = len(chrom_seq)
    full_probs = np.zeros(seq_len, dtype=np.float16)
    counts = np.zeros(seq_len, dtype=np.int8)
    
    inputs, coords = [], []
    window_size = config['window_size']
    stride = config['stride']
    batch_size = config['batch_size']
    
    for start in range(0, seq_len, stride):
        end = min(start + window_size, seq_len)
        seq_chunk = chrom_seq[start:end]
        
        pad_len = 0
        if len(seq_chunk) < window_size:
            pad_len = window_size - len(seq_chunk)
            seq_chunk += 'N' * pad_len
            
        inputs.append(seq_to_tensor(seq_chunk))
        coords.append((start, end, pad_len))
        
        if len(inputs) == batch_size or (start + stride >= seq_len):
            batch_tensor = torch.stack(inputs).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = model(batch_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()
            
            for i, (s, e, p_len) in enumerate(coords):
                valid_len = window_size - p_len
                pred_slice = probs[i, :valid_len]
                full_probs[s:e] += pred_slice
                counts[s:e] += 1
            
            inputs, coords = [], []

    counts[counts == 0] = 1
    avg_probs = full_probs / counts
    return avg_probs

def find_best_orf(seq_chunk, local_probs, relative_start, config):
    """在给定的 DNA 片段中寻找符合规则的 sORF"""
    seq_len = len(seq_chunk)
    candidates = []
    min_len = config['min_orf_len']
    max_len = config['max_orf_len']
    threshold = config['threshold']

    for frame in range(3):
        for i in range(frame, seq_len - 2, 3):
            codon = seq_chunk[i:i+3]
            if codon in START_CODONS:
                start_idx, current_start_codon = i, codon 
                for j in range(start_idx + 3, seq_len - 2, 3):
                    stop_codon = seq_chunk[j:j+3]
                    if stop_codon in STOP_CODONS:
                        end_idx = j + 3 
                        orf_len = end_idx - start_idx
                        
                        if min_len <= orf_len <= max_len:
                            orf_probs = local_probs[start_idx:end_idx]
                            mean_score = np.mean(orf_probs) if len(orf_probs) > 0 else 0
                            
                            candidates.append({
                                'start': relative_start + start_idx,
                                'end': relative_start + end_idx,
                                'score': mean_score,
                                'len': orf_len,
                                'start_type': current_start_codon 
                            })
                        break 

    if not candidates:
        return None
    
    def sort_key(x):
        penalty = 0.15 if x['start_type'] == START_CODONS[1] else 0.3
        return x['score'] - penalty

    best_candidate = max(candidates, key=sort_key)
    
    actual_penalty = 0.15 if best_candidate['start_type'] == START_CODONS[1] else 0.3
    if (best_candidate['score'] - actual_penalty) < threshold: 
        return None
        
    return best_candidate

def refine_gff_with_structure(config, chrom_id, chrom_seq, probs, output_handle):
    """结合模型概率与生物学特征生成 GFF3"""
    seq_len = len(chrom_seq)
    threshold = config['threshold']
    
    binary_map = (probs > threshold).astype(np.int8)
    diffs = np.diff(np.concatenate(([0], binary_map, [0])))
    roi_starts = np.where(diffs == 1)[0]
    roi_ends = np.where(diffs == -1)[0]
    
    count = 0
    PADDING = 3 
    
    for s, e in zip(roi_starts, roi_ends):
        if (e - s) < 10: continue 
            
        search_start = max(0, s - PADDING)
        search_end = min(seq_len, e + PADDING)
        
        seq_chunk = chrom_seq[search_start:search_end]
        local_probs = probs[search_start:search_end]
        
        best_orf = find_best_orf(seq_chunk, local_probs, search_start, config)
        
        if best_orf:
            count += 1
            final_s, final_e = best_orf['start'], best_orf['end']
            score, s_type, length = best_orf['score'], best_orf['start_type'], best_orf['len']
            
            attr_str = (f"ID=sorf_{chrom_id}_{count};len={length};"
                        f"conf={score:.2f};start_codon={s_type}")
            gff_line = f"{chrom_id}\tPanno\tsORF\t{final_s+1}\t{final_e}\t{score:.4f}\t+\t0\t{attr_str}\n"
            output_handle.write(gff_line)
            
    return count