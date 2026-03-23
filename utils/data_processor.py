import os
import numpy as np
import torch
from Bio import SeqIO
import random
import gc
import re

# Fixed underlying vocabulary mapping (No need to pass via config)
VOCAB_MAP = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def natural_sort_key(s):
    """Helper function for natural sorting of alphanumeric strings"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def is_valid_autosome(chrom_name, seq_len):
    """Checks if a sequence is a valid main chromosome/autosome."""
    name = chrom_name.lower()
    blacklist = ['mt', 'pt', 'mitochondrion', 'chloroplast', 'scaffold', 
                 'contig', 'supercontig', 'un', 'random', 'unknown', 'hap', 'alt','sc']
    for bad in blacklist:
        if bad in name:
            return False
    # Hard threshold: Sequences under 1MB are usually fragments/organelles
    if seq_len < 1_000_000:
        return False
    return True

def parse_genome_and_gff(fasta_path, gff_path_list, window_size):
    """Reads FASTA and multiple GFFs, returns sequences and merged annotation masks"""
    print(f"  Reading genome: {os.path.basename(fasta_path)}")
    if not os.path.exists(fasta_path):
        print(f"  [Error] Genome file not found: {fasta_path}")
        return None, None

    genome_seqs, genome_masks = {}, {}
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            # Ignore sequences shorter than the window size to save memory
            if len(record.seq) < window_size: 
                continue
            seq_str = str(record.seq).upper()
            genome_seqs[record.id] = seq_str
            genome_masks[record.id] = np.zeros(len(seq_str), dtype=np.int8)
    except Exception as e:
        print(f"  [Error] Failed to read FASTA: {e}")
        return None, None

    print(f"        Successfully loaded {len(genome_seqs)} sequences.")
    print(f"  Parsing GFF list and merging annotations...")
    total_cds_count = 0
    
    for gff_path in gff_path_list:
        if not os.path.exists(gff_path):
            print(f"  [Warning] GFF file not found, skipping: {gff_path}")
            continue
            
        print(f"        -> Processing GFF: {os.path.basename(gff_path)}")
        current_gff_cds = 0
        
        with open(gff_path, 'r') as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split('\t')
                if len(parts) < 9: continue
                if parts[2] != 'CDS': continue
                
                chrom = parts[0]
                if chrom not in genome_masks: continue

                try:
                    # GFF is 1-based, Python is 0-based
                    start = int(parts[3]) - 1
                    end = int(parts[4])
                except ValueError:
                    continue
                
                seq_len = len(genome_masks[chrom])
                real_end = min(end, seq_len)
                
                if start < seq_len:
                    genome_masks[chrom][start:real_end] = 1
                    current_gff_cds += 1
        
        print(f"           This file contributed {current_gff_cds} CDS regions.")
        total_cds_count += current_gff_cds
                
    print(f"        All GFFs processed. Total CDS regions marked: {total_cds_count}")
    return genome_seqs, genome_masks

def sequence_to_indices(seq_str):
    return [VOCAB_MAP.get(base, 0) for base in seq_str]

def split_chromosomes(genome_seqs):
    """Automatically splits chromosomes into Train / Val / Test sets"""
    canonical_chroms, other_seqs = [], []
    for k, seq in genome_seqs.items():
        if is_valid_autosome(k, len(seq)):
            canonical_chroms.append(k)
        else:
            other_seqs.append(k)
            
    canonical_chroms.sort(key=natural_sort_key)
    print(f"All valid autosomes (natural sort): {canonical_chroms}")
    
    if len(canonical_chroms) < 2:
        print("  [Warning] Less than 2 valid main chromosomes found. Cannot create test set! All data will go to training.")
        return list(genome_seqs.keys()), [], []

    # Assign the last two valid chromosomes to Test and Validation
    test_chrom = [canonical_chroms[-1]]
    val_chrom = [canonical_chroms[-2]]
    train_chroms = canonical_chroms[:-2] + other_seqs
    
    print(f"  Test: {test_chrom} | Val: {val_chrom} | Train: {len(train_chroms)} seqs")
    return train_chroms, val_chrom, test_chrom

def process_chromosome(chrom_name, seq, mask, dataset_type, species_name, config):
    """Slices a single chromosome into fixed-size windows"""
    local_samples = []
    seq_len = len(seq)
    window_size = config['window_size']
    stride = config['stride']
    
    for start_idx in range(0, seq_len, stride):
        end_idx = min(start_idx + window_size, seq_len)
        seq_chunk = seq[start_idx : end_idx]
        mask_chunk = mask[start_idx : end_idx]
        
        # Padding
        current_len = len(seq_chunk)
        if current_len < window_size:
            pad_len = window_size - current_len
            seq_chunk += 'N' * pad_len
            mask_chunk = np.concatenate([mask_chunk, np.zeros(pad_len, dtype=np.int8)])
        
        # N-content filtering
        if (seq_chunk.count('N') / window_size) > config['max_n_ratio']:
            continue
        
        # Sampling logic
        has_coding = np.sum(mask_chunk) > 0
        keep_sample = False
        
        if dataset_type == 'train':
            keep_sample = has_coding or (random.random() < config['negative_keep_rate'])
        elif dataset_type == 'val':
            if has_coding:
                keep_sample = True
            elif not config.get('downsample_val', False) or (random.random() < config['negative_keep_rate']):
                keep_sample = True
        else: # test
            keep_sample = True 

        if keep_sample:
            sample = {
                'input': np.array(sequence_to_indices(seq_chunk), dtype=np.int16),
                'label': mask_chunk.astype(np.int8), 
                'chrom': chrom_name,
                'start': start_idx,
                'species': species_name
            }
            local_samples.append(sample)
            
    return local_samples

def process_single_species(species_name, species_info, config):
    """Main pipeline for processing a single species"""
    print(f"\n{'='*20} Processing: {species_name} {'='*20}")
    
    out_dir = os.path.join(config['base_output_dir'], species_name)
    ensure_dir(out_dir)
    
    genome_seqs, genome_masks = parse_genome_and_gff(species_info['fasta'], species_info['gff_list'], config['window_size'])
    if genome_seqs is None: return 
    
    train_chroms, val_chroms, test_chroms = split_chromosomes(genome_seqs)
    datasets = {'train': [], 'val': [], 'test': []}
    
    print(f"  Starting slicing process...")
    for chrom in list(genome_seqs.keys()):
        if chrom in test_chroms: dtype = 'test'
        elif chrom in val_chroms: dtype = 'val'
        elif chrom in train_chroms: dtype = 'train'
        else: continue 
        
        samples = process_chromosome(chrom, genome_seqs[chrom], genome_masks[chrom], dtype, species_name, config)
        datasets[dtype].extend(samples)
        
        # Free memory immediately
        del genome_seqs[chrom], genome_masks[chrom]
        gc.collect()

    print(f" Saving datasets to {out_dir}")
    for dtype, data in datasets.items():
        if not data: continue
        if dtype == 'train': random.shuffle(data)
            
        torch.save(data, os.path.join(out_dir, f"{dtype}_dataset.pt"))
        print(f"        -> {dtype}: {len(data)} samples (Saved)")

    print(f"Successfully processed {species_name}.\n")
    del datasets
    gc.collect()

# ================= Core Call Interface =================
def data_process(config):
    """
    Main entry point for external calls.
    """
    print(">>> Starting Multi-Species Data Processing Pipeline <<<")
    ensure_dir(config['base_output_dir'])
    
    # 1. Process each species sequentially
    for s_name, s_info in config['species_catalog'].items():
        process_single_species(s_name, s_info, config)

    # 2. Merge all datasets if enabled (Optional, depending on your needs)
    # if config.get('merge_datasets', False):
    #     merge_all_datasets(config)
