# predict.py
import argparse
import json
import torch
import os
import sys
from Bio import SeqIO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.Panno_model import Panno
from utils.predict import predict_chromosome, refine_gff_with_structure

def main(config):
    requested_device = str(config.get("device", "cuda")).lower()

    if "cuda" in requested_device and not torch.cuda.is_available():
        print(f" '{requested_device}'，但系统未检测到可用 GPU 驱动")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
        
    print(f"Using device: {device}")

    print(f"Loading Model from {config['model_path']}...")
    model = Panno(
        vocab_size=config['vocab_size'],
        base_filters=config['base_filters'],
        input_len=config['input_seq_len'],
        dropout_rate=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    print(f"Processing FASTA: {config['input_fasta']}")
    os.makedirs(os.path.dirname(config['output_gff']), exist_ok=True) # 确保输出目录存在
    
    with open(config['output_gff'], 'w') as out_f:
        out_f.write("##gff-version 3\n")
        
        for record in SeqIO.parse(config['input_fasta'], "fasta"):
            chrom_id = record.id
            seq_str = str(record.seq).upper() 
            
            if len(seq_str) < config['window_size']:
                continue
                
            print(f"Predicting {chrom_id} (Length: {len(seq_str)} bp)...")

            probs = predict_chromosome(config, model, seq_str, device)
            cnt = refine_gff_with_structure(config, chrom_id, seq_str, probs, out_f)
            
            print(f"  -> Found {cnt} valid sORFs.")

    print(f"Done! Predictions saved to {config['output_gff']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sORF Prediction using Panno")

    parser.add_argument("-c", "--config", type=str, default="configs/predict_config.json", help="Path to config file")
    

    parser.add_argument("-i", "--input_fasta", type=str, help="Override input FASTA file")
    parser.add_argument("-o", "--output_gff", type=str, help="Override output GFF file")
    parser.add_argument("-t", "--threshold", type=float, help="Override confidence threshold")
    args = parser.parse_args()
    
    config_path = args.config
    default_config_path = os.path.join(parent_dir, "configs", "predict_config.json")
    
    if not os.path.exists(config_path):
        print(f"Warning: 指定的配置文件 '{config_path}' 不存在，自动回退使用默认配置 '{default_config_path}'")
        config_path = default_config_path

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error:  '{config_path}' 请检查 configs 目录！")
        sys.exit(1) 
        
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value

    main(config)