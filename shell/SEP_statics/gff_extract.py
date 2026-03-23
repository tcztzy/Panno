import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import os
import re

GFF_PATH = "/data/user_home/2023122004/FFTransformer/protein/predict_sorf/st/Pann_Annotation_0.gff3"
FASTA_PATH = "/data/user_home/2023122004/FFTransformer/data/Solanum_tuberosum/Solanum_tuberosum.SolTub_3.0.dna_rm.toplevel.fa"
OUTPUT_CSV = "/data/user_home/2023122004/FFTransformer/protein/predict_sorf/st/Codon/Panno_SP.csv" 

def normalize_chrom_name(name):
    return str(name).replace("Chr", "").replace("chr", "")

def parse_attributes(attr_str):
    attributes = {}
    for item in attr_str.split(';'):
        if '=' in item:
            key, value = item.strip().split('=', 1)
            attributes[key] = value
    return attributes.get('ID', attributes.get('Parent', 'Unknown_ID'))

def extract_sequences(gff_path, fasta_path, output_csv):
    print(f"[-] 正在加载基因组 FASTA: {fasta_path} ...")

    genome_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        norm_id = normalize_chrom_name(record.id)
        genome_dict[norm_id] = record.seq
        genome_dict[record.id] = record.seq
        
    print(f"    已加载 {len(genome_dict)} 条序列记录 (包含标准化映射)。")

    print(f"[-] 正在处理 GFF 文件: {gff_path} ...")
    try:
        df = pd.read_csv(gff_path, sep='\t', comment='#', header=None, 
                         names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'],
                         low_memory=False)
    except Exception as e:
        print(f"    [Error] 读取 GFF 失败: {e}")
        return

    extracted_data = []

    for index, row in df.iterrows():
        chrom = str(row['seqid'])
        norm_chrom = normalize_chrom_name(chrom)

        start = int(row['start']) - 1
        end = int(row['end'])
        strand = row['strand']
        feat_id = parse_attributes(str(row['attributes']))

        if norm_chrom in genome_dict:
            ref_seq = genome_dict[norm_chrom]
        elif chrom in genome_dict:
            ref_seq = genome_dict[chrom]
        else:
            print(f"    [Warning] 染色体 {chrom} 未在 FASTA 中找到，跳过 {feat_id}")
            continue
        seq_slice = ref_seq[start:end]

        if strand == '-':
            # 如果是负链，取反向互补
            final_seq = seq_slice.reverse_complement()
        else:
            # 正链保持不变
            final_seq = seq_slice

        seq_str = str(final_seq)
        seq_len = len(seq_str)
        start_codon = seq_str[:3] 
        
        extracted_data.append({
            'ID': feat_id,
            'Chrom': chrom,
            'Start': row['start'],
            'End': row['end'],
            'Strand': strand,
            'Length': seq_len,
            'Start_Codon': start_codon,
            'Sequence': seq_str
        })
    result_df = pd.DataFrame(extracted_data)
    result_df.to_csv(output_csv, index=False)
    
    print(f"[-] 处理完成！")
    print(f"    共提取 {len(result_df)} 条序列")
    print(f"    结果已保存至: {output_csv}")

    print(result_df[['ID', 'Strand', 'Length', 'Start_Codon']].head())

if __name__ == "__main__":
    extract_sequences(GFF_PATH, FASTA_PATH, OUTPUT_CSV)