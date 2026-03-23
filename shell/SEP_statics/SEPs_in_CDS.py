import pandas as pd 
import pyranges as pr
import numpy as np
import sys

REF_GFF3_PATH = "/data/user_home/2023122004/FFTransformer/data/maize/gff3/ZmB73.gff3"
INPUT_CSV = '/data/user_home/2023122004/FFTransformer/protein/predict_sorf/Z.mays/Pann_Annotation_Detailed.csv'
OUTPUT_CSV = '/data/user_home/2023122004/FFTransformer/protein/predict_sorf/Z.mays/startCDS_with_Position.csv'

def build_transcript_structure(gff_path):
    print(f"[-] Loading Reference GFF from: {gff_path}")
    gr = pr.read_gff3(gff_path)
    
    # 提取 CDS
    cds_df = gr[gr.Feature == 'CDS'].df.copy()
    
    if "Parent" not in cds_df.columns:
        cds_df["Parent"] = cds_df["ID"]

    p_to_t_map = pd.Series(cds_df.Parent.values, index=cds_df.ID).to_dict()

    tx_map = {}
    print("[-] Building transcript structure models...")
    
    for tx_id, group in cds_df.groupby("Parent"):
        strand = group.iloc[0]["Strand"]
        exons = list(zip(group["Start"], group["End"]))
        
        # 排序
        if strand == "+":
            exons.sort(key=lambda x: x[0])
        else:
            exons.sort(key=lambda x: x[1], reverse=True)
            
        total_len = sum(e - s for s, e in exons)
        
        tx_map[tx_id] = {
            "strand": strand,
            "exons": exons,
            "total_len": total_len
        }
        
    return tx_map, p_to_t_map

def get_relative_pos(row, tx_map, p_to_t_map):
    ref_raw = str(row['Ref_Feature_ID'])
    
    target_tx_id = None

    if ref_raw in tx_map:
        target_tx_id = ref_raw

    elif ref_raw in p_to_t_map:
        target_tx_id = p_to_t_map[ref_raw]

    else:
        clean_id = ref_raw.replace("CDS:", "").replace("transcript:", "")
 
        if clean_id in p_to_t_map:
            target_tx_id = p_to_t_map[clean_id]
        elif clean_id in tx_map:
            target_tx_id = clean_id

    if target_tx_id is None or target_tx_id not in tx_map:
        return np.nan, np.nan, "ID_Not_Found"

    tx_info = tx_map[target_tx_id]
    strand = tx_info['strand']
    exons = tx_info['exons']
    total_len = tx_info['total_len']

    s_start_0 = row['start'] - 1
    s_end_0 = row['end'] 
    
    site_loc = s_start_0 if strand == '+' else s_end_0 

    accumulated_bp = 0
    dist_from_atg = -1
    
    for (e_start, e_end) in exons:
        e_len = e_end - e_start
        
        if strand == '+' and (e_start <= site_loc < e_end):
            offset = site_loc - e_start
            dist_from_atg = accumulated_bp + offset
            break
        elif strand == '-' and (e_start < site_loc <= e_end):
            offset = e_end - site_loc
            dist_from_atg = accumulated_bp + offset
            break
        accumulated_bp += e_len
        
    if dist_from_atg != -1 and total_len > 0:
        return dist_from_atg, dist_from_atg / total_len, "Success"
    else:
        return np.nan, np.nan, "Out_of_CDS_Bounds"

print("[-] Reading input CSV...")
df = pd.read_csv(INPUT_CSV)

# 构建映射
tx_structure, protein_to_transcript = build_transcript_structure(REF_GFF3_PATH)

print("[-] Calculating relative positions...")

results = []
for row in df.to_dict('records'):

    if "CDS" in str(row.get('Region_Category', '')) or "ORF" in str(row.get('Region_Category', '')):
        dist, ratio, status = get_relative_pos(row, tx_structure, protein_to_transcript)
    else:
        dist, ratio, status = np.nan, np.nan, "Not_Target_Region"
    results.append((dist, ratio))

df['CDS_Distance_bp'] = [x[0] for x in results]
df['CDS_Relative_Pos'] = [x[1] for x in results]
df['CDS_Relative_Pos_Formatted'] = df['CDS_Relative_Pos'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

df.to_csv(OUTPUT_CSV, index=False)

valid_count = df['CDS_Relative_Pos'].notna().sum()
print(f"\n[Done] Results saved to: {OUTPUT_CSV}")
print(f"Total sORFs processed: {len(df)}")
print(f"Mapped Successfully: {valid_count} ({(valid_count/len(df)*100):.2f}%)")

if valid_count > 0:
    print("\n[Preview of Success Matches]")
    print(df.dropna(subset=['CDS_Relative_Pos'])[['ID', 'Ref_Feature_ID', 'CDS_Distance_bp', 'CDS_Relative_Pos_Formatted']].head())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

if 'df' not in locals():
    print("错误：请先加载您的 dataframe 到变量 df 中")
else:

    valid_data = df.dropna(subset=['CDS_Relative_Pos']).copy()

    bins = np.arange(0, 1.0001, 0.05)
    labels = [f"{int(b*100)}-{int((b+0.05)*100)}%" for b in bins[:-1]]
    valid_data['Bin_Label'] = pd.cut(
        valid_data['CDS_Relative_Pos'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    bin_stats = valid_data['Bin_Label'].value_counts().sort_index()
    total_count = len(valid_data)
    bin_percentages = (bin_stats / total_count) * 100

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['axes.linewidth'] = 1.0
    fig, ax = plt.subplots(figsize=(4.5, 2.5), dpi=1300)
    bar_color = '#009688' 
    x_indices = np.arange(len(bin_percentages))
    
    bars = ax.bar(
        x=x_indices,
        height=bin_percentages.values,
        width=0.85, 
        color=bar_color,
        linewidth=0,
        zorder=3,
        alpha=0.5,
    )

    major_ticks = np.arange(0, 21, 4) # 0, 4, 8, 12, 16, 20
    tick_labels = [f"{i*5}%" for i in major_ticks]

    ax.set_xticks(major_ticks - 0.5) 
    ax.set_xticklabels(tick_labels, fontsize=6)

    ax.set_ylabel('Proportion of initiation codons', fontsize=6)
    ax.set_xlabel('Relative Position on CDS', fontsize=6)

    max_y = bin_percentages.max()
    ax.set_ylim(0, max_y * 1.15) 
    ax.tick_params(axis='y', labelsize=6)
    # ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.savefig('/data/user_home/2023122004/FFTransformer/protein/predict_sorf/Z.mays/sORF_Distribution_Fixed.pdf', format='pdf', dpi=1300,  bbox_inches='tight')
    plt.savefig('/data/user_home/2023122004/FFTransformer/protein/predict_sorf/Z.mays/sORF_Distribution_Fixed.png', format='png', dpi=1300, bbox_inches='tight')
    plt.savefig('/data/user_home/2023122004/FFTransformer/protein/predict_sorf/Z.mays/sORF_Distribution_Fixed.svg', format='svg', dpi=1300, bbox_inches='tight')
    
    plt.show()