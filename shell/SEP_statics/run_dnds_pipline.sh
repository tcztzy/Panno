#!/bin/bash
INPUT_DIR="/data/user_home/2023122004/FFTransformer/protein/predict_sorf/cluster/ds/dnds_results_rice_vs_maize/Family_CDS_Files"

OUTPUT_DIR="/data/user_home/2023122004/FFTransformer/protein/predict_sorf/cluster/ds/dnds_results_rice_vs_maize/CDS/cds_results_maize_vs_rice"

MODEL="YN"
# ===========================================

TMP_DIR="$OUTPUT_DIR/tmp"
mkdir -p "$TMP_DIR"

echo "🚀 开始环境检查..."
for cmd in mafft pal2nal.pl KaKs_Calculator python; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ 致命错误: 未找到命令 $cmd，请确保它已安装并添加到了 PATH 环境变量中。"
        exit 1
    fi
done
echo "✅ 环境检查通过！"

cat << 'EOF' > "$TMP_DIR/helper.py"
# -*- coding: utf-8 -*-
import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

action = sys.argv[1]

if action == 'translate':
    infile, outfile = sys.argv[2], sys.argv[3]
    records = []
    for rec in SeqIO.parse(infile, 'fasta'):
        dna = str(rec.seq).upper()
        
        # Ensure length is a multiple of 3
        remainder = len(dna) % 3
        if remainder != 0:
            dna = dna[:len(dna) - remainder]
            
        # Remove stop codons (TAA, TAG, TGA)
        if len(dna) >= 3 and dna[-3:] in ['TAA', 'TAG', 'TGA']:
            dna = dna[:-3]
            
        prot_seq = Seq(dna).translate()
        records.append(SeqRecord(prot_seq, id=rec.id, description=""))
        
    SeqIO.write(records, outfile, 'fasta')

elif action == 'toaxt':
    infile, outfile, fam_id = sys.argv[2], sys.argv[3], sys.argv[4]
    records = list(SeqIO.parse(infile, 'fasta'))
    if len(records) == 2:
        with open(outfile, 'w') as f:
            f.write(fam_id + "\n")
            f.write(str(records[0].seq) + "\n")
            f.write(str(records[1].seq) + "\n\n")
EOF
# =========================================================

echo "🧬 开始流水线处理 (MAFFT -> Pal2Nal -> KaKs_Calculator)..."

TOTAL=$(ls -1 "$INPUT_DIR"/*.fasta 2>/dev/null | wc -l)
COUNT=0
SUCCESS=0

for dna_fasta in "$INPUT_DIR"/*.fasta; do

    base_name=$(basename "$dna_fasta" .fasta)
    COUNT=$((COUNT + 1))

    if [ $((COUNT % 100)) -eq 0 ] || [ "$COUNT" -eq "$TOTAL" ]; then
        echo "   ⏳ 正在处理: $COUNT / $TOTAL ($base_name)"
    fi

    # 1. 翻译 DNA 到 蛋白
    python "$TMP_DIR/helper.py" translate "$dna_fasta" "$TMP_DIR/${base_name}_prot.fasta"
    
    # 2. MAFFT 蛋白比对 
    mafft --quiet "$TMP_DIR/${base_name}_prot.fasta" > "$TMP_DIR/${base_name}_prot_aln.fasta" 2>/dev/null
    
    # 3. Pal2Nal 密码子比对 
    # -nogap 参数极其重要：去除有 gap 和终止密码子的列，保证 dN/dS 计算准确
    pal2nal.pl "$TMP_DIR/${base_name}_prot_aln.fasta" "$dna_fasta" -output fasta -nogap > "$TMP_DIR/${base_name}_codon_aln.fasta" 2>/dev/null
    
    # 检查 Pal2Nal 是否成功生成结果
    if [ -s "$TMP_DIR/${base_name}_codon_aln.fasta" ]; then
        # 4. 转为 AXT 格式
        python "$TMP_DIR/helper.py" toaxt "$TMP_DIR/${base_name}_codon_aln.fasta" "$TMP_DIR/${base_name}.axt" "$base_name"
        
        # 5. 计算 dN/dS (Ka/Ks)
        KaKs_Calculator -i "$TMP_DIR/${base_name}.axt" -o "$TMP_DIR/${base_name}.kaks" -m "$MODEL" >/dev/null 2>&1
        
        if [ -s "$TMP_DIR/${base_name}.kaks" ]; then
            SUCCESS=$((SUCCESS + 1))
        fi
    fi
done

echo "📊 处理完成！成功计算了 $SUCCESS / $TOTAL 个家族。"
echo "📝 正在合并结果表..."

FIRST_FILE=$(ls -1 "$TMP_DIR"/*.kaks | head -n 1)
head -n 1 "$FIRST_FILE" > "$OUTPUT_DIR/All_Families_${MODEL}_Results.tsv"

# 提取所有结果的数据行并合并
for kaks_file in "$TMP_DIR"/*.kaks; do
    sed '1d' "$kaks_file" >> "$OUTPUT_DIR/All_Families_${MODEL}_Results.tsv"
done

echo " dN/dS 汇总表在: $OUTPUT_DIR/All_Families_${MODEL}_Results.tsv"