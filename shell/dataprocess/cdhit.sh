#BSUB -J CDHIT
#BSUB -n 5
#BSUB -o /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/GCF_valid/CDHIT.out
#BSUB -e /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/GCF_validg/CDHIT.err
#BSUB -q cpu
#BSUB -m node09

cd /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/GCF_valid/CD-HIT
conda activate py12
cd-hit -i MS.fasta -o CD-hit-90.fa -c 0.9 -n 5 -M 0 -T 0