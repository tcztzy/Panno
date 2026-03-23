#BSUB -J test
#BSUB -n 4
#BSUB -o /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/MS-support/minprot/test.out
#BSUB -e /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/MS-support/minprot/test.err
#BSUB -q cpu
#BSUB -m node08


cd /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/
#
# miniprot --gff genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa MS-support/CD-HIT/MS-90.faa > MS-support/minprot/TAIR.gff3

miniprot -k 5 -L 10 --gff /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/0Data/MtSSPdb/genome/Medicago_truncatula.MedtrA17_4.0.dna.toplevel.fa /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/0Data/MtSSPdb/MS-support/SPs.faa > /data/user_home/2023122004/Helixer/Socp/sOCP/0dataset/TAIR10/GCF1735.4/Dateset/gff3/MT_sps.gff3