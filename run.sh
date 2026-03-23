#BSUB -J predict
#BSUB -n 1
#BSUB -o panno.out
#BSUB -e panno.err
#BSUB -q gpu
#BSUB -m gpu01
#BSUB -gpu num=4
CUDA_VISIBLE_DEVICES=0

# /data/user_home/2023122004/anaconda3/envs/Enformer/bin/python Panno.py \
#   --input_fasta data/TAIR/genome/Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa \
#   --output_gff output/11.gff3 \
#   --threshold 0.60

# /data/user_home/2023122004/anaconda3/envs/Enformer/bin/python train.py \
#   --train_path data/tensor_data/Arabidopsis/train_dataset.pt \
#   --val_path data/tensor_data/Arabidopsis/val_dataset.pt \
#   --save_prefix model/panno_TAIR

# /data/user_home/2023122004/anaconda3/envs/Enformer/bin/python train.py \
#   --train_path data/tensor_data/MultiSpecies_5/train_dataset.pt \
#   --val_path data/tensor_data/MultiSpecies_5/val_dataset.pt \
#   --save_prefix data/tensor_data/MultiSpecies_5/finetuned_model \
#   --pretrained_path model/finetuned_model.pth \
#   --lr 1e-5 \
#   --epochs 200

# python evaluate.py \
#   --test_path data/tensor_data/Arabidopsis/test_dataset.pt \
#   --model_path model/Panno.pth \
#   --save_dir output/evaluation_results