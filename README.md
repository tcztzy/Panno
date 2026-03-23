### Panno: time-frequency deep learning for context-aware annotation of plant sORFs

**Panno** is an *ab initio* deep learning framework designed for the genome-wide annotation of small open reading frames (sORFs) and sORF-encoded polypeptides (SEPs) at single-nucleotide resolution.![panno](C:\Users\S\Desktop\panno.png)

### Installation & Requirements

Panno requires Python 3.8+ and PyTorch. We recommend using a GPU environment for accelerated training

```python
conda create -n Panno python=3.8.0
conda activate Panno
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt 
```

### Inference (Recommended)

The command below takes an input DNA sequence ( `FATST` ), converts it to numerical matrices, predicts base-wise coding probabilities using the pre-trained Panno model, and post-processes these probabilities into primary sORF models, returning a standard `GFF3` output file.

```python
# 1. Create an output directory
mkdir -p output

# 2. Download an example chromosome (e.g., Arabidopsis thaliana Chr 1)
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-59/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz
gunzip Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz

# 3. Run Panno to predict sORFs
python predict.py \
  --input_fasta Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa \
  --output_gff output/TAIR_chr1.gff3 \
  --threshold 0.60
```

| Parameter             | Default                        | Explanation                                                  |
| --------------------- | ------------------------------ | ------------------------------------------------------------ |
| `-i`, `--input_fasta` | None                           | **[Required]** FASTA input file (the genome or chromosome sequence). |
| `-o`, `--output_gff`  | None                           | **[Required]** Output GFF3 file path for the predicted SEPs. |
| `-m`, `--model_path`  | Panno.pth                      | Path to the pre-trained Panno `.pth` model weights.          |
| `-t`, `--threshold`   | 0.60                           | Confidence threshold for sORF prediction. Increase to reduce false positives. |
| `-c`, `--config`      | `\configs/predict_config.json` | Path to the configuration file for advanced sliding-window settings. |

### Data Processing

Prepare your genome sequences and SEP GFF3 annotations. 

To process your own species, open `configs/data_process_config.json` and update the `"species_catalog"` block. You can define one or multiple species by specifying their respective FASTA and GFF3 paths

```python
python data_split.py
```


### Model Training & Fine-Tuning

#### To Train from Scratch:

```python
python train.py \
  --train_path data/tensor_data/Arabidopsis/train_dataset.pt \
  --val_path data/tensor_data/Arabidopsis/val_dataset.pt \
  --save_prefix model/panno_TAIR
```

#### To fine-tune on new species:

```
python train.py \
  --train_path data/tensor_data/MultiSpecies_5/train_dataset.pt \
  --val_path data/tensor_data/MultiSpecies_5/val_dataset.pt \
  --save_prefix data/tensor_data/MultiSpecies_5/finetuned_model \
  --pretrained_path model/panno.pth \
  --lr 1e-5 \
  --epochs 200
```

### Model Evaluation

Evaluate your trained model at single-nucleotide resolution. Panno prioritizes the Matthews Correlation Coefficient (MCC) and Recall to address extreme class imbalance.

```
python evaluate.py \
  --test_path data/tensor_data/Arabidopsis/test_dataset.pt \
  --model_path model/Panno.pth \
  --save_dir output/evaluation_results
```

 ### Whole-Genome Prediction & GFF3 Generation

```
python predict.py \
  --input_fasta Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa \
  --output_gff output/TAIR_chr1.gff3 \
  --threshold 0.60
```



### Acknowledgements

This work is based on [pytorch](https://pytorch.org/) and  [scikit-learn](https://scikit-learn.org/). The project is developed by following author and supervised by Prof. Xiangchao Gan(gan@njau.edu.cn)

Authors:

Song Jin  (jinsong@stu.njau.edu.cn): overall framework design,  model architecture formulation, and interpretability analysis 

Ziya Tang (Tang@stu.njau.edu.cn): prototype development, data processing, model validation

Yanhui Li(T2025126@njau.edu.cn) prototype development and data analysis

