#### Captum 可解释性

##### 1. captum.py

 It loads a pre-trained model along with a validation dataset. Using the InputXGradient method from the Captum interpretability library, the script computes and visualizes, in batches, the contribution (or importance) scores of each nucleotide position in the input DNA sequences during the model’s prediction process.

2. **plot_bp.py**

After the model successfully predicts an sORF, the contribution of each nucleotide is computed by backpropagating gradients.

