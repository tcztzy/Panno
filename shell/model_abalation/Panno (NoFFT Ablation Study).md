### Panno (NoFFT Ablation Study)

This project is designed for the pixel-level/site-level prediction of small Open Reading Frames (sORFs) within genomic sequences. The current codebase represents an **Ablation Study** version. It deliberately removes the frequency-domain feature extraction module (the FFT branch) from the original model, retaining only the pure 1D convolutional branch. The primary goal is to verify the effectiveness and necessity of frequency-domain features for long-distance genomic sequence annotation tasks.

#### 1.Model Architecture Overview (NoFFT Version)

**Stem**: `HybridStem_NoFFT` (Embedding -> 1D Conv -> Fusion; FFT branch removed)

**Encoder**: 3x `ResConvBlock` (Downsampling)

**Bottleneck**: 6-layer `TransformerEncoder` (with Positional Encoding)

**Decoder**: 3x `ConvTranspose1d` + `ResConvBlock` (Upsampling and Skip-Connection concatenation)

**Output**: `Conv1d` outputting single-channel probability logits.

#### 2. No u-net

removes all `ResConvBlock` local feature extractors and U-Net style skip connections, forcing the model to rely entirely on global context for inference.

**Frequency Domain (FFT)**: Extracts the periodic global frequency signals of the DNA sequence via the `SpectralFeatureBlock`.

**Spatial Domain (Transformer)**: Utilizes a 6-layer Transformer Encoder to capture long-range attention dependencies across the sequence.

####  3. input context

3kb-100kb

