### Loss Function Comparative Study for Protein Sequence Labeling

#### Overview

This experiment aims to evaluate the effectiveness of various loss functions in optimizing the **Panno** model for SEP sequence identification. The primary goal is to address common challenges in biological sequence labeling, such as **class imbalance** and **boundary ambiguity**.

#### Experimental Design

The study follows a controlled variable methodology:

- Fixed Parameters: Model architecture (Panno), training data, Batch Size (64), Optimizer (AdamW), and Learning Rate scheduler (CosineAnnealing).
- Core Variable: The `criterion` (Loss Function) is the only variable modified across different training runs.
- Evaluation Metrics: Performance is benchmarked using F1-score, MCC (Matthews Correlation Coefficient), Recall, and Precision.

#### Loss Functions Under Evaluation

We have implemented and tested a spectrum of loss functions, ranging from standard baselines to advanced boundary-aware objectives

| **Loss Function**               | **Key Mechanism & Objective**                                |
| ------------------------------- | ------------------------------------------------------------ |
| **BCEWithLogitsLoss**           | **Baseline**: Standard binary cross-entropy with `pos_weight` to handle class imbalance. |
| **Focal Loss**                  | **Hard Example Mining**: Focuses on difficult samples by down-weighting easy-to-classify examples using the $\gamma$ parameter. |
| **DiceFocal Loss**              | **Region + Pixel Accuracy**: Combines Dice Loss (overlap-based) and Focal Loss (pixel-based) for robust segmentation. |
| **BoundaryAware Focal Loss**    | **Edge Focus**: Introduces higher weights for sequence boundaries to improve transition-point detection. |
| **Comprehensive Boundary Loss** | **(Current Optimization)**: A sophisticated loss integrating boundary weights, internal weights, and Dice constraints for holistic region identification. |

#### Hyperparameter Settings

For the most advanced configuration (**ComprehensiveBoundaryLoss**), the following parameters were utilized:

- **Alpha (alph):** 0.60
- **Gamma (gammz):** 2.0
- **Boundary Weight:** 2.0
- **Internal Weight:** 1.0
- **Lambda Dice:** 0.7
- **Early Stopping:** Monitored via **MCC** with a patience of 10 epochs to ensure optimal generalization.