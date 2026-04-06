# Semantifier: Cross-Field Semantic Consistency Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Semantifier** is a hybrid deep learning framework designed to detect logical inconsistencies across heterogeneous e-commerce product fields (e.g., Summary vs. Review Text). By integrating a **Dual-Encoder** backbone with a **Multi-Head Cross-Attention** mechanism, the model moves beyond simple keyword matching to perform directional logical inference.

## Key Features
- **Dual-Encoder Architecture**: Independently encodes product metadata using `RoBERTa-base` to preserve the distinct structural properties of different fields.
- **Cross-Attention Interaction**: Implements an asymmetrical attention layer where the Summary acts as a Query against the Review Text context to identify fine-grained contradictions.
- **NLI-Driven Silver Labeling**: Overcomes the lack of annotated ground truth by using `DeBERTa-v3` to generate high-precision synthetic consistency labels.
- **High-Performance Pipeline**: Optimized for Dual-GPU environments (NVIDIA T4), achieving a 92% reduction in epoch training time (90m → 7m) via PyTorch DDP and Mixed Precision (AMP).

## System Architecture
The Semantifier architecture is a modular, high-throughput pipeline designed for semantic validation.



### 1. NLI-Driven Silver Labeling (Data Pipeline)
Due to the scarcity of manually annotated consistency labels, the system employs an **Inference-as-Labeler** strategy.
* **Backbone**: Utilizes a DeBERTa-v3 architecture pre-trained on MultiNLI and SNLI datasets.
* **Logic**: The "Premise" is mapped to the detailed product review text ($f_t$), and the "Hypothesis" is mapped to the product summary ($f_s$).
* **Heuristic**: To maintain a high precision threshold, only pairs where the model predicts `entailment` are labeled as **Consistent ($L=1$)**. Samples predicted as `neutral` or `contradiction` are treated as **Inconsistent ($L=0$)**.

### 2. Dual-Encoder Feature Extraction
Unlike traditional single-encoders that concatenate inputs, Semantifier uses a **Dual-Encoder** approach to maintain distinct field properties.
* **Backbone**: Utilizes a `RoBERTa-base` backbone to project the summary and text into a shared latent space.
* **Operation**: This stage independently encodes fields into dense semantic embeddings, enabling scalable representation learning

### 3. Multi-Head Cross-Attention Interaction
The core innovation is the **Asymmetrical Cross-Attention** layer that identifies fine-grained semantic contradictions.
* **Query ($Q$)**: The summary hidden states serve as the Query ($Q = H_sW_Q$).
* **Key ($K$) & Value ($V$)**: The review text hidden states serve as the Key and Value ($K = H_tW_K, V = H_tW_V$).
* **Mechanism**: This ensures every token in the summary is cross-referenced against the entire context of the review text to identify "logical anchors".

### 4. Output & Inference Head
The attended features are aggregated and passed through a classification head for the final consistency score.
* **Pooling**: Features are processed through a global average pooling layer.
* **Regularization**: A dropout rate of 0.3 is applied to prevent overfitting on the synthetic silver labels.
* **Optimization**: The model is optimized using Binary Cross-Entropy (BCE) loss.

## Model Performance
| Metric | Value |
| :--- | :--- |
| **Accuracy** | ~82% |
| **AUC-ROC** | 0.78 |
| **Consistent Samples** | 159 (True Positive) |
| **Inconsistent Samples**| 137 (True Negative) |

The model demonstrates strong discriminative power, successfully mitigating "Majority Class Gravity" through strategic downsampling and balanced training.
