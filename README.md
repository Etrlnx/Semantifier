# Semantifier: Cross-Field Semantic Consistency Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Semantifier** is a hybrid deep learning framework designed to detect logical inconsistencies across heterogeneous e-commerce product fields (e.g., Summary vs. Review Text). [cite_start]By integrating a **Dual-Encoder** backbone with a **Multi-Head Cross-Attention** mechanism, the model moves beyond simple keyword matching to perform directional logical inference[cite: 10, 12, 45].

## 🚀 Key Features
- [cite_start]**Dual-Encoder Architecture**: Independently encodes product metadata using `RoBERTa-base` to preserve the distinct structural properties of different fields[cite: 11, 128].
- [cite_start]**Cross-Attention Interaction**: Implements an asymmetrical attention layer where the Summary acts as a Query against the Review Text context to identify fine-grained contradictions[cite: 12, 137].
- [cite_start]**NLI-Driven Silver Labeling**: Overcomes the lack of annotated ground truth by using `DeBERTa-v3` to generate high-precision synthetic consistency labels[cite: 99, 102].
- [cite_start]**High-Performance Pipeline**: Optimized for Dual-GPU environments (NVIDIA T4), achieving a 92% reduction in epoch training time (90m → 7m) via PyTorch DDP and Mixed Precision (AMP).

## 🏗️ System Architecture
[cite_start]The Semantifier architecture is a modular, high-throughput pipeline designed for semantic validation[cite: 10, 93].



### 1. NLI-Driven Silver Labeling (Data Pipeline)
[cite_start]Due to the scarcity of manually annotated consistency labels, the system employs an **Inference-as-Labeler** strategy[cite: 100, 101].
* [cite_start]**Backbone**: Utilizes a DeBERTa-v3 architecture pre-trained on MultiNLI and SNLI datasets[cite: 102].
* [cite_start]**Logic**: The "Premise" is mapped to the detailed product review text ($f_t$), and the "Hypothesis" is mapped to the product summary ($f_s$)[cite: 101].
* **Heuristic**: To maintain a high precision threshold, only pairs where the model predicts `entailment` are labeled as **Consistent ($L=1$)**. [cite_start]Samples predicted as `neutral` or `contradiction` are treated as **Inconsistent ($L=0$)**[cite: 105, 110].

### 2. Dual-Encoder Feature Extraction
[cite_start]Unlike traditional single-encoders that concatenate inputs, Semantifier uses a **Dual-Encoder** approach to maintain distinct field properties[cite: 129].
* [cite_start]**Backbone**: Utilizes a `RoBERTa-base` backbone to project the summary and text into a shared latent space[cite: 128].
* [cite_start]**Operation**: This stage independently encodes fields into dense semantic embeddings, enabling scalable representation learning[cite: 11, 132].

### 3. Multi-Head Cross-Attention Interaction
[cite_start]The core innovation is the **Asymmetrical Cross-Attention** layer that identifies fine-grained semantic contradictions[cite: 133].
* [cite_start]**Query ($Q$)**: The summary hidden states serve as the Query ($Q = H_sW_Q$)[cite: 137].
* [cite_start]**Key ($K$) & Value ($V$)**: The review text hidden states serve as the Key and Value ($K = H_tW_K, V = H_tW_V$)[cite: 137].
* [cite_start]**Mechanism**: This ensures every token in the summary is cross-referenced against the entire context of the review text to identify "logical anchors"[cite: 138, 252].

### 4. Output & Inference Head
[cite_start]The attended features are aggregated and passed through a classification head for the final consistency score[cite: 157].
* [cite_start]**Pooling**: Features are processed through a global average pooling layer[cite: 157].
* [cite_start]**Regularization**: A dropout rate of 0.3 is applied to prevent overfitting on the synthetic silver labels[cite: 158].
* [cite_start]**Optimization**: The model is optimized using Binary Cross-Entropy (BCE) loss[cite: 159, 161].

## 📊 Model Performance
| Metric | Value |
| :--- | :--- |
| **Accuracy** | ~82% |
| **AUC-ROC** | 0.78 |
| **Consistent Samples** | 159 (True Positive) |
| **Inconsistent Samples**| 137 (True Negative) |

[cite_start]The model demonstrates strong discriminative power, successfully mitigating "Majority Class Gravity" through strategic downsampling and balanced training[cite: 177, 207].

## 📁 Repository Structure
```text
└── Semantifier.ipynb       # Main Kaggle notebook containing the full pipeline
