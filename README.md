# Loss Functions for Handling Class Imbalance in Avalanche Forecasting
This repository provides the official reference implementation of the loss functions discussed in the accompanying paper on handling class imbalance in avalanche forecasting. All loss formulations analyzed theoretically and empirically in the manuscript are implemented directly within this codebase, ensuring a one-to-one correspondence between the methodological descriptions in the paper and the executable training pipeline. The repository includes implementations of Binary Cross-Entropy, Weighted Cross-Entropy, Focal Loss, Class-Balanced Loss, the proposed Performance-Guided Focal Loss (PGFL), and the proposed Asymmetric Margin Reinforcement Loss (AMRL). The code is designed to enable systematic comparison under a controlled experimental setup, support reproducibility (subject to data access), and facilitate research on rare-event prediction in highly imbalanced classification settings.



## Overview


The primary objective of this repository is not to introduce alternative model architectures, but to **isolate and study the effect of different loss formulations** under extreme class imbalance, which is characteristic of avalanche occurrence prediction. All models are trained under a controlled and consistent experimental setup, differing only in the choice of loss function.

The implementation is designed to support transparency, reproducibility (subject to data access), and extensibility for future research on rare-event forecasting.

---

## Implemented Loss Functions (Paper-Aligned)

This section follows the exact structure used in the **Loss functions for operational avalanche forecasting** section of the paper.

### 1 Binary Cross-Entropy (BCE)

Binary Cross-Entropy is included as the baseline loss function. In the paper, BCE serves as a reference point to illustrate the limitations of standard likelihood-based objectives when applied to highly imbalanced avalanche datasets. The implementation follows the canonical formulation without class reweighting.

### 2. Weighted Binary Cross-Entropy

Weighted BCE modifies the standard cross-entropy by introducing explicit class weights derived from class frequencies, as described in the paper. This loss partially mitigates imbalance by increasing the contribution of minority-class (avalanche) errors to the total loss. The implementation allows flexible adjustment of weights to match the experimental settings reported in the manuscript.

### 3. Focal Loss

Focal Loss is implemented according to the formulation discussed in the paper, introducing a focusing parameter that down-weights well-classified samples. In the avalanche forecasting context, this loss encourages the model to concentrate on difficult and rare avalanche events rather than being dominated by abundant non-avalanche samples.

### 4. Class-Balanced Loss (CBL)

Class-Balanced Loss is implemented following the formulation described in the paper. Instead of relying on raw class frequencies, this loss rescales contributions based on the effective number of samples, thereby providing a principled normalization under severe class imbalance. In the avalanche forecasting context, this improves convergence stability while moderately enhancing minority-class sensitivity.

### 5. Performance-Guided Focal Loss (PGFL)

The **Performance-Guided Focal Loss (PGFL)** is implemented directly within the training code as a dedicated loss function method. PGFL extends conventional focal loss by adaptively modulating gradient strength based on model performance, thereby balancing class importance and sample difficulty throughout training. This dynamic behavior reduces overfitting to rare avalanche samples while maintaining high detection capability.

### 6. Asymmetric Margin Reinforcement Loss (AMRL)

The proposed **Asymmetric Margin Reinforcement Loss (AMRL)** constitutes the primary methodological contribution of this work. AMRL introduces class-dependent, trainable margins directly in logit space to explicitly encode the asymmetric risk structure of avalanche forecasting. Avalanche samples are required to exceed a positive safety margin, while non-avalanche samples are constrained by a separate majority margin to limit false alarms. All margin and penalty parameters are optimized jointly with network weights, enabling adaptive and risk-aware decision boundary shaping under extreme class imbalance.

---

## Code Structure

The repository structure mirrors the experimental workflow described in relevant sections of the paper, with particular emphasis on loss-function experimentation. Unlike modular loss libraries, **all loss functions are implemented as distinct methods within the training code**, ensuring consistent data flow, shared model state, and identical optimization conditions across experiments. **The detailed structure is defined in code_structure.md file.**


### Loss Function Implementation Strategy

In alignment with the paper:

* **BCE, Weighted BCE, Focal Loss, Class-Balanced Loss, PGFL, and AMRL** are implemented as separate loss computation methods within the training loop.
* A configuration flag or function selector determines which loss function is active during training.
* All experiments use an identical model architecture, optimizer, learning-rate schedule, and stopping criteria, ensuring that observed performance differences arise solely from the loss formulation.

This design choice directly reflects the methodological objective of the paper: isolating the effect of loss-function design on avalanche forecasting performance.

---

### Deep Learning Architecture

The forecasting model follows the architecture described in the paper and is intentionally kept fixed across all experiments to isolate the effect of loss-function design.

1. Input Layer: Accepts standardized meteorological and snowpack predictors, matching the dimensionality of the input feature set.

2. Hidden Layers: Multiple fully connected (dense) layers with ReLU activations to model nonlinear relationships between predictors and avalanche occurrence.

3. Batch Normalization: Incorporated after hidden layers to stabilize training and improve convergence across different loss functions.

4. Dropout Regularization: Applied to reduce overfitting, which is especially important given the rarity of avalanche events.

5. Output Layer: A final sigmoid-activated layer that outputs probabilistic avalanche forecasts.

Training is performed using the Adam optimizer with gradient clipping for numerical stability. Early stopping based on validation loss is employed to prevent overfitting. All architectural and optimization settings are held constant across loss-function experiments, ensuring that performance differences can be attributed solely to the loss formulation.

### Evaluation Metrics

Model performance is evaluated using a comprehensive set of imbalance-aware and operationally relevant metrics, as described in **Evaluation Metrics** Section of the paper. These metrics are computed from the confusion matrix and implemented as reusable utility functions within the code.


**Detection Metrics**

1. **Probability of Detection (POD / Recall):** Proportion of avalanche days correctly identified; the primary operational metric.

2. **Precision (PRE):** Reliability of avalanche predictions, measuring the fraction of predicted avalanche days that are true events.

3. **True Negative Rate (TNR / Specificity):** Ability to correctly reject non-avalanche days.
   

**Balanced Performance Metrics**

1. **Balanced Accuracy (BA):** Mean of POD and TNR, providing a class-balanced measure of performance.

2. **Geometric Mean (GM):** Square root of the product of POD and TNR, emphasizing simultaneous improvement in both classes.
   

**Skill Scores**

1. **Heidke Skill Score (HSS):** Measures forecast improvement relative to random chance.

2. **True Skill Statistic (TSS / KSS):** Difference between POD and false-alarm rate; insensitive to event prevalence.

3. **Matthews Correlation Coefficient (MCC):** A robust single-score metric incorporating all elements of the confusion matrix and well suited to imbalanced datasets.

   

## Dataset Availability

The datasets used in the paper and in this codebase are **not included in this repository**.

Due to data access restrictions, and institutional constraints, the avalanche forecasting datasets cannot be publicly redistributed. This repository therefore contains **only the model, loss function, and training code**.

Researchers wishing to reproduce the experiments must obtain the datasets independently from the relevant data providers. Once available, the data can be integrated by adapting the data-loading utilities without altering the core experimental pipeline.

This separation is intentional and ensures compliance with data governance requirements while still enabling full methodological transparency.

---

## Reproducibility Notes

* All experiments were conducted using fixed random seeds to ensure consistency across runs.
* Hyperparameters associated with each loss function are documented in the training scripts and the accompanying paper.
* Evaluation metrics include precision, recall, F1-score, and event-based detection performance, with particular emphasis on minority-class recall.

---

## Intended Use

This repository is intended for:

* Researchers studying class imbalance in avalanche forecasting
* Comparative analysis of loss functions under extreme imbalance
* Extension of AMRL to other rare-event prediction domains

---

## Citation

If you use this code or the AMRL loss function in your research, please cite the associated paper as described in the manuscript.

---

## Disclaimer

This code is provided for **research purposes only**. The authors make no guarantees regarding operational avalanche forecasting performance or real-world deployment suitability.

