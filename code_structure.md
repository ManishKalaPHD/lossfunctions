# Code Structure and Methodology Documentation

## Overview

This module (`loss_functions_for_rare_events.py`) is a Python script exported from a Google Colab notebook. It is designed to **benchmark multiple loss functions for rare-event (imbalanced) binary classification problems** using a consistent deep learning architecture and a common set of evaluation metrics.

The script performs the following high-level tasks:

1. Loads and preprocesses multiple datasets
2. Defines and compares specialized loss functions for rare events
3. Trains a neural network model under each loss function
4. Evaluates performance using threshold-independent and threshold-dependent metrics
5. Aggregates and summarizes results across datasets

The code is written using **TensorFlow / Keras**, with auxiliary support from **NumPy, Pandas, and scikit-learn**.

---

## Libraries and Dependencies

The primary libraries used are:

* **TensorFlow / Keras**: Model definition, training, custom losses, and metrics
* **Pandas**: Dataset loading and tabular manipulation
* **NumPy**: Numerical operations
* **scikit-learn**:

  * `train_test_split` for dataset splitting
  * `StandardScaler` for feature normalization

---

## Data Handling and Preprocessing

### Dataset Loading

* **`load_data_from_drive(file_path)`**

  * Loads datasets from Google Drive or local paths
  * Returns feature matrix and binary labels

### Class Imbalance Utilities

* **`get_class_ratio(y)`**

  * Computes positive-to-negative class ratio
  * Used to initialize loss reweighting parameters

### Train–Test Split and Scaling

* Uses `train_test_split` with stratification
* Feature normalization via `StandardScaler`

---

## Deep Learning Architecture

Model architecture is encapsulated in **explicit model classes and factory functions**, ensuring reproducibility and modularity.

### Core Architecture Functions

* **`initialize_model(input_dim)`**

  * Constructs and returns a fully connected neural network
  * Uses the Keras Functional API

* **`Model` subclasses with `__init__()` and `call()` methods**

  * Encapsulate forward-pass logic
  * Enable integration with custom training loops

### Architecture Pattern

The neural network follows a consistent structure:

1. **Input Layer**

   * Shape determined by `input_dim`

2. **Hidden Dense Layers**

   * Multiple fully connected layers
   * Nonlinear activations (e.g., ReLU)

3. **Output Layer**

   * Single neuron
   * `sigmoid` activation for probability output

This architecture remains fixed across experiments to isolate the effect of loss-function design.

---

## Loss Functions for Rare Events

The script defines and evaluates **explicitly named loss-function mechanisms and training procedures**, implemented as custom TensorFlow training loops rather than relying solely on `model.fit()`.

### Adaptive and Custom Loss Components

Key loss-related logic is implemented through the following functions and classes:

* **`train_step_amrl()` / `test_step_amrl()`**

  * Custom training and evaluation steps for the *Adaptive Margin–Reweighted Loss (AMRL)* variant
  * Explicit control over:

    * Forward pass
    * Loss computation
    * Gradient calculation
    * Gradient clipping (`CLIP_NORM`)

* **`custom_train_loop()`**

  * Orchestrates epoch-level training using `train_step_amrl`
  * Handles metric tracking and early stopping logic

* **`update_alpha_dynamically()`**

  * Dynamically adjusts class-imbalance weighting parameters
  * Uses moving averages of prediction confidence

* **`update_confidence_moving_averages()`**

  * Tracks positive and negative class confidence trends
  * Supports adaptive loss reweighting for rare events

These mechanisms allow the loss function to **adapt during training**, rather than remaining static.

---

## Model Compilation and Training

Training is performed using **explicit training loops**, rather than `model.fit()`, to allow fine-grained control over loss dynamics.

### Training Functions

* **`train_model()`**

  * High-level wrapper that:

    * Initializes the model
    * Selects the appropriate custom training loop
    * Tracks metrics and early stopping

* **`custom_train_loop()` / `custom_train_loop_pgfl()`**

  * Epoch-level control over:

    * Forward pass
    * Backpropagation
    * Metric updates
    * Early stopping (`EARLY_STOPPING_PATIENCE`, `MIN_DELTA`)

* **Gradient Clipping**

  * Implemented explicitly using `CLIP_NORM` to stabilize training

---

## Evaluation Metrics

The script implements **domain-specific and imbalance-aware metrics** as standalone functions, in addition to standard Keras metrics.

### Custom Metric Functions

The following functions compute evaluation metrics directly from predictions and ground truth labels:

* **`false_alarm(y_true, y_pred)`**
  Measures the false alarm rate (false positives among negative instances).

* **`probability_of_detection(y_true, y_pred)`**
  Measures recall for the rare (positive) class.

* **`bias_score(y_true, y_pred)`**
  Ratio of predicted positives to actual positives, indicating systematic over- or under-prediction.

* **`heidke_skill_score(y_true, y_pred)`**
  Skill score comparing the model against random chance.

* **`pierce_skill_score(y_true, y_pred)`**
  Measures discrimination ability relative to a reference forecast.

* **`f1_score(y_true, y_pred)`**
  Harmonic mean of precision and recall.

These metrics are particularly common in **rare-event forecasting and risk modeling domains**.

### Keras Metrics

In addition to custom metrics, the model uses:

* `BinaryAccuracy`
* `Precision`
* `Recall`


---

## Experiment Loop and Result Collection

For each dataset:

1. Iterate over all defined loss functions
2. Train a fresh model instance
3. Evaluate performance on the test set
4. Store results (dataset name, loss type, metrics)

Results are accumulated into a **Pandas DataFrame**, enabling downstream analysis.

---

## Performance Aggregation and Summary

After processing all datasets:

* Metrics are grouped by **Loss_Type**
* Mean performance is computed across datasets
* Results are rounded for readability

Example summary operation:

* Average Precision, Recall per loss function

This provides a **global comparison** of loss functions across heterogeneous datasets.

---

## Output and Reporting

* Intermediate results are printed during execution
* Final summaries are displayed as tabular outputs
* The structure supports easy export to CSV or LaTeX for academic reporting

---

## Intended Use Cases

* Rare-event prediction (e.g., rare event forecasting, avalanche forecasting)
* Empirical comparison of loss functions
* Reproducible deep learning experiments under class imbalance

---

## Notes and Extensions

* The modular structure allows easy addition of new loss functions
* Architecture can be swapped with minimal code changes
* Threshold optimization and calibration can be added as a post-processing step

---

## Summary

This codebase provides a **controlled experimental framework** for studying the impact of loss function design on rare-event classification performance, using a consistent neural architecture and a comprehensive evaluation protocol.


---

