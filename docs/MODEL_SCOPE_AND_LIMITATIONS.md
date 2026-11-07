# Model scope and current limitations

This repository is an **experimental pipeline scaffold**, not yet a validated crystal-property prediction system. The current code is useful for exercising the data, training, and inference flow, but its outputs should not be interpreted as scientifically meaningful material predictions.

## What the current implementation does

- Loads benchmark records from the configured JSON file.
- Converts three configured target properties into regression labels.
- Builds a simplified graph-shaped input and trains a multi-output PyTorch model.
- Keeps dropout active during repeated inference passes to calculate a mean prediction and a Monte Carlo spread.

## Important limitations

### 1. CIF files are not parsed into crystal structures

`battery_cathode_ai/graph_builder.py` currently creates deterministic pseudo-random atom and neighbor features from each input path. It does not read CIF contents, identify elements, calculate interatomic distances, or construct a crystallographically meaningful neighbor list. The generated neighbor indices are also not currently consumed by the convolution layer.

Consequently, the current graph inputs do not encode the actual crystal chemistry or geometry.

### 2. The network is a simplified CGCNN-like scaffold

The convolution layer combines atom features with an average of the supplied neighbor-feature vectors. It does not perform the full message-passing operation of the original Crystal Graph Convolutional Neural Network. The class name should therefore be understood as describing the intended model family, not a claim of full architectural equivalence.

### 3. Missing and interval-valued labels need explicit data policy

In `data_loader.py`, a missing target is currently converted to `0.0`. A non-empty list or tuple is reduced to its arithmetic mean. These are convenience defaults, not generally valid scientific assumptions: zero may be a real measurement, and an interval may represent uncertainty or a range rather than a point estimate.

Before training on real data, define per-task missing-value handling and decide whether interval-valued measurements should be excluded, modeled as intervals, or converted using a documented domain-specific rule.

### 4. Validation behavior has edge cases

The loader uses a seeded random split, but does not stratify by chemistry, material family, or target distribution. When the computed validation size is zero, the same dataset is returned for both training and validation. Validation metrics from that fallback are not an independent estimate of generalization.

For scientific evaluation, use a deliberately designed split (for example, by material family or another leakage-aware grouping) and ensure the validation set is disjoint from training data.

### 5. Inference can run without a trained checkpoint

If the configured checkpoint path does not exist, `predict.py` leaves the model at its random initialization and still returns predictions. The command-line demo uses synthetic input tensors. Such output only demonstrates that the inference path executes; it is not a trained model result.

The reported Monte Carlo standard deviation measures variation across dropout-enabled forward passes. By itself, it is not evidence that the uncertainty estimate is calibrated.

## Minimum requirements before scientific use

1. Replace the mock graph builder with a tested CIF parser and physically meaningful atom, neighbor, and distance features.
2. Implement and test the intended CGCNN message-passing equations, including correct use of neighbor indices and features.
3. Make missing labels explicit (for example, masked losses) and document treatment of interval-valued targets.
4. Use leakage-aware train/validation/test splits and report per-task metrics on held-out materials.
5. Require a trained checkpoint for normal prediction, and label synthetic/demo inference clearly.
6. Evaluate uncertainty calibration on held-out data before using MC Dropout uncertainty for decisions.

Until these steps are complete, treat training and prediction runs as pipeline demonstrations rather than evidence of predictive performance.
