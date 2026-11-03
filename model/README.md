---
license: mit
tags:
- astronomy
- tabular-classification
- scikit-learn
---

# Gamma Event Lab model card

An MLP classifier trained **from scratch** on ten numeric features from simulated events in the UCI MAGIC Gamma Telescope dataset. Intended for educational exploration of background rejection, not real observations or scientific decisions. Input order and inference code are documented in the [project repository](https://github.com/pedrojbDev/gamma-event-lab).

## Training and evaluation

R. Bock, MAGIC Gamma Telescope (2004), [UCI](https://doi.org/10.24432/C52C8B), CC BY 4.0. The dataset is transformed by grouping identical feature vectors, splitting, scaling, and training; raw data are not included. Seed 42; 60/20/20 stratified unique-feature group split. Train-only standardization; 32/16 ReLU hidden units. Threshold selected on validation for an empirical 5% false-positive-rate target. The fixed 200-iteration run reached its iteration cap without convergence.

On 3,806 held-out simulated events: ROC AUC 0.9288; hadron false-positive rate 0.0463; gamma recall 0.5702. Confusion TN/FP/FN/TP: 1278/62/1060/1406. Most-frequent-class baseline: 0/1340/0/2466. See `metrics.json` for exact counts and `training-manifest.json` for source and artifact hashes.

## Limitations

The data are simulated, background frequency is artificially low, and groups share simulator assumptions. The validation threshold provides no guaranteed test or operational FPR. Scores are not calibrated real-world probabilities. Inputs are extracted feature vectors, not images. No deployment or scientific validity is claimed. The JSON weights are loaded by the project's NumPy inference code; do not use pickle.
