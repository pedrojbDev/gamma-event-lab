# Gamma Event Lab

An educational classifier for **simulated** atmospheric Cherenkov telescope events. It demonstrates a practical trade-off: recover gamma-like events while limiting background acceptance. It is not a telescope instrument, an image classifier, or a validated scientific analysis.

## How it works

The UCI MAGIC Monte Carlo dataset supplies ten precomputed shower features. A checksum-verified download feeds a deterministic split that keeps identical feature vectors together and stratifies unique groups. A scaler and a two-layer neural network are trained from scratch on the training partition. The decision threshold is chosen **only on validation negatives** to target a 5% false-positive rate. The held-out test partition is used for final evaluation. A portable JSON artifact contains the scaler and network parameters; no pickle is loaded.

`UCI data → group split → train-only scaling + MLP → validation threshold → test metrics → JSON → FastAPI / browser demo`

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The repository contains the trained artifact; no dataset download is needed for inference.

```sh
uv sync --locked
uv run --locked uvicorn gamma_lab.app:app --reload
```

Open http://127.0.0.1:8000 for the interactive demo or `/docs` for the API schema. The ten features, in order, are `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`. Units and definitions are in the [dataset description](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope). Example (illustrative input, not a real event):

```sh
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"features":[100,20,2,0.4,0.2,0,0,0,20,100]}'
```

The response has `gamma_score`, `threshold`, `selected` and `simulation_only`; malformed, nonfinite, wrong-length or numerically extreme (absolute value above 1e9) vectors receive HTTP 422, and a missing artifact receives HTTP 503. This broad numerical safety bound is not an in-distribution detector. Scores are not real-world probabilities.

Reproduce training on CPU (downloads 1.5 MB of public data, checksum checked):

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run --locked python -m gamma_lab.train
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

Training overwrites `model/` artifacts and records the current Git commit; use a clean checkout to preserve provenance. Timing and numerical results can vary across platforms. See `model/training-manifest.json` and `model/metrics.json` for the recorded run.

## Recorded results

One fixed run, seed 42: 11,408 train / 3,806 validation / 3,806 test rows. Threshold 0.90604 produced validation FPR 4.93%; held-out FPR **4.63%**, gamma recall **57.02%**, ROC AUC **0.9288**. Test confusion matrix (negative=hadron, positive=gamma): TN 1,278, FP 62, FN 1,060, TP 1,406. The training-majority baseline predicts every test event as gamma: TN 0, FP 1,340, FN 0, TP 2,466. Run time was 7.14 s and peak process RSS 141,692 KiB in the recorded environment; the fixed 200-iteration budget ended with a convergence warning. Results are not evidence of convergence or performance on actual observations.

## Limitations and data

The source uses Monte Carlo events and explicitly underrepresents hadrons versus real observations. Random grouped splits are not instrument-, simulation-configuration-, or time-held-out, and duplicate grouping does not eliminate other correlations. The validation-derived FPR is an estimate, **not a guarantee**; changing class priors or simulation conditions can change performance. No uncertainty intervals are reported. Do not use the scores for operational or scientific decisions.

Dataset: R. Bock, *MAGIC Gamma Telescope* (2004), UCI Machine Learning Repository, [doi:10.24432/C52C8B](https://doi.org/10.24432/C52C8B), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This project modifies the data by grouping, splitting, scaling and training a model; raw data are not redistributed. Code is MIT licensed; dataset rights remain separate. See `model/README.md` for the model card.
