# Forecasting Ventricular Arrhythmias from ECG

An explainable deep learning pipeline that predicts whether a life-threatening ventricular arrhythmia (VT or VF) will begin in the next 30 seconds, given the previous 60 seconds of ECG. Course project for BTH DV2586/DV2646 (Deep Machine Learning).

The problem is framed as **forecasting**, not classification. The model never sees any signal from inside the prediction horizon.

```
input window:        [t-60s ─────────── t]
prediction horizon:                       [t ───── t+30s]   ← will VT/VF start here?
```

## What's in here

- Six PyTorch models: a stacked LSTM baseline, our proposed CNN-LSTM-Attention model, a Transformer encoder, and three published SOTA comparators (ResNet1D in the Hannun-2019 style, InceptionTime, and a Temporal Convolutional Network).
- A reproducible pipeline that downloads SDDB, MIT-BIH, and INCART from PhysioNet, preprocesses them, trains all models, and emits the metric tables and figures used in the report.
- Patient-disjoint train/val/test splits, with a unit test that fails if any patient leaks across splits.
- Explainability: attention heatmaps, input-gradient saliency, Grad-CAM, and SHAP DeepExplainer.
- Cross-dataset evaluation (train on SDDB, test on MIT-BIH and INCART).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all_experiments.sh
```

Full instructions, including Colab setup and the four-step manual variant, are in `INSTRUCTIONS.md`.

## Repository layout

```
config/         hyperparameters per model (YAML)
src/data/       download, preprocessing, windowing, Dataset
src/models/     the six model architectures
src/training/   model-agnostic trainer, losses, scheduler
src/evaluation/ metrics, cross-dataset eval
src/explainability/  attention, saliency, Grad-CAM, SHAP
scripts/        the shell entry points
notebooks/      the five .ipynb deliverables
tests/          patient-split and no-leakage invariants
report/         LaTeX source (IEEEtran)
```

## Reproducing the report

The report quotes numbers from the CSVs in `results/metrics/` produced by `scripts/run_all_experiments.sh` with `--seed 42`. Changing the seed will move AUROC by roughly 0.01–0.03.

## Datasets

All three are public PhysioNet datasets. The download script (`scripts/01_download_data.sh`) fetches them into `data/raw/`. Total download is about 1.1 GB.

| Dataset | Role |
|---|---|
| SDDB (Sudden Cardiac Death Holter) | training and held-out test |
| MIT-BIH Arrhythmia Database | cross-dataset evaluation |
| INCART (St. Petersburg) | secondary cross-dataset eval |

## License

Academic use only. Datasets retain their original PhysioNet licenses.
