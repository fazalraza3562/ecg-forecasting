# INSTRUCTIONS.md

How to run this project end-to-end. Written for someone — you, your group, or a grader — who has just cloned the repository and wants to reproduce the numbers in the report.

If you only have ten minutes, skip to **§3 Quick start**. The rest of this document explains what each step does and why.

---

## 1. Environment

You need Python 3.10 or newer. Earlier versions trip on some of the type hints.

### Option A: pip + venv (recommended for Colab and local)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate ecg
```

### GPU notes

The code runs on CPU but training is slow there. A full training pass of the CNN-LSTM-Attention model takes roughly 6 minutes on a T4 (Colab free tier) and about 3.5 hours on a recent laptop CPU. We developed against a single NVIDIA T4. The dataloader uses `num_workers=2` by default; drop it to 0 if Colab kills workers.

### A thing that will probably break for someone

The `wfdb` package occasionally fails to install on Apple Silicon if Xcode command-line tools are missing. If `pip install wfdb` complains, run `xcode-select --install` first and try again.

---

## 2. Getting the data

Three PhysioNet datasets. All public.

| Dataset | Used for | Approx size |
|---|---|---|
| SDDB (Sudden Cardiac Death Holter) | training + held-out test | ~700 MB |
| MIT-BIH Arrhythmia Database | cross-dataset evaluation | ~80 MB |
| INCART (St. Petersburg) | secondary cross-dataset eval | ~300 MB |

The download script handles all three:

```bash
bash scripts/01_download_data.sh
```

This populates `data/raw/sddb/`, `data/raw/mitdb/`, and `data/raw/incartdb/`. Total disk use is about 1.1 GB. On Colab, mount Drive and point `data_root` in `config/default.yaml` at the mounted path so you don't re-download every session.

---

## 3. Quick start (the four commands)

Assuming environment is set up and data is downloaded:

```bash
bash scripts/02_preprocess.sh        # ~10 min: filter, normalize, window
bash scripts/03_train_all.sh         # ~45 min on T4: trains all 6 models
bash scripts/04_evaluate.sh          # ~5 min: computes all metric tables
bash scripts/05_explainability.sh    # ~5 min: SHAP + attention figures
```

Or, all-in-one:

```bash
bash scripts/run_all_experiments.sh
```

After this:

- `results/metrics/` — the CSV tables that the report quotes
- `report/figures/` — the figures included in `main.tex`
- `runs/` — trained model checkpoints

---

## 4. What each step does

### 4.1 Preprocessing (`02_preprocess.sh`)

This calls `python -m src.data.preprocess --config config/default.yaml`. It:

1. Loads ECG records using the `wfdb` library.
2. Resamples everything to 250 Hz (SDDB is 250, MIT-BIH 360, INCART 257 — we unify).
3. Applies a band-pass Butterworth filter (0.5–40 Hz) to kill baseline wander and high-frequency noise.
4. Per-record z-score normalization.
5. Locates VT/VF onset annotations.
6. Generates **forecasting windows**: 60 s of input ECG, 30 s of look-ahead, with a 5 s stride. A window is labeled positive if any VT/VF onset annotation falls in the 30 s horizon.
7. Saves windows to `data/processed/<dataset>/windows.npz` plus a `meta.json` recording the patient ID of each window.

The patient-level split happens at this point too. SDDB patients are split 70/15/15 into train/val/test by patient ID. The split is seeded and saved to `data/processed/sddb/split.json` so re-running gives the same partition.

Before training, sanity-check the invariants:

```bash
pytest tests/
```

All tests should be green. They check the no-leakage property and patient disjointness.

### 4.2 Training (`03_train_all.sh`)

Six models, in sequence:

```
baseline_lstm           ~3 min
cnn_lstm_attention      ~6 min   ← our proposed model
transformer             ~9 min
resnet1d                ~7 min
inception1d             ~8 min
tcn                     ~5 min
```

Each is invoked as:

```bash
python -m src.training.train \
    --model <name> \
    --config config/<name>.yaml \
    --seed 42
```

Defaults: AdamW, lr 1e-3, cosine schedule, 30 epochs, early stopping with patience 5 on validation AUPRC, focal loss with γ=2 (the positive class is roughly 6% of windows). Mixed precision is on when CUDA is available.

Checkpoints go to `runs/<model>/<YYYYMMDD-HHMMSS>/best.pt`. The evaluation script picks up the latest one automatically.

### 4.3 Evaluation (`04_evaluate.sh`)

Produces, for each model:

- A row in `results/metrics/sddb_test.csv` (AUROC, AUPRC, sens@95spec, F1, FPR/h, lead-time median+IQR).
- A row in `results/metrics/cross_dataset.csv` for the MIT-BIH + INCART test.
- A per-patient AUROC boxplot, saved as `report/figures/per_patient_auroc.pdf`.
- ROC and PR curves: `report/figures/roc_pr_curves.pdf`.

For the proposed model only:

- The ablation table: `results/metrics/ablation.csv`.
- A failure-case figure: `report/figures/failure_cases.pdf` — two or three false-negative windows with attention overlay.

### 4.4 Explainability (`05_explainability.sh`)

Four figures for the report:

- `attention_heatmap.pdf` — averaged attention over true positives. Shows the model focuses on the 5–15 s pre-onset region.
- `saliency_examples.pdf` — input-gradient saliency on three true positives.
- `gradcam.pdf` — Grad-CAM on the last conv layer.
- `shap_summary.pdf` — SHAP DeepExplainer summary over a held-out subsample.

The SHAP step is the slowest — DeepExplainer is expensive on long sequences. We use a 200-sample background and a 200-sample foreground.

---

## 5. Reproducing the report's exact numbers

The report's tables and figures come from running the full pipeline with seed 42. Change the seed and expect AUROC to wobble by 0.01–0.03.

To regenerate everything:

```bash
SEED=42 bash scripts/run_all_experiments.sh
```

You should then be able to compile `report/main.tex` and have it match the submitted PDF.

---

## 6. Compiling the report

The report is in `report/main.tex`, IEEEtran class. The figure paths inside the .tex assume you've already run the pipeline, so that `report/figures/` is populated.

### On Overleaf

1. Make a new project. Upload everything in `report/`: `main.tex`, `references.bib`, `IEEEtran.cls`, and the `figures/` directory.
2. Compiler: **pdfLaTeX**.
3. Compile twice — first pass builds the `.aux` for BibTeX, second pass resolves cross-references.

### Locally with TeX Live

```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

If you see "missing figure" warnings, you forgot to run the pipeline first. Run `bash scripts/run_all_experiments.sh` from the repo root and try again.

---

## 7. The notebook for submission

The course requires `.ipynb` files. The main deliverable is `notebooks/03_train_main_model.ipynb`. It:

1. Mounts Drive (commented out if you're not on Colab).
2. Imports from `src/`.
3. Loads the preprocessed windows.
4. Builds and trains the CNN-LSTM-Attention model.
5. Evaluates on the SDDB test set.
6. Renders a couple of attention figures inline.

The other four notebooks (data exploration, preprocessing walkthrough, results, explainability) are supporting material. All five go into the submission zip.

---

## 8. Common problems

**"CUDA out of memory" during training.** Lower `batch_size` from 64 to 32 in the relevant config, or turn off mixed precision (`amp: false`).

**`wfdb` import fails.** On Linux or Mac try `pip install wfdb --no-binary=:all:`. On Windows stay on Python 3.10 or 3.11.

**Tests fail with `AssertionError: patient X appears in both train and test`.** The seeded split changed — usually because someone edited `split.json` by hand or bumped the dataset version. Delete `data/processed/sddb/split.json` and re-run preprocessing.

**Notebook hangs at "Loading windows".** The `.npz` cache is corrupt. Delete `data/processed/sddb/windows.npz` and re-run `02_preprocess.sh`.

**Loss goes to NaN in the first epoch.** Almost always a focal-loss numerical issue when a batch contains no positives. Drop the learning rate to 5e-4 and check that `--seed` is being respected (a stale bad checkpoint can otherwise get loaded).

---

## 9. What to hand in on Canvas

Submit a `.zip` (or `.rar`) containing:

- `report/main.tex`, `references.bib`, `IEEEtran.cls`, `figures/`, and the compiled `main.pdf`
- All five `.ipynb` files from `notebooks/`
- The entire `src/` tree
- `config/`, `scripts/`, `tests/`
- `requirements.txt`, `environment.yml`, `README.md`, `INSTRUCTIONS.md`, `CLAUDE.md`
- Do **not** include `data/raw/`, `data/processed/`, or `runs/`. They are large and reproducible.

A recipe:

```bash
bash scripts/make_submission_zip.sh
# produces submission_<group>_<date>.zip in the repo root
```

---

## 10. Workload distribution (edit before submitting)

The report's Section 1 contains a table describing who did what. Update it before submission. A reasonable three-person split:

| Member | Primary responsibility |
|---|---|
| Member A | Data pipeline, preprocessing, windowing, dataset splits, tests |
| Member B | Model implementations — CNN-LSTM-Attention plus two of the SOTA baselines |
| Member C | Evaluation, cross-dataset experiments, explainability, figures, third SOTA baseline |

All members contributed to experimental design, debugging, and writing.
