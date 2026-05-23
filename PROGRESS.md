# PROGRESS

  Status as of 2026-05-23.
  Phases 1-5 complete; Phase
   6 (full training runs on
  Colab) is next.

  ## Phase 1 — Repository
  and data layer

  The repo has the layout
  CLAUDE.md describes:
  `src/{data,models,...}`,
  `config/`, `scripts/`,
  `tests/`, `notebooks/`,
  `report/`. The data layer
  (`src/data/`) downloads,
  preprocesses, windows, and
   serves the three
  PhysioNet datasets (SDDB,
  MIT-BIH, INCART, all
  present in `data/raw/`).
  Three test files cover the
   load-bearing invariants —
   `test_windows.py`
  (no-leakage /
  in-progress-skip /
  label-consistency),
  `test_onsets.py` (the SDDB
   onset rule), and
  `test_patient_split.py`
  (disjoint train/val/test
  by patient). All nine
  tests are green.

  ## Phase 2 —
  Memory-efficient
  preprocessing

  `src/data/preprocess.py`
  was rewritten to survive a
   16 GB-RAM box: float32
  throughout (with an
  explicit downcast after
  `sosfiltfilt`'s float64
  internals),
  first-two-leads only
  across all three datasets
  (INCART went from 12 -> 2
  leads), per-record `del` +
   `gc.collect()` so the
  largest live allocation is
   bounded, and a
  per-dataset save+free so
  we never hold two
  datasets' window tensors
  at once. Missing `.atr`
  files now raise a targeted
   `FileNotFoundError` so
  the record-loop log line
  names the cause.

  ## Phase 3 — SDDB onset
  rule, stride,
  configuration

  The SDDB onset rule was
  rebuilt from scratch
  because SDDB's `.atr`
  files do not contain
  rhythm-onset annotations —
   only beat-level symbols.
  We now infer VT onset as
  the first `V` beat in any
  run of >= 3 consecutive
  `V`-coded beats, then
  merge onsets within `onset
  _merge_window_seconds =
  60` so a single episode
  broken by occasional non-V
   beats is not
  double-counted. The window
   stride was bumped from 5
  s to 30 s in
  `config/default.yaml` to
  keep the
  windows-per-record count
  sane on multi-hour Holter
  traces.

  ## Phase 4 — Six model
  architectures

  All six models in
  `src/models/` are
  implemented, smoke-tested,
   and under the 5 M
  parameter cap. Each
  consumes `(B, 2, 15000)`
  and returns `(B,)`
  pre-sigmoid logits.

  | Model | Params | Notes |
  |---|--:|---|
  | `BaselineLSTM` | 150,625
   | Conv stem + 2-layer
  BiLSTM + mean-pool head |
  | `CNNLSTMAttention` |
  181,153 | Proposed model;
  Bahdanau attention,
  `last_attn_weights` cached
   |
  |
  `TransformerEncoderModel`
  | 102,593 | Conv stem,
  sinusoidal PE buffer, 3
  pre-norm encoder layers |
  | `ResNet1D` | 2,627,841 |
   8 residual blocks,
  channel ladder *capped at 
  128* (see limitations) |
  | `InceptionTime1D` |
  226,177 | 3 inception
  blocks, kernels 9/19/39 +
  pool branch |
  | `TCN1D` | 346,753 | 6
  dilated blocks, symmetric
  (non-causal) padding,
  weight_norm |

  ## SDDB data — what 
  survived preprocessing

  | | |
  |---|--:|
  | Windows | 24,196 |
  | Positive | 169 (**0.70 
  %**) |
  | Patients | 12 (`30, 31, 
  32, 34, 35, 36, 41, 45, 
  46, 49, 51, 52`) |
  | Split (train / val / 
  test) | 8 / 2 / 2 patients
   |
  | Tensor shape / dtype | 
  `(24196, 2, 15000)` 
  float32 |

  ## Known limitations

  **SDDB annotation 
  coverage.** Only 12 of 23 
  SDDB records ship with 
  `.atr` annotation files. 
  The other 11 (`33, 37, 38,
   39, 40, 42, 43, 44, 47,
  48, 50`) have signal data
  but no ground-truth onset
  labels and are correctly
  skipped during
  preprocessing. Two of the
  three SDDB cross-dataset
  patients reserved for test
   are therefore a small
  sample and per-patient
  metrics will be noisy.

  **Severe class 
  imbalance.** The 0.70 % 
  positive rate means a 
  naive model that always 
  predicts negative scores 
  99.30 % accuracy. This
  forces us into focal loss
  / AUPRC /
  sensitivity-at-95-spec
  rather than accuracy, and
  we'll need to be careful
  about confidence intervals
   on the positive class.

  **ResNet1D deviation from 
  spec.** The canonical 
  Hannun-style channel 
  ladder (32 -> 64 -> 128 ->
   256) lands at ~6.6 M 
  parameters with our
  8-block depth, over the 5
  M cap. Blocks 6-8 were
  capped at 128 channels
  instead of doubling to
  256, dropping the model to
   2.6 M. The deviation and
  its motivation are
  documented at the top of
  `src/models/resnet1d.py`.

  ## Phase 5 — Training pipeline

  `src/training/{losses.py,scheduler.py,train.py}` is in place. The trainer is a single CLI entry point — `python -m src.training.train --model <name> [--config ...] [--seed ...] [--device auto|cpu|cuda]` — that's fully model-agnostic: it picks the architecture from a `name -> class` registry and reads every hyperparameter from `config/default.yaml` overlaid with `config/<model>.yaml`. Each run writes `best.pt`, `interrupted.pt` (on Ctrl+C), `config.yaml` (effective merged snapshot), `train.log`, and `meta.json` (git SHA, seed, best AUPRC, training time, full config) to `runs/<model>/<timestamp>/`.

  Loss is `FocalLoss(γ=2)` computed from logits via the numerically stable `binary_cross_entropy_with_logits` form. Optimizer is `AdamW` with cosine LR decay and 10 % linear warmup, stepped per-batch. Mixed precision (`torch.amp.autocast` + `GradScaler`) is on only when the resolved device is CUDA; CPU runs stay in fp32. Validation runs under `model.eval()` and `torch.no_grad()`, computes AUROC and AUPRC via sklearn, and AUPRC is the early-stopping metric (AUROC is too easy to inflate at 0.7 % positive prevalence). Eleven losses-and-scheduler tests pass; trainer is verified by a 2-epoch smoke run rather than a unit test because it does too much I/O.

  **Smoke test result (BaselineLSTM, CPU, seed=42).** Two epochs took ~14 min wall-clock (~7 min/epoch, LSTM on CPU is the bottleneck). Train loss decreased 0.0624 -> 0.0123; validation AUROC moved off chance 0.503 -> 0.660; validation AUPRC is still tiny at 0.0057 but moving the right way at 0.6 % prevalence. Both `best.pt` and `interrupted.pt` were written; `meta.json` recorded `training_seconds=836.5` and `interrupted=true`.

  ## Phase 6 — Full training runs on Colab (next)

  CPU training is not viable: ~7 min/epoch on the smallest model means a single 30-epoch BaselineLSTM run is ~3.5 h, and the heavier ResNet1D / Transformer / TCN are several times that. End-to-end, six models x 30 epochs is on the order of 30+ hours of CPU compute. Phase 6 is to run the full training matrix on Colab (T4 or A100), pull the checkpoints back into `runs/`, then run cross-dataset evaluation (`src/evaluation/`) on the MIT-BIH and INCART splits.
