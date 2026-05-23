# Running training on Colab

CPU training is not viable for this project — a single 30-epoch run on the smallest model (BaselineLSTM) takes ~3.5 hours, and the full six-model grid is on the order of 30+ hours. The intended workflow is to do data preprocessing locally (where wfdb and scipy are happy) and then push the cached windows to Colab for the GPU-bound training stage.

This guide covers the proposed model in `notebooks/03_train_main_model.ipynb`. The same recipe works for any of the other five models — change the `--model` flag in cell 6.

## One-time setup

1. **Preprocess locally.** Run `./scripts/01_download_data.sh && ./scripts/02_preprocess.sh` on a workstation with enough RAM (~16 GB is enough after the Phase 2 fixes). The output is `data/processed/sddb/windows.npz` (~280 MB) and `data/processed/sddb/split.json`.
2. **Upload to Drive.** Copy the entire `data/processed/sddb/` folder to `My Drive/ecg_project/data/processed/sddb/`. Only the two files above are needed — you do not need to upload `data/raw/`.
3. **Push the repo to GitHub.** The notebook clones a public repo; the easiest path is a private fork that you authenticate with a fine-grained PAT, or a public repo if you are comfortable with that.

## Per-session steps

1. Open the notebook in Colab: File -> Open notebook -> GitHub tab -> paste the repo URL.
2. **Set the runtime to GPU.** Runtime -> Change runtime type -> Hardware accelerator: T4 GPU. The free tier is enough — the proposed model trains in ~60 min and the largest (ResNet1D) in ~90 min on a T4.
3. Edit cell 2 to replace `<user>` in `REPO_URL` with your GitHub user/org.
4. Run all cells top-to-bottom. Mounting Drive will pop a Google auth prompt the first time.

## Expected duration

| Model                | T4 wall-clock (30 epochs) |
|----------------------|---------------------------|
| BaselineLSTM         | ~25 min                   |
| CNNLSTMAttention     | ~60 min                   |
| TransformerEncoderModel | ~40 min                |
| ResNet1D             | ~90 min                   |
| InceptionTime1D      | ~55 min                   |
| TCN1D                | ~45 min                   |

These are rough estimates from comparable model sizes; actual numbers depend on what hardware Colab assigns you on the day. Add ~5 min of fixed overhead per notebook for the clone + pip install + data stage-in.

## If you get disconnected mid-training

Colab idles out after ~90 minutes of inactivity in the browser tab, and free tier sessions can be reclaimed at any time. Two mitigations:

* The trainer writes `best.pt` and the run-level `meta.json` to disk after every validation-improvement epoch, so the latest checkpoint is durable even if the session dies. Cell 9 copies the run directory back to Drive at the end — if you reconnect after a death, the partial run is gone unless you copied earlier. For long runs, periodically run the cell 9 contents manually.
* For very long runs (the full six-model grid), use `scripts/03_train_all.sh` from a Colab terminal rather than the notebook. Pipe output to a file in Drive (`./scripts/03_train_all.sh --device cuda 2>&1 | tee /content/drive/MyDrive/ecg_project/train_all.log &`) and check the log via the file browser without keeping the tab focused.

## After training

The notebook ends by copying `runs/cnn_lstm_attention/<timestamp>/` back to `MyDrive/ecg_project/runs/cnn_lstm_attention/<timestamp>/`. Pull that directory down to your local repo to run the cross-dataset evaluation and the explainability notebooks (which we expect to be cheap enough to run locally on CPU).
