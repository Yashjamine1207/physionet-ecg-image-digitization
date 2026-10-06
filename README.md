# PhysioNet ECG Image Digitization

Work for the Kaggle competition [PhysioNet - Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization): turn photos and scans of printed 12-lead ECGs into digital signals (mV) and write `submission.csv`. The score is a signal-to-noise ratio in dB, higher is better.

The repository has two notebooks for the same task:

| Notebook | Method | Public score |
|---|---|---|
| [`physionet-ecg-image-digitization.ipynb`](physionet-ecg-image-digitization.ipynb) | Pretrained models of other competitors, inference only | 23.38235 (private 23.27202) |
| [`ecg-own-code-baseline.ipynb`](ecg-own-code-baseline.ipynb) | Own code, classical image processing, nothing trained | 21.15672 |

The pretrained notebook was submitted first. The own-code notebook was written afterwards as a baseline, to measure how much the pretrained models add. On the public test set they add about 2.2 dB.

## Dataset

Competition data from the [competition page](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data), not included in this repository: ECG images in `train/` and `test/`, ground-truth CSVs of the training ECGs, `train.csv`, `test.csv` (lead, sampling rate, number of rows to predict) and `sample_submission.parquet`. Each page shows three rows of four leads (2.5 s each) and a 10 s lead II rhythm strip.

Extra data used by the notebooks:

- Submitted notebook, two public Kaggle datasets with pretrained models:
  - [hengck23/hengck23-demo-submit-physionet](https://www.kaggle.com/datasets/hengck23/hengck23-demo-submit-physionet): stage 0 and 1 code and weights.
  - [takashisomeya/physionet-final-submission-models](https://www.kaggle.com/datasets/takashisomeya/physionet-final-submission-models): six stage 2 checkpoints from the 2nd place solution.
- Own-code notebook, local benchmark: PTB-XL records 00001 to 00016 (CC BY 4.0, Wagner et al., Scientific Data 7, 154, 2020) drawn to images with [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) (BSD-3-Clause), plus distorted copies. It is not included here.

## Approach

Neither notebook trains a model, so there is no feature engineering and no train/validation split.

**Submitted notebook (pretrained models)**

1. Stage 0 (hengck23, ResNet18d U-Net): finds the page orientation and 9 reference points and warps the page onto a fixed frame.
2. Stage 1 (hengck23, ResNet34 U-Net): finds the printed grid and rectifies the page to 1700 x 2200 px, where 1 mV is 79 px.
3. Stage 2 (Takashi Someya, EfficientNet U-Nets): two whole-page and four lead-window models predict a probability map of the trace. The maps of all models and of the horizontally flipped page are averaged. The networks are re-implemented in PyTorch and timm, and the released checkpoints load with `strict=True`.
4. Stage 3: soft centroid of the map per column, conversion to mV, FFT resampling to the length in `test.csv`, split into the 12 leads.

The notebook also runs one worker per GPU, uses a cheaper ensemble for some images if the full one does not fit into the time limit, and scores a few training ECGs.

**Own-code notebook (no pretrained weights, no code from other participants)**

1. Ink and grid maps from the colour channels.
2. Page corners from the grid, perspective warp, tilt from the grid gradients, orientation (0, 90, 180 degrees) from the ink profile and the calibration pulse.
3. Grid spacing from autocorrelation and a comb fit, which gives mm per pixel.
4. Baselines of the four strips from the row profile of the ink.
5. Dynamic-programming (Viterbi) path through the ink runs of every strip, apex correction for sharp spikes, conversion to mV.
6. Split into the 12 leads. An image that fails gets zeros.

**Evaluation.** Both notebooks use a local approximation of the competition metric (`snr_ecg`: per lead a shift of up to 0.2 s, offset removed, powers pooled over the 12 leads, in dB). The own-code notebook scores its local benchmark (48 images) and, when the competition data is attached, training ECGs.

**Submission.** Both notebooks write `submission.csv` with ids `{ecg}_{index}_{lead}` and check the ids and the row count against `sample_submission.parquet`.

## Results

Scores from the Kaggle Submissions page (both submitted after the deadline):

| Notebook | Public score |
|---|---|
| Submitted notebook, pretrained models | 23.38235 (private 23.27202) |
| Own-code notebook | 21.15672 |

The 2nd place write-up reports 23.37 (public) and 23.27 (private) for the six-model stage 2 ensemble that the submitted notebook runs. The own-code notebook is 2.2 dB lower on the public set.

Local benchmark of the own-code notebook (16 ECGs drawn with ecg-image-kit and distorted copies, 48 images, scores in dB, mean over images): own code 14.05, pretrained pipeline 26.88. The pretrained value comes from its cheapest setting (one of the six stage 2 models, no flip averaging). This gap of 12.8 dB is much larger than the 2.2 dB gap on Kaggle. I did not find out why, so the benchmark should not be read as a prediction of the leaderboard score. Ablations, error analysis and the image size test are in the own-code notebook.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── physionet-ecg-image-digitization.ipynb   # pretrained models, submitted notebook
└── ecg-own-code-baseline.ipynb              # own code, with saved outputs
```

The submitted notebook writes its helper modules (`ecg_models.py`, `ecg_pipeline.py`, `ecg_worker.py`, `cc3d_compat.py`) to `/kaggle/working/ecg_infer/` when it runs.

## Installation

```bash
pip install -r requirements.txt
```

The own-code notebook needs only numpy, pandas, opencv-python-headless, scipy, matplotlib and pyarrow. torch and timm are for the submitted notebook.

## Usage

**Own-code notebook** (CPU, no internet):

1. Import `ecg-own-code-baseline.ipynb` into a Kaggle notebook and add the competition data as input. Internet and GPU can stay off.
2. Run it with Save Version (Save & Run All). It writes `submission.csv`.
3. Submit the notebook to the competition.

The benchmark sections need a `benchmark/` folder (`images/`, `records/`, `reference_final_pipeline_level3.csv`) next to the notebook or in `/kaggle/input`. Without it they are skipped, and the saved outputs in the notebook show their results.

**Submitted notebook** (GPU):

1. Import `physionet-ecg-image-digitization.ipynb` into a Kaggle notebook.
2. Add three inputs: the competition data and the two datasets listed above.
3. Set the accelerator to GPU (2 x T4 if available) and turn internet off.
4. Run it with Save Version (Save & Run All), then submit the notebook to the competition.

## Credits and license

- Stage 0 and 1 code and weights: hengck23. Stage 2 models and ensemble: Takashi Someya (2nd place). Both keep their own licenses.
- Competition: [PhysioNet - Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization) on Kaggle.
- The license of this repository is in `LICENSE`.
