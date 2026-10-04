# PhysioNet ECG Image Digitization

Inference notebook for the Kaggle competition [PhysioNet - Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization). It converts photos and scans of 12-lead ECG printouts into the digital signals (mV) that the competition asks for and writes `submission.csv`.

Nothing is trained in this repository. The notebook runs pretrained models from two public Kaggle datasets and was run as a Kaggle notebook with internet off.

## Dataset

- Competition data: ECG images in `test/` and `train/`, `test.csv` (lead, sampling rate, number of rows to predict) and `sample_submission.parquet`. Each page shows three rows of four leads (2.5 s each) and a 10 s lead II rhythm strip. The data is not included here, download it from the competition page. It is released under CC BY 4.0.
- Pretrained models:
  - [hengck23/hengck23-demo-submit-physionet](https://www.kaggle.com/datasets/hengck23/hengck23-demo-submit-physionet): stage 0 and stage 1 code and weights (page normalisation and rectification).
  - [takashisomeya/physionet-final-submission-models](https://www.kaggle.com/datasets/takashisomeya/physionet-final-submission-models): six stage 2 checkpoints from the 2nd place solution.

## Approach

1. **Stage 0 (hengck23, ResNet18d U-Net):** finds the page orientation and 9 reference points and warps the page onto a fixed frame.
2. **Stage 1 (hengck23, ResNet34 U-Net):** finds the printed grid and rectifies the page to 1700 x 2200 px, where 1 mV is 79 px.
3. **Stage 2 (Takashi Someya, EfficientNet U-Nets):** two whole-page models and four lead-window models predict a probability map of the trace. The maps of all models and of the horizontally flipped page are averaged with the weighting of the 2nd place submission. The stage 2 networks are re-implemented in PyTorch and timm, because the original code needs `segmentation_models_pytorch`. The released checkpoints load with `strict=True`.
4. **Stage 3:** soft centroid of the probability map per column, conversion to mV, FFT resampling to the length in `test.csv`, and split into the 12 leads.

Other parts of the notebook:

- One worker process per GPU. Workers share the image list and the faster GPU takes more images.
- A time budget. The full ensemble is used for every image if it fits into the 9 hour limit, otherwise a cheaper ensemble is used for some images (level table in the notebook).
- A check on a few training ECGs with known ground truth, scored with an approximation of the competition metric (SNR after a shift of up to 0.2 s and offset removal).
- The submission ids and row count are compared with `sample_submission.parquet` before the notebook finishes.

## Results

The 2nd place write-up reports 23.37 (public) and 23.27 (private) for the six-model ensemble used in stage 2. This notebook only runs inference with those models, nothing is retrained or tuned. If the time budget forces a cheaper ensemble the score can be lower.

Score of this notebook's submission: [add from the Kaggle Submissions page]

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── physionet-ecg-image-digitization.ipynb   # full pipeline, one notebook
```

The notebook writes its own helper modules (`ecg_models.py`, `ecg_pipeline.py`, `ecg_worker.py`, `cc3d_compat.py`) to `/kaggle/working/ecg_infer/`.

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is needed for a full run. `allow_cpu=True` in the notebook config runs on CPU but is very slow. `nvidia-smi` must be available to list the GPUs.

## Usage

On Kaggle:

1. Create a notebook and import `notebooks/physionet-ecg-image-digitization.ipynb`.
2. Add the three inputs: the competition data and the two datasets above.
3. Set the accelerator to GPU (2 x T4 if available) and turn internet off.
4. Run the notebook with Save Version (Save & Run All). It writes `/kaggle/working/submission.csv`.
5. Submit the notebook to the competition.

The notebook is written for Kaggle's folder layout (`/kaggle/input`, `/kaggle/working`). To run it elsewhere, recreate these folders with the competition data and the two datasets.

## Credits

- Stage 0 and 1 code and weights: hengck23.
- Stage 2 models and ensemble: Takashi Someya (2nd place).

## License

No license is set for this repository yet. The pretrained models and the stage 0/1 code belong to their authors and keep their own licenses.
