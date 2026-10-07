# PhysioNet ECG Image Digitization

This repository contains my complete work for the Kaggle competition [PhysioNet - Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization). The task is to convert photos and scans of printed 12-lead ECGs into digital signals in millivolts and generate a valid `submission.csv` file.

The competition metric is signal-to-noise ratio (SNR) in decibels. Higher scores are better.

## Notebooks

| Notebook | Method | Public score |
|---|---|---:|
| [physionet-ecg-image-digitization.ipynb](physionet-ecg-image-digitization.ipynb) | Inference using pretrained deep-learning models | 23.38235 |
| [ecg-own-code-baseline.ipynb](ecg-own-code-baseline.ipynb) | Classical image-processing pipeline developed from scratch | 21.15672 |

The pretrained-model notebook was submitted first. I then developed the own-code notebook as an independent baseline to measure the improvement provided by the pretrained models. On the public test set, the pretrained pipeline improved the score by approximately 2.2 dB.

## Dataset

The competition data is available from the [Kaggle competition data page](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data). The dataset is not included in this repository because of its size and competition restrictions.

The competition dataset contains:

- ECG images in `train/` and `test/`.
- Ground-truth CSV files for the training ECGs.
- `train.csv`.
- `test.csv`, containing the lead name, sampling rate and number of rows to predict.
- `sample_submission.parquet`.

Each ECG image contains three rows of four leads, with each lead covering 2.5 seconds. Lead II is also shown as a 10-second rhythm strip.

I also created a local benchmark using PTB-XL records 00001 to 00016. I rendered these records into ECG images using [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit), then created distorted copies to test the robustness of my pipelines.

The PTB-XL dataset was published by Wagner et al. in *Scientific Data*, volume 7, article 154, 2020, and is released under the CC BY 4.0 licence. The ecg-image-kit project is released under the BSD-3-Clause licence.

## Project Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── physionet-ecg-image-digitization.ipynb
└── ecg-own-code-baseline.ipynb
```

The notebooks contain my implementation, experiments, evaluation code and submission-generation code.

The submitted notebook creates the following helper modules in `/kaggle/working/ecg_infer/` when it runs:

```text
ecg_infer/
├── ecg_models.py
├── ecg_pipeline.py
├── ecg_worker.py
└── cc3d_compat.py
```

## My Approach

I implemented two different approaches to the ECG digitisation problem.

The first approach uses pretrained segmentation models. The second approach is a classical computer-vision baseline that does not require model training or external pretrained weights.

Neither notebook trains a model. Therefore, neither approach uses feature engineering or a train-validation split in the conventional supervised-learning sense.

## Pretrained-Model Pipeline

The pretrained pipeline contains four main stages.

### Stage 0: Page Detection and Alignment

I implemented the first stage using a pretrained ResNet18d U-Net model. The model detects the page orientation and nine reference points on the ECG image.

I use these reference points to apply a perspective transformation and warp the photographed or scanned ECG page onto a fixed reference frame.

### Stage 1: Grid Detection and Rectification

I implemented the second stage using a pretrained ResNet34 U-Net model. This model detects the printed ECG grid and helps rectify the page.

The page is transformed to a resolution of approximately 1700 × 2200 pixels. At this resolution, 1 mV corresponds to approximately 79 pixels.

### Stage 2: ECG Trace Segmentation

I re-implemented the stage 2 segmentation models in PyTorch using timm.

The stage 2 pipeline uses EfficientNet-based U-Net models operating on:

- The complete ECG page.
- Individual lead windows.
- Horizontally flipped versions of the input page.

The predictions from the available models are combined by averaging their probability maps. I also verified that the released checkpoints load correctly with `strict=True`.

The full ensemble contains:

- Two whole-page models.
- Four lead-window models.
- Predictions from the original page.
- Predictions from the horizontally flipped page.

To handle GPU memory and execution-time limits, I implemented multiple inference configurations. The pipeline can use the full ensemble when sufficient resources are available and a smaller ensemble when necessary.

### Stage 3: Signal Extraction and Conversion

For every image column, I calculate a soft centroid from the predicted ECG trace probability map. This produces the vertical position of the ECG waveform for each column.

I then:

- Convert pixel coordinates into millivolts.
- Resample the extracted signals using FFT-based resampling.
- Match the required output length from `test.csv`.
- Split the reconstructed signal into the 12 individual leads.
- Generate the required submission identifiers.

The final output uses identifiers in the format:

```text
{ecg}_{index}_{lead}
```

I also validate the generated submission against `sample_submission.parquet` by checking:

- The identifier values.
- The number of rows.
- The required submission structure.

## Classical Image-Processing Pipeline

I developed the own-code baseline independently using classical image-processing techniques. This notebook does not use pretrained neural-network weights or code from other competition participants.

The pipeline contains the following steps.

### 1. Ink and Grid Extraction

I process the colour channels of each ECG image to create:

- An ink map representing the ECG trace and printed text.
- A grid map representing the red or coloured ECG grid.

These maps provide the basis for page detection, orientation estimation and waveform extraction.

### 2. Page Detection and Perspective Correction

I detect the page boundaries using the grid information and estimate the page corners.

I then apply a perspective transformation to correct:

- Camera perspective.
- Page skew.
- Non-uniform scaling.
- Rotation caused by image capture.

### 3. Tilt and Orientation Estimation

I estimate page tilt from gradients in the grid map.

I determine the page orientation using:

- The ink profile.
- The calibration pulse.
- The spatial arrangement of the ECG traces.
- The detected grid structure.

The pipeline handles the main orientation cases, including 0°, 90° and 180° rotations.

### 4. Grid Spacing and Physical Calibration

I estimate the grid spacing using:

- Autocorrelation of the grid signal.
- A comb-like grid fit.
- The detected spacing between major and minor grid lines.

This allows me to estimate the physical scale in millimetres per pixel and convert the extracted waveform into millivolts.

### 5. Baseline Detection

I estimate the baseline of each ECG strip from the row profile of the ink map.

The baseline estimation is used to separate the waveform from the surrounding grid and printed content.

### 6. ECG Trace Tracking

I extract the ECG waveform using a dynamic-programming path through the detected ink runs.

The tracking method searches for the most likely continuous ECG path through each lead strip while accounting for:

- Trace continuity.
- Noise.
- Broken lines.
- Grid interference.
- Local gaps in the waveform.

I also apply apex correction to improve the reconstruction of sharp ECG spikes, such as QRS complexes.

### 7. Signal Conversion and Lead Splitting

After extracting the waveform path, I convert the pixel coordinates into millivolts using the estimated grid scale.

I then split the reconstructed page into the 12 required leads and format the signals according to the competition submission format.

If an image cannot be processed reliably, the pipeline produces a zero-filled signal for that image rather than breaking the complete submission process.

## Evaluation

I implemented a local approximation of the competition scoring metric, `snr_ecg`.

The local metric includes:

- A possible time shift of up to 0.2 seconds for each lead.
- Removal of signal offset.
- Power pooling across the 12 leads.
- Final scoring in decibels.

I used this metric to evaluate the classical pipeline on my local benchmark and to check selected training ECGs when the competition data was available.

## Results

### Kaggle Results

| Pipeline | Public score | Private score |
|---|---:|---:|
| Pretrained-model pipeline | 23.38235 | 23.27202 |
| Classical image-processing pipeline | 21.15672 | Not available |

The pretrained pipeline achieved a public score of 23.38235 and a private score of 23.27202.

The classical pipeline achieved a public score of 21.15672. The pretrained pipeline therefore improved the public score by approximately 2.2 dB.

### Local Benchmark

I created a local benchmark using:

- 16 PTB-XL ECG records.
- ECG images rendered using ecg-image-kit.
- Distorted copies of the rendered images.
- 48 images in total.

The mean local benchmark scores were:

| Pipeline | Mean local score |
|---|---:|
| Classical image-processing pipeline | 14.05 dB |
| Pretrained pipeline using the cheapest configuration | 26.88 dB |

The pretrained pipeline was evaluated using one stage 2 model without flip averaging for this local test. The difference between the local benchmark and the Kaggle results is much larger than the difference observed on the public leaderboard.

I did not identify the exact reason for this discrepancy. Therefore, the local benchmark should be treated as an engineering comparison between the two pipelines rather than as a direct prediction of leaderboard performance.

The own-code notebook contains the related ablations, error analysis and image-size experiments.

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

The classical image-processing notebook requires:

```text
numpy
pandas
opencv-python-headless
scipy
matplotlib
pyarrow
```

The pretrained-model notebook additionally requires:

```text
torch
timm
```

## Running the Classical Pipeline

The classical pipeline can run on CPU and does not require internet access or a GPU.

1. Import `ecg-own-code-baseline.ipynb` into a Kaggle notebook.
2. Add the PhysioNet competition dataset as an input.
3. Keep internet and GPU disabled.
4. Run the notebook using **Save Version** and **Save & Run All**.
5. The notebook generates `submission.csv`.
6. Submit the generated file to the competition.

The local benchmark sections require a `benchmark/` folder with the following structure:

```text
benchmark/
├── images/
├── records/
└── reference_final_pipeline_level3.csv
```

The benchmark can be placed next to the notebook or inside `/kaggle/input`.

If the benchmark data is not available, the benchmark sections are skipped. The saved notebook outputs still contain the benchmark results from my experiments.

## Running the Pretrained Pipeline

The pretrained pipeline requires a GPU and the pretrained model datasets.

1. Import `physionet-ecg-image-digitization.ipynb` into a Kaggle notebook.
2. Add the PhysioNet competition dataset as an input.
3. Add the `hengck23/hengck23-demo-submit-physionet` dataset.
4. Add the `takashisomeya/physionet-final-submission-models` dataset.
5. Select a GPU accelerator, preferably two T4 GPUs if available.
6. Disable internet access after adding the required inputs.
7. Run the notebook using **Save Version** and **Save & Run All**.
8. The notebook generates `submission.csv`.
9. Submit the generated file to the competition.

The two additional Kaggle datasets contain the pretrained checkpoints required by the pipeline:

- Stage 0 and stage 1 models used for page alignment and grid rectification.
- Six stage 2 checkpoints used for ECG trace segmentation.

## Submission Generation

Both notebooks generate a `submission.csv` file using the competition format.

The submission includes:

- The ECG identifier.
- The sample index.
- The lead name.
- The reconstructed ECG value in millivolts.

The identifier format is:

```text
{ecg}_{index}_{lead}
```

Before writing the final file, I check the generated identifiers and row count against `sample_submission.parquet`.

This prevents common submission errors such as:

- Missing ECG samples.
- Incorrect lead names.
- Incorrect row ordering.
- Extra rows.
- Missing rows.
- Incorrect identifier formatting.

## Main Findings

- The pretrained segmentation models produced the strongest Kaggle result, reaching 23.38235 dB on the public leaderboard and 23.27202 dB on the private leaderboard.
- The classical image-processing pipeline reached 21.15672 dB without model training or pretrained weights. It provides a fully independent baseline based on grid detection, geometric correction, calibration and dynamic-programming trace tracking.
- The local benchmark showed a much larger difference between the two approaches than the Kaggle leaderboard. This suggests that the local benchmark and competition test set contain different image characteristics or failure cases.
- The repository therefore provides two complete implementations: a high-performing pretrained deep-learning inference pipeline and an independently developed classical computer-vision baseline.
- Both pipelines produce competition-compatible ECG signal submissions.
