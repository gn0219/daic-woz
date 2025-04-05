<details>
<summary> Setup (click to expand)</summary>

# Setup
## Clone this repository

```bash
git clone https://github.com/gn0219/daic-woz.git
cd daic-woz
```

## Create a virtual environment

Make sure Python is installed. If not, download it from the [official Python website](https://www.python.org/).

```bash
python -m venv .venv
```

Activate the virtual environment:

- **Windows**:

  ```bash
  .venv\Scripts\activate
  ```

- **Mac/Linux**:

  ```bash
  source .venv/bin/activate
  ```

## Install required packages

```bash
pip install -r requirements.txt
```
---

## Download the DAIC-WOZ dataset

### Option 1: Manual download

1. Go to the [DAIC-WOZ official site](https://dcapswoz.ict.usc.edu/)
2. Complete the EULA form and obtain the download link.
2. Download the `.zip` files manually.
3. Move all downloaded `.zip` files into the `downloads/` folder.

### Option 2: Download full dataset with scripts

Make sure to change “BaseUrl” to the actual download URL.

- **Mac/Linux users**:

  ```bash
  bash etc/download.sh
  ```

- **Windows users**:

  Open PowerShell and run:

  ```powershell
  etc\download.ps1
  ```

> These scripts will download and place the files into the `downloads/` folder.

## Unzip and organize files

Run the script below to extract the zip files and organize audio and transcript files:

```bash
python unzip_files.py
```

- Zip files in `downloads/` will be extracted to `unzipped_files/`.
- All `.wav` files will be moved to `wav_files/`.
- All transcript files (`_TRANSCRIPT.csv`) will be moved to `transcript_files/`.

## Project folder structure (after setup)

```plaintext
daic-woz/
│
├── downloads/               # .zip files go here
├── unzipped_files/          # Extracted contents
├── wav_files/               # All .wav audio files
├── transcript_files/        # Transcript CSV files
├── unzip_files.py           # Unzipping and file-moving script
├── src/                     # Source code modules
│   ├── ...
│   └── unzip_files.py 
├── ...
├── requirements.txt         # Python dependencies
└── etc/
    ├── download.sh          # For Mac/Linux
    └── download.ps1         # For Windows
```
</details>

# Analysis
## 0. extract_features.ipynb
After this process, we have `info_df.csv`, `utterance_features.csv`, and `smile_features.csv`.
- Done
  - Extract utterance related features from transcription files
  - Extract openSMILE features from raw wav files
  - `src/extract_features.py`
- TODO: Extract openSMILE features after silence removal

## 1. eda.ipynb
- Done
  - Visualize label information
  - Modularize visualization code `src/visualize_features.py`
    - histogram, numeric(boxplot or violinplot)
  - Visualize audio file `src/visualize_audio.py`
    - waveform, log spectrogram

## 2. evaluate_basic_models.ipynb
- Done
  - Implement code `src/evaluate_ml_models.py` by only using `train` and `test` (not `dev`)
    - Evaluate 7 models (including majority voting as a baseline model)
      -  Baseline, SVM, KNN, RF, LR, GB, XGB
    - Experiment with three different settings (smile features on raw data, utterance features, and merged version of them)
    - Perform two differnt feature selection methods by setting `use_feature_selection=True`
      - `selection_method='kbest'` or `selection_method='model'`

## 3. topological_features.ipynb
- Done (Seunghyun)
  - Initial visualization
- TODO: Expand to all UID (Gyuna)

## TODO: DL models

## Project folder structure (after Analysis)

```plaintext
daic-woz/
│
├── data/                           # CSV files (preprocessed / feature-extracted)
│   ├── df.csv
│   ├── info_df.csv
│   ├── smile_features.csv
│   ├── utterance_features.csv
│
├── downloads/                      # Raw .zip files (not used during analysis)
├── unzipped_files/                 # Extracted zip contents (not used)
├── wav_files/                      # Raw audio files (.wav) (used during feature extraction)
├── transcript_files/               # Transcription files (_TRANSCRIPT.csv) (used during feature extraction)
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── unzip_files.py              # Unzipping script (not used in analysis)
│   ├── extract_features.py         # Feature extraction logic
│   ├── evaluate_ml_models.py       # ML model pipeline
│   ├── visualize_features.py       # Feature-level plots
│   ├── visualize_audio.py          # Audio waveform / spectrogram visualization
│   └── visualize_results.py        # Performance visualization (TODO: add confusion matrix visualization)
│
├── 0. extract_features.ipynb       # Step 0 - Extract features from audio/transcripts
├── 1. eda.ipynb                    # Step 1 - Exploratory Data Analysis
├── 2. evaluate_basic_models.ipynb  # Step 2 - Classical ML model evaluation
├── 3. topological_features.ipynb   # Step 3 - Topological Data Analysis
│
├── README.md                       
├── requirements.txt               
└── .gitignore                     

```