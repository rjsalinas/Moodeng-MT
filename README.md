## MoodX-MT: Dataset Setup and Model Training Guide

This guide walks you through preparing the dataset, preprocessing and annotation, and running LoRA fine-tuning for mBART.

### 1) Prerequisites
- **Python environment**: Activate the provided virtual environment if you plan to use it:
```powershell
moodeng_env\Scripts\Activate.ps1
```
- **CUDA (optional but recommended)**: If your machine has a supported NVIDIA GPU and CUDA, PyTorch will automatically use it.
- Ensure your dataset JSON exists under `dataset/`.

### 2) Dataset placement
Place your collected TweetTaglish JSON file in the `dataset/` directory. Its name should look like:
```
dataset_tweettaglish-extraction---remaining-links-part-3_2025-08-14_00-47-25-188.json
```

### 3) Preprocess the dataset
The preprocessing script reads the JSON in `dataset/`, cleans and normalizes the data, and saves a cleaned CSV.

Run from the project root:
```powershell
python python-script\preprocess.py
```
Expected output:
- Cleaned data saved to `python-script/init-preprocess/cleaned_tweets.csv`

Notes:
- The script targets the TweetTaglish JSON (like the filename above) and performs normalization/filters before writing the CSV.

### 4) Annotate (auto-translate) with Google AI Studio API
The annotation script translates cleaned Taglish tweets into English and writes a translated CSV.

Requirements:
- A Google AI Studio API key (Gemini). Set it in your environment before running.

Windows PowerShell (current session only):
```powershell
$env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```

Run the annotator from the project root:
```powershell
python python-script\annotate.py
```

Expected output:
- Translations saved to `python-script/annotated-preprocess/translated_tweets.csv` (or similarly named translated CSV as configured in the script).

Tips:
- Make sure `python-script/init-preprocess/cleaned_tweets.csv` exists before running `annotate.py`.
- If you encounter quota/auth issues, re-check the `GOOGLE_API_KEY` value and your API access in Google AI Studio.

### 5) LoRA fine-tuning of mBART
Open the notebook and run all cells:
```
python-script/lora-ft-mbart.ipynb
```
In your IDE or Jupyter, select "Run All". The notebook expects the preprocessed and (optionally) translated data prepared by steps 3 and 4.

### 6) Parallel corpus (optional)
Parallel text data is available under `corpus-parallel-txt/` (`train`, `val`, `test` in `.en`/`.tl`). You can integrate or evaluate with these as your workflow requires.

### 7) Git note on CSVs
CSV files are important artifacts in this project (cleaned/translated). Ensure your `.gitignore` does not exclude them if you intend to track these files.

### 8) Paths quick reference
- Input JSON: `dataset/your_dataset.json`
- Cleaned CSV: `python-script/init-preprocess/cleaned_tweets.csv`
- Translated CSV: `python-script/annotated-preprocess/translated_tweets.csv`
- LoRA notebook: `python-script/lora-ft-mbart.ipynb`

### 9) Troubleshooting
- If `python` is not recognized, try `py` or ensure your environment is activated.
- If GPU is not used, verify your PyTorch install supports CUDA and your drivers are up to date.


