## MoodX-MT: Dataset Setup, Training, and Demo

This guide covers dataset preparation, preprocessing, annotation, training, and running the Streamlit demo.

### 1) Environment
- Activate the local environment:
```powershell
moodeng_env\Scripts\Activate.ps1
```
- CUDA is used automatically if available.

### 2) Dataset placement
Put your TweetTaglish JSON under `dataset/`. Filenames typically look like:
```
dataset_tweettaglish-extraction---remaining-links-part-3_2025-08-14_00-47-25-188.json
```

### 3) Preprocess (cleaning)
Reads all JSON files in `dataset/`, cleans/normalizes, and writes a CSV.
```powershell
python python-script\preprocess.py
```
Output: `python-script/init-preprocess/cleaned_tweets.csv`

### 4) Annotate (auto-translate with Google AI Studio)
Translates cleaned Taglish tweets to English and writes a translated CSV.
1. Set your API key:
```powershell
$env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```
2. Run the annotator:
```powershell
python python-script\annotate.py
```
Output: `python-script/annotated-preprocess/translated_tweets.csv`

### 5) Build/curate parallel corpus (optional)
- Cleaned parallel text exists under `corpus-parallel-txt/` (`*.en.cleaned`, `*.tl.cleaned`).
- Utility: `python-script/create_corpus.py` (if you need to generate or merge corpora).

### 6) Train: LoRA fine-tuning (mBART50)
Open and run all cells in the notebook:
```
python-script/lora-ft-mbart.ipynb
```
Artifacts are saved under `python-script/mbart-lora-finetuned/` and its checkpoints.

### 7) Streamlit demo (inference UI)
Run the web demo using the fine-tuned model:
```powershell
streamlit run app.py
```
Notes:
- `app.py` loads the LoRA adapters from `./python-script/mbart-lora-finetuned`.
- Ensure the folder exists (from training) or provide a compatible model there.

### 8) Paths quick reference
- Input JSON: `dataset/your_dataset.json`
- Cleaned CSV: `python-script/init-preprocess/cleaned_tweets.csv`
- Translated CSV: `python-script/annotated-preprocess/translated_tweets.csv`
- Notebook: `python-script/lora-ft-mbart.ipynb`
- Fine-tuned model: `python-script/mbart-lora-finetuned/`
- Demo entrypoint: `app.py`

### 9) Git note on CSVs
CSV outputs are important artifacts; ensure `.gitignore` does not exclude them if you need them tracked.

### 10) Troubleshooting
- `python` not recognized? Activate the env or use `py`.
- GPU not used? Verify CUDA-capable PyTorch and drivers.

