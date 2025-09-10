# 🇵🇭 Filipino→English Translation: Training and Inference

This guide documents the current training and inference workflow for the mBART‑50 + LoRA pipeline used in this project. It supersedes earlier notebook‑converted examples and aligns with `model_training_enhanced.py` and `translate_with_model.py`.

## 📁 Relevant Files

- `model_training_enhanced.py` – Primary training script (prefers enhanced corpus)
- `model_training.py` – Baseline training (no CalamanCy enhancements)
- `translate_with_model.py` – Inference/translation CLI using best adapter
- `requirements_translation.txt` – Dependencies for training/inference

## 🚀 Quick Start

### 1) Install dependencies
```bash
pip install -r requirements_translation.txt
```

### 2) Prepare data (recommended)
Generate the enhanced parallel corpus once:
```bash
python batch_process_calamancy.py
```
This creates `full_enhanced_parallel_corpus.csv` with columns `src`, `tgt`, and optionally `src_enhanced`/`tgt_enhanced`.

### 3) Train
Preferred (uses the enhanced CSV if available):
```bash
python model_training_enhanced.py
```
Baseline (no enhanced preprocessing):
```bash
python model_training.py
```

Artifacts:
- `fine-tuned-mbart-tl2en/` – checkpoints during training
- `fine-tuned-mbart-tl2en-best/` – best adapter and configs (used by inference)

### 4) Inference
Translate Filipino text using the best adapter:
```bash
python translate_with_model.py --text "kamusta ka?"
```
Or translate a CSV column:
```bash
python translate_with_model.py --input_csv filipino_english_parallel_corpus.csv \
  --src_col src --out_csv english_translation_from_preprocessed_texts.csv
```

## 📊 Dataset Formats

- Enhanced: `full_enhanced_parallel_corpus.csv` with `src`, `tgt` (+ optional `src_enhanced`, `tgt_enhanced`)
- Fallback: `filipino_english_parallel_corpus.csv` with `src`/`tgt` or `text`/`english_translation`

Example:
```csv
src,tgt
"Kamusta ka?","How are you?"
"Salamat sa tulong mo.","Thank you for your help."
```

## ⚙️ Key Training Details (summary)

- Base: `facebook/mbart-large-50-many-to-many-mmt`
- Tokenizer: `MBart50Tokenizer` with `src_lang=tl_XX`, `tgt_lang=en_XX`
- PEFT: LoRA on attention/FFN; mixed precision when CUDA is available
- Curriculum: optional complexity‑aware scheduling when enhanced features exist
- Selection: best adapter saved to `fine-tuned-mbart-tl2en-best/`

## 🧪 Troubleshooting

- CUDA OOM: reduce batch size or max length; use gradient accumulation
- Missing enhanced CSV: run `batch_process_calamancy.py` first, or rely on fallback in training script
- Windows tokenizer threads: set `TOKENIZERS_PARALLELISM=true` if needed

## 📚 See also

- `PREPROCESSING_PIPELINE.md` – End‑to‑end preprocessing and filtering
- `MODEL_TRAINING.md` – Full objectives, losses, schedules, and metrics
- `README_CALAMANCY_INTEGRATION.md` – Details of CalamanCy‑based enhancements
