# 🧠 MODEL TRAINING (mBART‑50 + LoRA) for Filipino→English

## What this covers
- Goals, data, and preprocessing assumptions
- Model, tokenizer, and adaptation strategy
- Losses, schedules, regularization
- How to run, outputs, and evaluation

## Objectives
Fine‑tune MBART‑50 for Filipino→English using parameter‑efficient LoRA adapters, leveraging a Filipino‑aware preprocessing pipeline for robust translation of social‑media text.

## Data
- Preferred input: `full_enhanced_parallel_corpus.csv` produced by `batch_process_calamancy.py` with columns: `src`, `tgt`, `src_enhanced`, `tgt_enhanced`.
- Fallback: `filipino_english_parallel_corpus.csv` with `src`/`tgt` or `text`/`english_translation`.

## Model and tokenizer
- Base: `facebook/mbart-large-50-many-to-many-mmt`
- Architecture: `MBartForConditionalGeneration`
- Tokenizer: `MBart50Tokenizer` with `src_lang = tl_XX`, `tgt_lang = en_XX`
- Adaptation: LoRA via `peft` targeting attention and FFN modules; best adapters saved under `fine-tuned-mbart-tl2en-best/`.

## Training scripts
- Enhanced pipeline (uses enhanced CSV if present):
```bash
python model_training_enhanced.py
```
- Baseline pipeline (no CalamanCy enhancements):
```bash
python model_training.py
```

## Key training details
- Sequence length: typical max 128 tokens
- Optimizer: AdamW; schedule: cosine with warmup
- Regularization: label smoothing, focal loss (for hard examples), R‑Drop consistency
- Mixed precision: enabled (AMP) when available
- Splits: train/validation from the provided CSV
- Generation config (eval/demo): beam search, length penalty, repetition control

## Monitoring and evaluation
- Metrics: training/validation loss; sentence‑level BLEU with smoothing
- Early selection: best adapter based on validation performance saved to `fine-tuned-mbart-tl2en-best/`
- Optional: integrate with standard Transformers callbacks if configured (TensorBoard/W&B)

## Outputs
- fine-tuned-mbart-tl2en/ (checkpoints during training)
- fine-tuned-mbart-tl2en-best/ (best adapter + configs)
- Console logs; optional batch logs from preprocessing

## Recommended workflow
```bash
# 1) Enhance the parallel corpus (fast, reproducible)
python batch_process_calamancy.py

# 2) Train with the enhanced dataset
python model_training_enhanced.py

# (Optional) Train a baseline for ablation
python model_training.py
```

## Reproducibility notes
- Keep `full_enhanced_parallel_corpus.csv` under version control with metadata; maintain `batch_processing.log` for provenance.
- Record seed, learning‑rate schedule, and LoRA hyperparameters for reported runs.

## Troubleshooting
- If CUDA OOM occurs, reduce batch size or sequence length; use gradient accumulation.
- If the enhanced CSV is missing, the enhanced script can preprocess on the fly, but `batch_process_calamancy.py` first is recommended.
- Enable tokenizers parallelism on Windows PowerShell if needed:
```powershell
$env:TOKENIZERS_PARALLELISM="true"
```

---
For Filipino‑aware preprocessing features, see `ENHANCED_FEATURES.md`. For step‑by‑step preprocessing, see `PREPROCESSING_PIPELINE.md`.

