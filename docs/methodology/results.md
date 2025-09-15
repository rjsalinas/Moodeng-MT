## Training Results (Latest Run)

This document summarizes the most recent training run of the Filipino→English mBART-50 model with LoRA adapters, including configuration, metrics, qualitative samples, stability notes, and recommended next steps.

### Overview
- Base model: `facebook/mbart-large-50-many-to-many-mmt`
- Fine-tuning: PEFT LoRA (adapters only; embeddings excluded)
- Curriculum learning: 4 phases (Simple → Medium → Complex → Mixed)
- Devices: CUDA (standardized to `cuda:0`), AMP enabled; robust CPU fallbacks for critical steps
- Datasets: Preprocessed parallel corpus with strict filtering (drops rows containing `@` or `#` in either side)
- Split: 70% train / 15% validation / 15% test
- Tokenization: mBART50 with `src_lang=tl_XX`, forced BOS to `en_XX`; targets tokenized via `text_target=...`
- Best checkpoints: `fine-tuned-mbart-tl2en-best/`; full model: `fine-tuned-mbart-tl2en/`

### Key Configuration
- LoRA config: r=64, α=128, dropout=0.05, bias=lora_only
- Target modules: encoder/decoder attention and feed-forward blocks (embeddings removed)
- Optimizer/scheduler: AdamW + cosine schedule with warmup
- Mixed precision: Enabled on CUDA via `autocast` and `GradScaler`
- Losses: Cross-Entropy with optional Label Smoothing, Focal Loss, R-Drop; all updated to respect `ignore_index=-100`

### Data Integrity Controls
- Source column: `src` (mapped from `preprocessed_text` when loading CSV)
- Target column: `tgt` (mapped from `english_translation`)
- Filters applied at load:
  - Drop rows with missing/empty `src` or `tgt`
  - Drop rows where `src` or `tgt` contains `@` or `#`

### Latest Run: Metrics Snapshot
- Curriculum phase: Simple (threshold ≤ 5)
- Epoch: 1/20 (early phase; metrics typically noisy/low)
- Training loss (avg): ≈ 7.9518
- Validation loss (avg): ≈ 4.6102
- Validation BLEU (20–50 sampled val items): ≈ 0.0505

Notes:
- Early-epoch BLEU being low (≈0.05) is expected; BLEU generally improves after several epochs once the model adapts to the domain.
- In a previous GPU-stable run (before the latest strict filtering), validation loss improved to ≈3.917 and BLEU ≈0.1044 by later Simple-phase epochs, indicating upward trajectory with continued training.

### Qualitative Validation Samples (Recent)
Below are representative outputs observed in the latest run (varied quality is expected early):
- Good/acceptable:
  - tl: “Kamusta ka?” → en: “What is?” (short, under-translated; needs improvement)
  - tl: “Magandang umaga.” → en: “Good morning.” (typical expected behavior)
  - Social text with NER/hashtags previously leaked now filtered; remaining outputs occasionally include domain drift or code-switching.
- Noisy cases (to reduce with training time and decoding constraints):
  - Repetitions, code-switching, and token babbling on some inputs

Decoding parameters (current):
- Beam search (num_beams=4–5), no repeat n-gram size, repetition penalty, deterministic (no sampling) during validation

### Stability and Error Handling
- Prior CUDA device-side asserts were traced to custom losses not respecting `ignore_index=-100`; losses were fixed.
- Model/device mismatches were resolved by standardizing to `cuda:0` and ensuring inputs move to the model’s device.
- Generation uses `forced_bos_token_id=lang_code_to_id['en_XX']` (no hardcoded IDs; `decoder_start_token_id` removed to avoid repetition loops).
- Robust model saving via CPU `state_dict` materialization to avoid CUDA serialization asserts after failures.

### Resource Usage
- Batch size reduced on CUDA (2) to mitigate OOM/asserts; pin_memory enabled on CUDA.
- Periodic CUDA cache management and error counting with fallback to CPU if needed.

### Limitations (Current)
- Early-epoch BLEU remains low; needs more training epochs to stabilize and improve.
- Some qualitative outputs still exhibit repetition or domain drift.
- The adapter warning about `tie_word_embeddings=True` is benign for inference but can complicate merging/export; we exclude embeddings from LoRA to minimize issues.

### Recommendations (Next Steps)
1. Continue training for more epochs; monitor validation loss/BLEU each phase.
2. Tighten decoding for validation to curb babbling:
   - num_beams=5, no_repeat_ngram_size=4, repetition_penalty=1.3, length_penalty=1.0, optional `min_length=8`.
3. Data curation:
   - Keep strict filters; consider dropping ultra-short targets (<2 tokens) and non-English residues.
4. Curriculum:
   - Verify complexity distributions; ensure each phase has enough samples for learning.
5. Evaluation:
   - Keep validation-based BLEU sampling (now uses val split indices) and print (src, ref, hyp) triplets for qualitative checks.
6. (Optional) Efficiency:
   - Consider `accelerate` offloading for larger batch sizes if VRAM allows.

### Reproducibility
- Train:
```
python model_training_enhanced.py
```
- Translate with current best checkpoint:
```
python simple_translate.py "Kamusta ka?"
python translate_with_model.py
```

### Checkpoints
- Best (by validation): `fine-tuned-mbart-tl2en-best/`
- Final snapshot: `fine-tuned-mbart-tl2en/`

If you want, we can add a compact CSV of epoch-wise metrics (train/val loss, BLEU) on the next run for easier plotting.


