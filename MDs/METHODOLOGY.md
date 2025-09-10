## Methodology

This section details the end‑to‑end model training pipeline for Filipino → English translation, from dataset preparation and CalamanCy‑enhanced preprocessing to fine‑tuning mBART‑50 with LoRA and final evaluation using the BLEU metric.

### 1. Data Sources

- Primary parallel dataset: `filipino_english_parallel_corpus.csv` with columns `id`, `text` (Filipino), `preprocessed_text` (Filipino cleaned), and `english_translation` (English).
- Enhanced training dataset (generated): `full_enhanced_parallel_corpus.csv` with columns `src`, `tgt`, `complexity_score`, `quality_score`, `tagalog_complexity`, `is_augmented`, `uses_calamancy`.

### 2. Preprocessing Pipeline (CalamanCy‑enhanced)

We employ a two‑tier preprocessing strategy implemented in `enhanced_preprocessing.py`:

1) CalamanCy priority loading
   - Attempts Tagalog models: `tl_calamancy_md-0.2.0`, `tl_calamancy_lg-0.2.0`, `tl_calamancy_trf-0.2.0` (falls back to spaCy multilingual models if unavailable).
   - Ensures sentence boundary detection via `sentencizer`.

2) Linguistic feature extraction and quality control
   - Tagalog‑aware tokenization handling contractions (e.g., `Di'ba → Di ba`).
   - Complexity computation with Tagalog signals (verb affixes such as `mag-`, `nag-`, reduplication, particles `ng/sa/ang/ay`, VSO patterns, entities). The result is stored as `tagalog_complexity` and rolled into `complexity_score`.
   - Quality validation produces `quality_score` with robust fallbacks (lenient thresholding if strict checks are unavailable).
   - Morphological augmentation (verbs and nouns) to expand training coverage; augmented rows are flagged with `is_augmented = True`.

3) Dataset enhancement
   - `batch_process_calamancy.py` processes the corpus in batches (default 500) to manage memory and provide progress/backup files.
   - Output: `full_enhanced_parallel_corpus.csv`, used directly by the training script. This avoids rerunning CalamanCy each training run unless data or preprocessing settings change.

### 3. Train/Validation Split

- The training script (`model_training_enhanced.py`) performs an 80/20 split by default (or uses the dataset’s preset splits if provided). Stratification is indirectly driven by curriculum filtering (Section 5).

### 4. Model and Tokenizer

- Base model: `facebook/mbart-large-50-many-to-many-mmt` (`MBartForConditionalGeneration`).
- Tokenizer: `MBart50Tokenizer` with language codes `src_lang = tl_XX`, `tgt_lang = en_XX` and sentencepiece backend.
- Mixed precision and device placement: CUDA if available, otherwise CPU. AMP/autocast is enabled only on CUDA.

### 5. Parameter‑Efficient Fine‑Tuning (LoRA)

- Implemented via `peft` with LoRA adapters on supported modules (e.g., attention/linear layers). LayerNorm targets are excluded to avoid PEFT incompatibilities.
- Benefits: fewer trainable parameters, faster convergence, smaller checkpoints.

### 6. Curriculum Learning via Linguistic Complexity

- We define progressive phases (Simple → Medium → Complex → Mixed) using thresholds on `complexity_score`/`tagalog_complexity`.
- For each phase/epoch group, the dataloader filters samples `complexity_score ≤ phase_threshold`, gradually exposing the model to harder examples.

### 7. Loss Functions and Regularization

- Cross‑Entropy (baseline) always available via `outputs.loss`.
- Label Smoothing Loss to reduce over‑confidence.
- Focal Loss to emphasize hard examples (later phases).
- R‑Drop Loss for consistency regularization (used in mixed phase).
- Dynamic phase‑based mixture: e.g.,
  - Simple: CE only
  - Medium: 0.7 CE + 0.3 Label Smoothing
  - Complex: 0.5 CE + 0.3 Label Smoothing + 0.2 Focal
  - Mixed: 0.4 CE + 0.3 Label Smoothing + 0.2 Focal + 0.1 R‑Drop

### 8. Optimization and Scheduling

- Optimizer: AdamW.
- Learning rate schedule: cosine with warmup (`get_cosine_schedule_with_warmup`).
- Gradient accumulation simulates larger effective batch sizes.
- AMP (CUDA only): `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` with `GradScaler(enabled=(device.type=="cuda"))`.

### 9. Training Loop

- For each curriculum phase:
  1) Filter data by `complexity_score` threshold.
  2) Iterate over batches; forward pass under AMP context when on GPU.
  3) Compute phase‑specific loss mixture.
  4) Backprop with gradient scaling; step optimizer/scheduler after accumulation steps.
  5) Track running loss; log phase statistics.

### 10. Validation and Early Stopping

- After each epoch, switch to `model.eval()` and compute validation loss on the held‑out split.
- BLEU is computed on a subset of validation samples (for fast iteration) via NLTK `sentence_bleu` with smoothing (`SmoothingFunction().method1`).
- Early stopping monitors validation loss and BLEU; the best model checkpoint is stored in `fine-tuned-mbart-tl2en-best/`.

### 11. Evaluation: BLEU Metric

- Tokenization: whitespace tokenization for references and hypotheses at evaluation time (consistent with the training tokenizer setting at inference).
- Metric: corpus‑level BLEU approximated by averaging sentence‑level BLEU on sampled validation pairs. For full evaluation, increase the sample size or compute corpus BLEU.
- Reporting:
  - `avg_val_loss` (validation loss)
  - `avg_bleu` (mean of sampled BLEU)
  - Best epoch metrics and patience status

### 12. Reproducibility and Efficiency

- Determinism: set seeds for NumPy/PyTorch where feasible (optional for speed).
- Caching: reuse `full_enhanced_parallel_corpus.csv` to bypass heavy NLP preprocessing on subsequent runs.
- Batch processing script `batch_process_calamancy.py` enables resumable, memory‑efficient generation of the enhanced dataset with progress logging and batch backups.

### 13. Commands (Minimal Reproduction)

1) Generate enhanced dataset once (batch processing):
```
python batch_process_calamancy.py
```

2) Analyze enhanced dataset (optional sanity check):
```
python analyze_enhanced_dataset.py
```

3) Train with emojis and UTF‑8 console:
```
python -X utf8 model_training_enhanced.py
```

### 14. Output Artifacts

- Enhanced dataset: `full_enhanced_parallel_corpus.csv`
- Best checkpoint: `fine-tuned-mbart-tl2en-best/`
- Final checkpoint: `fine-tuned-mbart-tl2en/`
- Logs: console output and optional files (e.g., `batch_processing.log`).

### 15. When to Regenerate the Enhanced Dataset

- New or modified data, changed preprocessing parameters, updated CalamanCy/spaCy versions, or different augmentation/quality thresholds.
- Otherwise, reuse the enhanced CSV to accelerate experiments.


