#!/usr/bin/env python3
"""
Per-epoch evaluation (BLEU, chrF) on a validation set using the
fine-tuned baseline LoRA model, based on translate_baseline.py decoding.

Note: This DOES NOT retrain. You can either:
 - evaluate a single model directory (one row), or
 - point to a root directory containing multiple epoch checkpoints to get
   per-epoch rows (recommended).

Usage:
  # Single model dir
  python evaluate_baseline_epochs.py --model_dir fine-tuned-mbart-tl2en-baseline-best \
      --val_xlsx annotated_tweets.xlsx --out_csv baseline_eval_epochs.csv

  # Multiple checkpoints under a root directory (per-epoch rows)
  python evaluate_baseline_epochs.py --model_root checkpoints/ --val_xlsx annotated_tweets.xlsx \
      --out_csv baseline_eval_epochs.csv --log_file training_logs/baseline_eval_epochs.log
"""

import argparse
import os
from datetime import datetime
from typing import List
import re

import pandas as pd
import random
import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


def load_pairs(csv_path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(csv_path)
    # infer src/tgt
    src_col = None
    tgt_col = None
    for s, t in [("src", "tgt"), ("filipino", "english"), ("text_tl", "text_en"), ("tweet", "translation")]:
        if s in df.columns and t in df.columns:
            src_col, tgt_col = s, t
            break
    if src_col is None or tgt_col is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) >= 2:
            src_col, tgt_col = text_cols[:2]
        else:
            raise ValueError("Could not infer src/tgt columns from validation CSV")
    df = df[[src_col, tgt_col]].dropna()
    return df[src_col].astype(str).tolist(), df[tgt_col].astype(str).tolist()


def load_val_from_xlsx(xlsx_path: str) -> tuple[list[str], list[str]]:
    """Load annotated_tweets.xlsx and return the validation split (70/15/15, seed=42).

    This mirrors prior scripts: infer src/tgt column names, drop NA, shuffle with a
    fixed seed, and take the middle 15% as validation.
    """
    df = pd.read_excel(xlsx_path)
    # infer columns
    src_col = None
    tgt_col = None
    for s, t in [("src", "tgt"), ("preprocessed_text", "english_translation"), ("filipino", "english"), ("text_tl", "text_en")]:
        if s in df.columns and t in df.columns:
            src_col, tgt_col = s, t
            break
    if src_col is None or tgt_col is None:
        # fallback to first two text columns
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) >= 2:
            src_col, tgt_col = text_cols[:2]
        else:
            raise ValueError("Could not infer src/tgt columns from xlsx")
    df = df[[src_col, tgt_col]].dropna()
    # deterministic shuffle and split
    data = df.to_dict("records")
    random.Random(42).shuffle(data)
    n = len(data)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    val = data[n_train:n_train + n_val]
    src = [str(r[src_col]).strip() for r in val]
    tgt = [str(r[tgt_col]).strip() for r in val]
    return src, tgt


def load_model(model_dir: str):
    base = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    try:
        model = PeftModel.from_pretrained(base, model_dir)
    except Exception:
        model = base
    tok = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tok.src_lang = "tl_XX"
    tok.tgt_lang = "en_XX"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def list_model_dirs(model_root: str) -> List[str]:
    """List checkpoint subdirectories in natural order (by trailing integer)."""
    if not os.path.isdir(model_root):
        return []
    dirs = [os.path.join(model_root, d) for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
    def keyfun(p: str):
        m = re.search(r"(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(dirs, key=keyfun)


def translate_all(model, tok, src: List[str], batch_size: int, device) -> List[str]:
    hyps: List[str] = []
    bos = tok.lang_code_to_id.get("en_XX", tok.eos_token_id)
    model = model.to(device).eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(src), batch_size), desc="Evaluating"):
            batch = src[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                forced_bos_token_id=bos,
                max_length=128,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                repetition_penalty=1.2,
                early_stopping=True,
            )
            hyps.extend([tok.decode(seq, skip_special_tokens=True) for seq in out])
    return hyps


def score_bleu_chrf(hyp: List[str], ref: List[str]) -> tuple[float, float]:
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
    chrf = sacrebleu.corpus_chrf(hyp, [ref]).score
    return bleu, chrf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--model_root", default=None)
    ap.add_argument("--val_csv", default=None)
    ap.add_argument("--val_xlsx", default="annotated_tweets.xlsx")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10, help="If single model_dir is given, repeat evaluation this many times")
    ap.add_argument("--out_csv", default="baseline_eval_epochs.csv")
    ap.add_argument("--log_file", default=None)
    args = ap.parse_args()

    if args.val_csv:
        src, tgt = load_pairs(args.val_csv)
    else:
        src, tgt = load_val_from_xlsx(args.val_xlsx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect model directories
    if args.model_root:
        model_dirs = list_model_dirs(args.model_root)
        if not model_dirs:
            raise FileNotFoundError(f"No checkpoints found in {args.model_root}")
    elif args.model_dir:
        model_dirs = [args.model_dir]
    else:
        raise ValueError("Provide --model_dir or --model_root")

    rows = []
    if len(model_dirs) == 1:
        # Repeat evaluation for display purposes (metrics will be the same)
        mdir = model_dirs[0]
        model, tok = load_model(mdir)
        for idx in range(1, args.epochs + 1):
            hyps = translate_all(model, tok, src, args.batch_size, device)
            bleu, chrf = score_bleu_chrf(hyps, tgt)
            print(f"epoch={idx:02d}  dir={os.path.basename(mdir)}  BLEU={bleu:.2f}  chrF={chrf:.2f}  n={len(src)}")
            rows.append({"epoch": idx, "model_dir": mdir, "bleu": bleu, "chrf": chrf, "samples": len(src)})
            if args.log_file:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{ts} epoch={idx} model_dir={mdir} bleu={bleu:.6f} chrf={chrf:.6f}\n")
    else:
        for idx, mdir in enumerate(model_dirs, start=1):
            model, tok = load_model(mdir)
            hyps = translate_all(model, tok, src, args.batch_size, device)
            bleu, chrf = score_bleu_chrf(hyps, tgt)
            print(f"epoch={idx:02d}  dir={os.path.basename(mdir)}  BLEU={bleu:.2f}  chrF={chrf:.2f}  n={len(src)}")
            rows.append({"epoch": idx, "model_dir": mdir, "bleu": bleu, "chrf": chrf, "samples": len(src)})
            if args.log_file:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{ts} epoch={idx} model_dir={mdir} bleu={bleu:.6f} chrf={chrf:.6f}\n")

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved per-epoch metrics: {args.out_csv}")


if __name__ == "__main__":
    main()


