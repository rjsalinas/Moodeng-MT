#!/usr/bin/env python3
"""
Evaluate chrF for one or many checkpoints.

Usage examples:
  # Evaluate a single fine-tuned dir on the test CSV
  python evaluate_chrf.py --model_root fine-tuned-mbart-tl2en-baseline-best \
                         --data_csv test_quality_split_v2.csv

  # Evaluate all checkpoint subfolders inside a parent dir
  python evaluate_chrf.py --model_root checkpoints/ --data_csv test_quality_split_v2.csv

The script will print a small table and write a CSV with columns:
  [model_dir, chrf, bleu, samples]

Notes:
 - chrF is computed with sacrebleu.corpus_chrf
 - BLEU is included for reference
 - If you want per-epoch scores, pass a parent folder that contains epoch
   checkpoints (e.g., checkpoints/epoch-1, epoch-2, ...). If you only have the
   best folder, you'll just get a single row.
"""

import argparse
import os
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


def load_pairs(data_csv: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(data_csv)
    # Try to infer src/tgt columns
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
            raise ValueError("Could not infer src/tgt columns from CSV")
    df = df[[src_col, tgt_col]].dropna()
    src = df[src_col].astype(str).tolist()
    tgt = df[tgt_col].astype(str).tolist()
    return src, tgt


def list_model_dirs(model_root: str) -> List[str]:
    # If the provided path itself contains an adapter or config, treat as single
    files = set(os.listdir(model_root)) if os.path.isdir(model_root) else set()
    if {"adapter_model.safetensors", "adapter_config.json"}.issubset(files) or "config.json" in files:
        return [model_root]
    # Else enumerate subdirectories
    subdirs = [os.path.join(model_root, d) for d in sorted(os.listdir(model_root))]
    return [d for d in subdirs if os.path.isdir(d)]


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


def translate_all(model, tok, src_texts: List[str], batch_size: int, device) -> List[str]:
    outs: List[str] = []
    bos = tok.lang_code_to_id.get("en_XX", tok.eos_token_id)
    model = model.to(device).eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(src_texts), batch_size), desc="Translating"):
            batch = src_texts[i : i + batch_size]
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
            outs.extend([tok.decode(seq, skip_special_tokens=True) for seq in out])
    return outs


def score_bleu_chrf(hyp: List[str], ref: List[str]) -> Tuple[float, float]:
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
    chrf = sacrebleu.corpus_chrf(hyp, [ref]).score
    return bleu, chrf


def to_0_100(x: float) -> float:
    """Normalize a score to 0..100 if it appears to be 0..1."""
    return x * 100.0 if x <= 1.0 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="fine-tuned-mbart-tl2en-baseline-best")
    ap.add_argument("--data_csv", default="test_quality_split_v2.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", default="chrf_eval_results.csv")
    args = ap.parse_args()

    src, tgt = load_pairs(args.data_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for mdir in list_model_dirs(args.model_root):
        model, tok = load_model(mdir)
        hyp = translate_all(model, tok, src, args.batch_size, device)
        bleu, chrf = score_bleu_chrf(hyp, tgt)
        bleu_n = to_0_100(bleu)
        chrf_n = to_0_100(chrf)
        print(f"{mdir}: BLEU={bleu_n:.2f}  chrF={chrf_n:.2f}  samples={len(src)}")
        rows.append({"model_dir": mdir, "bleu": bleu_n, "chrf": chrf_n, "samples": len(src)})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()


