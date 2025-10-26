#!/usr/bin/env python3
"""
Bulk translate TL→EN on test_quality_split_v2.csv using three models:
- v1: fine-tuned-mbart-tl2en-baseline-best  → column: tgt_v1
- v2: mbart50-finetuned-LoRA-best           → column: tgt_v2
- v3: mbart50-finetuned-LoRA-best_v3       → column: tgt_v3

Output CSV columns: [src, tgt_v1, tgt_v2, tgt_v3]

Run:
  python bulk_translate.py
Optionally specify input/output:
  python bulk_translate.py --in test_quality_split_v2.csv --out test_quality_split_v2_translated.csv
"""

import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


def load_model_dir(model_dir: str):
    """Load base mBART and (optionally) attach LoRA adapters from model_dir."""
    base = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    # Try attaching adapters; if missing, just use base
    try:
        model = PeftModel.from_pretrained(base, model_dir)
    except Exception:
        model = base
    tok = MBart50Tokenizer.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    tok.src_lang = "tl_XX"
    tok.tgt_lang = "en_XX"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def generate_batch(model, tokenizer, texts, device):
    enc = tokenizer(
        list(texts), return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    bos = tokenizer.lang_code_to_id.get("en_XX", tokenizer.eos_token_id)
    with torch.no_grad():
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
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="test_quality_split_v2.csv")
    ap.add_argument("--out", dest="out", default="test_quality_split_v2_translated.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise FileNotFoundError(f"Input CSV not found: {args.inp}")

    df = pd.read_csv(args.inp)

    # Find source column robustly
    src_col = None
    for cand in ["src", "filipino", "text_tl", "tweet"]:
        if cand in df.columns:
            src_col = cand
            break
    if src_col is None:
        # Fallback: first object-type column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            raise ValueError("Could not find a text column for sources.")
        src_col = text_cols[0]

    src_texts = df[src_col].astype(str).fillna("").tolist()

    # Load models once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v1, tok_v1 = load_model_dir("fine-tuned-mbart-tl2en-baseline-best")
    model_v2, tok_v2 = load_model_dir("mbart50-finetuned-LoRA-best")
    model_v3, tok_v3 = load_model_dir("mbart50-finetuned-LoRA-best_v3")
    model_v1 = model_v1.to(device).eval()
    model_v2 = model_v2.to(device).eval()
    model_v3 = model_v3.to(device).eval()

    results_v1 = []
    results_v2 = []
    results_v3 = []

    bs = max(1, args.batch_size)
    for i in tqdm(range(0, len(src_texts), bs), desc="Translating"):
        batch = src_texts[i : i + bs]
        # v1
        try:
            out_v1 = generate_batch(model_v1, tok_v1, batch, device)
        except Exception:
            out_v1 = [""] * len(batch)
        # v2
        try:
            out_v2 = generate_batch(model_v2, tok_v2, batch, device)
        except Exception:
            out_v2 = [""] * len(batch)
        # v3
        try:
            out_v3 = generate_batch(model_v3, tok_v3, batch, device)
        except Exception:
            out_v3 = [""] * len(batch)

        results_v1.extend(out_v1)
        results_v2.extend(out_v2)
        results_v3.extend(out_v3)

    out_df = pd.DataFrame({
        "src": src_texts,
        "tgt_v1": results_v1,
        "tgt_v2": results_v2,
        "tgt_v3": results_v3,
    })
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  (rows: {len(out_df)})")


if __name__ == "__main__":
    main()



