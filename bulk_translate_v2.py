#!/usr/bin/env python3
"""
Bulk translate TL→EN on test_quality_split_v2.csv using two models:
- baseline: facebook/mbart-large-50-many-to-many-mmt (base model)
- v3: mbart50-finetuned-LoRA-best_v3

Output CSV columns: [src, baseline, tgt_v3]

Run:
  python bulk_translate_v2.py
Optionally specify input/output:
  python bulk_translate_v2.py --in test_quality_split_v2.csv --out baseline_vs_v3_translated.csv
"""

import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


def load_baseline_model():
    """Load the base mBART model without any fine-tuning."""
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    tokenizer = MBart50Tokenizer.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    tokenizer.src_lang = "tl_XX"
    tokenizer.tgt_lang = "en_XX"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_finetuned_model(model_dir: str):
    """Load base mBART and attach LoRA adapters from model_dir."""
    base = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    # Try attaching adapters; if missing, just use base
    try:
        model = PeftModel.from_pretrained(base, model_dir)
    except Exception:
        model = base
    tokenizer = MBart50Tokenizer.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    tokenizer.src_lang = "tl_XX"
    tokenizer.tgt_lang = "en_XX"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


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
    ap.add_argument("--out", dest="out", default="baseline_vs_v3_translated.csv")
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
    print("Loading baseline mBART model...")
    baseline_model, baseline_tok = load_baseline_model()
    print("Loading fine-tuned LoRA model v3...")
    v3_model, v3_tok = load_finetuned_model("mbart50-finetuned-LoRA-best_v3")
    
    baseline_model = baseline_model.to(device).eval()
    v3_model = v3_model.to(device).eval()

    baseline_results = []
    v3_results = []

    bs = max(1, args.batch_size)
    for i in tqdm(range(0, len(src_texts), bs), desc="Translating"):
        batch = src_texts[i : i + bs]
        
        # Baseline model
        try:
            out_baseline = generate_batch(baseline_model, baseline_tok, batch, device)
        except Exception as e:
            print(f"Baseline model error: {e}")
            out_baseline = [""] * len(batch)
            
        # Fine-tuned v3 model
        try:
            out_v3 = generate_batch(v3_model, v3_tok, batch, device)
        except Exception as e:
            print(f"V3 model error: {e}")
            out_v3 = [""] * len(batch)

        baseline_results.extend(out_baseline)
        v3_results.extend(out_v3)

    out_df = pd.DataFrame({
        "src": src_texts,
        "baseline": baseline_results,
        "tgt_v3": v3_results,
    })
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  (rows: {len(out_df)})")
    print("Columns: [src, baseline, tgt_v3]")


if __name__ == "__main__":
    main()


