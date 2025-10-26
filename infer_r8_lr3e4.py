#!/usr/bin/env python3
"""
Lightweight CLI inference for mBART50 + LoRA (r8, lr=3e-4) checkpoint.

Defaults:
- Adapter dir: C:\\Users\\joren\\Desktop\\THESIS\\Moodeng-MT\\mbart-lora-r8-lr3e4\\final-adapter
- Test CSV:    test_quality_split_v2.csv (expects columns `src`, `tgt`)
- Samples:     5

Prints SRC / PRED / REF to stdout.
"""

import os
import sys
import json
import argparse
from typing import List, Dict

import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


def clean_adapter_config(config_path: str) -> str:
    """Return a path to a cleaned adapter config if extra keys exist.

    Some exported adapters include optional keys unsupported by certain PEFT versions.
    We filter them out to improve compatibility. If anything fails, we return the
    original path unchanged.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        drop_keys = {
            'corda_config', 'eva_config', 'loftq_config', 'megatron_config',
            'megatron_core', 'qalora_group_size', 'use_dora', 'use_qalora', 'use_rslora'
        }
        cleaned = {k: v for k, v in cfg.items() if k not in drop_keys}
        out_path = config_path.replace('.json', '_cleaned.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2)
        return out_path
    except Exception:
        return config_path


def load_model(adapter_dir: str):
    """Load base mBART50 and attach LoRA adapter from `adapter_dir`."""
    base_model_name = 'facebook/mbart-large-50-many-to-many-mmt'
    base = MBartForConditionalGeneration.from_pretrained(base_model_name)

    # Clean adapter config if present
    cfg_path = os.path.join(adapter_dir, 'adapter_config.json')
    if os.path.exists(cfg_path):
        cleaned = clean_adapter_config(cfg_path)
        adapter_dir = os.path.dirname(cleaned)

    try:
        model = PeftModel.from_pretrained(base, adapter_dir)
    except Exception:
        # Fallback to base if adapter fails to load
        model = base

    tok = MBart50Tokenizer.from_pretrained(base_model_name)
    tok.src_lang = 'tl_XX'
    tok.tgt_lang = 'en_XX'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    return model, tok


@torch.inference_mode()
def translate_batch(model, tokenizer, texts: List[str]) -> List[str]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}

    bos_id = tokenizer.lang_code_to_id.get('en_XX', tokenizer.eos_token_id)
    out = model.generate(
        **enc,
        forced_bos_token_id=bos_id,
        max_length=128,
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=4,
        length_penalty=1.0,
        repetition_penalty=1.2,
        early_stopping=True,
    )
    raw = tokenizer.batch_decode(out, skip_special_tokens=True)
    cleaned: List[str] = []
    for text in raw:
        # Heuristic de-duplication of a repeated trailing sentence
        parts = [p.strip() for p in text.replace("…", ".").split(".") if p.strip()]
        if len(parts) >= 2 and parts[-1].lower() == parts[-2].lower():
            parts = parts[:-1]
        fixed = ". ".join(parts)
        if text.endswith(".") and not fixed.endswith("."):
            fixed += "."
        cleaned.append(fixed)
    return cleaned


def run_inference(adapter_dir: str, csv_path: str, num_samples: int, start_index: int = 0):
    model, tok = load_model(adapter_dir)

    if not os.path.exists(csv_path):
        print(f"✗ Test CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if 'src' not in df.columns or 'tgt' not in df.columns:
        print("✗ CSV must contain columns 'src' and 'tgt'")
        sys.exit(1)

    sub = df.iloc[start_index:start_index + max(1, num_samples)].copy()
    src_list = [str(s) for s in sub['src'].tolist()]
    ref_list = [str(s) for s in sub['tgt'].tolist()]

    preds = translate_batch(model, tok, src_list)

    print(f"Loaded {len(src_list)} samples from {csv_path}")
    for i, (src, pred, ref) in enumerate(zip(src_list, preds, ref_list), start=1):
        print(f"\n=== Sample {i} ===")
        print(f"SRC : {src}")
        print(f"REF : {ref}")
        print(f"PRED: {pred}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run inference for mbart-lora-r8-lr3e4 on a CSV test set')
    parser.add_argument('--adapter_dir', type=str,
                        default=r'C:\\Users\\joren\\Desktop\\THESIS\\Moodeng-MT\\mbart-lora-r8-lr3e4\\final-adapter',
                        help='Path to LoRA adapter directory (contains adapter_config.json and adapter weights)')
    parser.add_argument('--csv', type=str, default='test_quality_split_v2.csv',
                        help='CSV file with columns src,tgt')
    parser.add_argument('-n', '--num', type=int, default=5, help='Number of samples to run')
    parser.add_argument('--start', type=int, default=0, help='Start index in the CSV')
    return parser.parse_args()


def main():
    # Ensure Windows consoles print UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONUTF8', '1')

    args = parse_args()
    run_inference(args.adapter_dir, args.csv, args.num, args.start)


if __name__ == '__main__':
    main()


