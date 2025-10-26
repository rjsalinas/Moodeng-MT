#!/usr/bin/env python3
"""
translate_v3: Use mbart50-finetuned-LoRA-best_v3 for TL→EN inference.
Optimized for speed - no METEOR or other evaluation metrics.
Uses the model trained with training_v3.py (original v1 parameters + cleaned CSV dataset).
"""

import sys
import os
import json
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


def clean_adapter_config(config_path: str) -> str:
    """Clean adapter config for compatibility"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # Remove problematic fields that can cause issues
        drop = ['corda_config','eva_config','loftq_config','megatron_config','megatron_core','qalora_group_size','use_dora','use_qalora','use_rslora']
        cleaned = {k: v for k, v in cfg.items() if k not in drop}
        out = config_path.replace('.json', '_cleaned.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2)
        return out
    except Exception:
        return config_path


def load_model(model_dir: str = "mbart50-finetuned-LoRA-best_v3"):
    """Load the model and tokenizer efficiently"""
    base = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    adapter_cfg = os.path.join(model_dir, "adapter_config.json")
    
    if os.path.exists(adapter_cfg):
        model_dir = os.path.dirname(clean_adapter_config(adapter_cfg))
    
    try:
        model = PeftModel.from_pretrained(base, model_dir)
    except Exception:
        model = base
    
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "tl_XX"
    tokenizer.tgt_lang = "en_XX"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


@torch.inference_mode()
def translate(text: str, model_dir: str = "mbart50-finetuned-LoRA-best_v3") -> str:
    """Fast translation without evaluation metrics"""
    model, tokenizer = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Encode input
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    
    # Generate translation with stricter decoding to reduce repetitions and hallucinations
    bos = tokenizer.lang_code_to_id.get("en_XX", tokenizer.eos_token_id)
    out = model.generate(
        **enc,
        forced_bos_token_id=bos,
        max_length=128,
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=4,
        length_penalty=1.0,
        repetition_penalty=1.2,
        early_stopping=True,
    )

    text_out = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    # Post-process: remove duplicated trailing sentence fragment
    # Split into sentences and dedupe last repeated sentence
    parts = [p.strip() for p in text_out.replace("…", ".").split(".") if p.strip()]
    if len(parts) >= 2 and parts[-1].lower() == parts[-2].lower():
        parts = parts[:-1]
    cleaned = ". ".join(parts)
    if text_out.endswith(".") and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python translate_v3.py 'Filipino text here'")
        return
    
    text = " ".join(sys.argv[1:])
    translation = translate(text)
    print(translation)


if __name__ == "__main__":
    main()
