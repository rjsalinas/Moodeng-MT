#!/usr/bin/env python3
"""
Clean Baseline Filipino-to-English Translation Script
Minimal output - just shows input and translation
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel
import sys
import os
import json

def clean_adapter_config(config_path):
    """Clean adapter config for compatibility"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Remove problematic fields
        problematic_fields = [
            'corda_config', 'eva_config', 'loftq_config', 'megatron_config',
            'megatron_core', 'qalora_group_size', 'use_dora', 'use_qalora', 'use_rslora'
        ]
        
        cleaned_config = {}
        for key, value in config.items():
            if key not in problematic_fields:
                cleaned_config[key] = value
        
        # Save cleaned config
        cleaned_config_path = config_path.replace('.json', '_cleaned.json')
        with open(cleaned_config_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_config, f, indent=2)
        
        return cleaned_config_path
        
    except Exception:
        return config_path

def translate_text(text, model_path="fine-tuned-mbart-tl2en-baseline-best"):
    """Translate Filipino text to English using baseline model"""
    try:
        # Load base model
        base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Clean adapter config if needed
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            cleaned_config_path = clean_adapter_config(adapter_config_path)
            model_path = os.path.dirname(cleaned_config_path)
        
        # Load LoRA adapters
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
        except Exception:
            # Fallback to base model
            model = base_model
        
        # Load tokenizer
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer.src_lang = "tl_XX"
        tokenizer.tgt_lang = "en_XX"
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set device and prepare model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Encode input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            bos_token_id = tokenizer.lang_code_to_id.get("en_XX", tokenizer.eos_token_id)
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=bos_token_id,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                repetition_penalty=1.2
            )
        
        # Decode and return
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
        
    except Exception as e:
        return f"Translation error: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_translate_baseline_clean.py 'Filipino text here'")
        return
    
    filipino_text = " ".join(sys.argv[1:])
    english_translation = translate_text(filipino_text)
    
    # Clean output - just input and translation
    print(f"🇵🇭 {filipino_text}")
    print(f"🇺🇸 {english_translation}")

if __name__ == "__main__":
    main()

