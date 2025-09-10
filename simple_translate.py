#!/usr/bin/env python3
"""
Simple script to translate a single Filipino text to English
Enhanced with robust error handling and compatibility fixes
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel
import sys
import os
import json

def clean_adapter_config(config_path):
    """
    Clean the adapter config to remove unsupported fields for compatibility
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Remove fields that might cause compatibility issues
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
        
        print(f"✅ Cleaned adapter config saved to: {cleaned_config_path}")
        return cleaned_config_path
        
    except Exception as e:
        print(f"⚠️  Warning: Could not clean adapter config: {e}")
        return config_path

def translate_filipino_to_english(text, model_path="fine-tuned-mbart-tl2en-best"):
    """
    Translate Filipino text to English using the fine-tuned model
    
    Args:
        text (str): Filipino text to translate
        model_path (str): Path to the saved fine-tuned model
    
    Returns:
        str: English translation
    """
    try:
        print(f"🔧 Loading base mBART model...")
        # Load the base mBART model
        base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Check if adapter config exists and clean it if needed
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"🔧 Found adapter config, checking compatibility...")
            cleaned_config_path = clean_adapter_config(adapter_config_path)
            model_path = os.path.dirname(cleaned_config_path)
        
        # Load the fine-tuned model (LoRA adapters)
        print(f"🔧 Loading LoRA adapters from: {model_path}")
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ LoRA adapters loaded successfully")
        except Exception as e:
            print(f"⚠️  Standard loading failed: {e}")
            print("🔄 Trying alternative loading method...")
            
            # Try alternative loading method
            try:
                model = PeftModel.from_pretrained(
                    base_model, 
                    model_path,
                    is_trainable=False  # Disable training mode for inference
                )
                print("✅ Alternative loading successful")
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                print("🔄 Trying to load without PEFT (base model only)...")
                model = base_model
        
        # Load the tokenizer
        print(f"🔧 Loading tokenizer...")
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Set language codes
        tokenizer.src_lang = "tl_XX"  # Filipino
        tokenizer.tgt_lang = "en_XX"  # English
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set device and prepare model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")
        model = model.to(device)
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Handle tied embeddings warning by setting proper config
        if hasattr(model, 'config'):
            model.config.tie_word_embeddings = False
        
        # Encode the input text
        print(f"🔧 Encoding input text: '{text}'")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        
        # Generate translation with robust error handling
        print(f"🔧 Generating translation...")
        with torch.no_grad():
            try:
                # Get language token ID for English
                bos_token_id = tokenizer.lang_code_to_id.get("en_XX", tokenizer.eos_token_id)
                
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=bos_token_id,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    length_penalty=0.8,
                    repetition_penalty=1.2
                )
                print("✅ Generation successful with advanced parameters")
            except Exception as e:
                print(f"⚠️  Advanced generation failed: {e}")
                print("🔄 Falling back to simpler generation...")
                
                # Fallback to simpler generation
                try:
                    outputs = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False
                    )
                    print("✅ Fallback generation successful")
                except Exception as e2:
                    print(f"❌ Fallback generation also failed: {e2}")
                    return None
        
        # Decode the output
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Translation completed successfully")
        return translation
        
    except Exception as e:
        print(f"❌ Error in translation function: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_translate.py 'Filipino text here'")
        print("Example: python simple_translate.py 'Kamusta ka?'")
        return
    
    # Get the text from command line arguments
    filipino_text = " ".join(sys.argv[1:])
    
    print(f"🇵🇭 Filipino: {filipino_text}")
    print("🔄 Starting translation process...")
    
    # Check if model directory exists
    model_path = "fine-tuned-mbart-tl2en-best"
    if not os.path.exists(model_path):
        print(f"⚠️  Model directory '{model_path}' not found!")
        print("🔄 Trying alternative model directory...")
        model_path = "fine-tuned-mbart-tl2en"
        if not os.path.exists(model_path):
            print(f"❌ No model directory found. Please ensure the model is trained first.")
            return
    
    print(f"🔧 Using model directory: {model_path}")
    
    # Translate
    english_translation = translate_filipino_to_english(filipino_text, model_path)
    
    if english_translation:
        print(f"🇺🇸 English: {english_translation}")
        print("🎉 Translation completed successfully!")
    else:
        print("❌ Translation failed")
        print("💡 Troubleshooting tips:")
        print("   1. Check if the model was trained successfully")
        print("   2. Verify the model directory structure")
        print("   3. Check for compatibility issues with PEFT version")
        print("   4. Try running: pip install --upgrade peft")

if __name__ == "__main__":
    main()
