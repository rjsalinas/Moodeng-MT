#!/usr/bin/env python3
"""
Simple script to translate a single Filipino text to English
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel
import sys

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
        # Load the base mBART model
        base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Load the fine-tuned model (LoRA adapters)
        # Handle tied embeddings properly for mBART
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
        except Exception as e:
            print(f"Warning: Standard loading failed: {e}")
            # Try alternative loading method
            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False  # Disable training mode for inference
            )
        
        # Load the tokenizer
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
        model = model.to(device)
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Handle tied embeddings warning by setting proper config
        if hasattr(model, 'config'):
            model.config.tie_word_embeddings = False
        
        # Encode the input text
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
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                    max_length=128,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False
                )
            except Exception as e:
                print(f"Generation error: {e}")
                # Fallback to simpler generation
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True
                )
        
        # Decode the output
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_translate.py 'Filipino text here'")
        print("Example: python simple_translate.py 'Kamusta ka?'")
        return
    
    # Get the text from command line arguments
    filipino_text = " ".join(sys.argv[1:])
    
    print(f"🇵🇭 Filipino: {filipino_text}")
    print("🔄 Translating...")
    
    # Translate
    english_translation = translate_filipino_to_english(filipino_text)
    
    if english_translation:
        print(f"🇺🇸 English: {english_translation}")
    else:
        print("❌ Translation failed")

if __name__ == "__main__":
    main()
