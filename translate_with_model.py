#!/usr/bin/env python3
"""
Script to use the saved fine-tuned Filipino-to-English translation model
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel
import os

def load_fine_tuned_model(model_path="fine-tuned-mbart-tl2en-best"):
    """
    Load the fine-tuned model from the saved directory
    """
    print(f"Loading model from: {model_path}")
    
    # Check if the model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return None, None
    
    try:
        # Load the base mBART model
        base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Load the fine-tuned model (LoRA adapters)
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Load the tokenizer
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Set language codes
        tokenizer.src_lang = "tl_XX"  # Filipino
        tokenizer.tgt_lang = "en_XX"  # English
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def translate_text(text, model, tokenizer, max_length=128):
    """
    Translate Filipino text to English using the fine-tuned model
    """
    if not text or not text.strip():
        return ""
    
    try:
        # Encode the input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Ensure tensors are on the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode the output
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
        
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return ""

def main():
    """
    Main function to demonstrate translation
    """
    print("Filipino-to-English Translation using Fine-tuned Model")
    print("=" * 60)
    
    # Load the model
    model, tokenizer = load_fine_tuned_model()
    
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"📱 Using device: {device}")
    
    # Example Filipino texts to translate
    filipino_texts = [
        "Kamusta ka?",
        "Salamat sa tulong mo.",
        "Magandang umaga.",
        "Gusto ko ng kape.",
        "Paalam na.",
        "Mahal kita.",
        "Nasaan ka?",
        "Kumain ka na ba?",
        "Anong oras na?",
        "Saan ka pupunta?"
    ]
    
    print("\n🧪 Testing translations:")
    print("-" * 60)
    
    for i, filipino_text in enumerate(filipino_texts, 1):
        print(f"\n{i}. 🇵🇭 Filipino: {filipino_text}")
        
        # Translate
        english_translation = translate_text(filipino_text, model, tokenizer)
        
        if english_translation:
            print(f"   🇺🇸 English: {english_translation}")
        else:
            print("   ❌ Translation failed")
    
    # Interactive translation
    print("\n" + "=" * 60)
    print("💬 Interactive Translation Mode")
    print("Type 'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🇵🇭 Enter Filipino text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Translate user input
            translation = translate_text(user_input, model, tokenizer)
            
            if translation:
                print(f"🇺🇸 Translation: {translation}")
            else:
                print("❌ Translation failed")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
