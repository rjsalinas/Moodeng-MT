#!/usr/bin/env python3
"""
Script to fix the baseline model by saving missing files without retraining
This addresses the issue where the baseline model was trained but not fully saved
"""

import os
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel

def fix_baseline_model():
    """Fix the baseline model by saving missing files."""
    print("🔧 Fixing baseline model...")
    
    # Check if baseline model exists
    baseline_path = "fine-tuned-mbart-tl2en-baseline-best"
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline model directory '{baseline_path}' not found!")
        return False
    
    try:
        # Load the base model
        print("🔧 Loading base mBART model...")
        base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Load the trained LoRA adapters using cleaned config
        print("🔧 Loading LoRA adapters...")
        
        # Use the cleaned config if available
        cleaned_config_path = os.path.join(baseline_path, "adapter_config_cleaned.json")
        if os.path.exists(cleaned_config_path):
            print("🔧 Using cleaned adapter config...")
            model = PeftModel.from_pretrained(base_model, baseline_path, config_file=cleaned_config_path)
        else:
            print("🔧 Using original adapter config...")
            model = PeftModel.from_pretrained(base_model, baseline_path)
        
        # Load tokenizer
        print("🔧 Loading tokenizer...")
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Set language codes
        tokenizer.src_lang = "tl_XX"  # Filipino
        tokenizer.tgt_lang = "en_XX"  # English
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Save the complete model
        print("🔧 Saving complete model...")
        model.save_pretrained(baseline_path)
        
        # Save tokenizer
        print("🔧 Saving tokenizer...")
        tokenizer.save_pretrained(baseline_path)
        
        # Save base model config
        print("🔧 Saving base model config...")
        base_model.config.save_pretrained(baseline_path)
        
        print("✅ Baseline model fixed successfully!")
        print(f"📁 Model directory: {baseline_path}")
        
        # Verify files
        print("\n📋 Verifying saved files...")
        files = os.listdir(baseline_path)
        for file in files:
            print(f"   ✅ {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing baseline model: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Baseline Model Fix Script")
    print("=" * 40)
    
    if fix_baseline_model():
        print("\n🎉 Baseline model is now ready for translation!")
        print("💡 You can now use: python simple_translate_baseline.py 'your text here'")
    else:
        print("\n❌ Failed to fix baseline model")
        print("💡 You may need to retrain: python model_training_baseline.py")

if __name__ == "__main__":
    main()
