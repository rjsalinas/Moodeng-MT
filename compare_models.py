#!/usr/bin/env python3
"""
Model Comparison Script

This script compares the performance between:
1. Enhanced model (trained on full_enhanced_parallel_corpus.csv with CalamanCy)
2. Baseline model (trained on filipino_english_parallel_corpus.csv without enhancements)

Usage:
    python compare_models.py "Filipino text to translate"
    python compare_models.py --test_samples 10  # Test with 10 random samples
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel
import sys
import os
import json
import pandas as pd
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

def clean_adapter_config(config_path):
    """Clean the adapter config to remove unsupported fields for compatibility"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        problematic_fields = [
            'corda_config', 'eva_config', 'loftq_config', 'megatron_config',
            'megatron_core', 'qalora_group_size', 'use_dora', 'use_qalora', 'use_rslora'
        ]
        
        cleaned_config = {}
        for key, value in config.items():
            if key not in problematic_fields:
                cleaned_config[key] = value
        
        cleaned_config_path = config_path.replace('.json', '_cleaned.json')
        with open(cleaned_config_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_config, f, indent=2)
        
        return cleaned_config_path
        
    except Exception as e:
        return config_path

def load_model(model_path, model_name):
    """Load a fine-tuned model with error handling"""
    try:
        print(f"🔧 Loading {model_name} model...")
        
        # Load base model
        base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Check and clean adapter config if needed
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            cleaned_config_path = clean_adapter_config(adapter_config_path)
            model_path = os.path.dirname(cleaned_config_path)
        
        # Load LoRA adapters
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"✅ {model_name} model loaded successfully")
        except Exception as e:
            print(f"⚠️  Standard loading failed for {model_name}: {e}")
            try:
                model = PeftModel.from_pretrained(
                    base_model, 
                    model_path,
                    is_trainable=False
                )
                print(f"✅ {model_name} model loaded with alternative method")
            except Exception as e2:
                print(f"❌ Failed to load {model_name} model: {e2}")
                return None, None
        
        # Load tokenizer
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer.src_lang = "tl_XX"
        tokenizer.tgt_lang = "en_XX"
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        if hasattr(model, 'config'):
            model.config.tie_word_embeddings = False
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")
        return None, None

def translate_with_model(text, model, tokenizer, model_name):
    """Translate text using a specific model"""
    if model is None or tokenizer is None:
        return None, 0.0
    
    try:
        start_time = time.time()
        
        # Encode input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move to device
        device = next(model.parameters()).device
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
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        return translation, inference_time
        
    except Exception as e:
        print(f"⚠️  Translation error with {model_name}: {e}")
        return None, 0.0

def calculate_bleu_score(hypothesis, reference):
    """Calculate BLEU score between hypothesis and reference"""
    try:
        if not hypothesis or not reference:
            return 0.0
        
        hypothesis_tokens = hypothesis.split()
        reference_tokens = reference.split()
        
        if not hypothesis_tokens or not reference_tokens:
            return 0.0
        
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0

def compare_single_text(text):
    """Compare translation quality for a single text"""
    print(f"\n🇵🇭 Input Text: {text}")
    print("=" * 60)
    
    # Load models
    enhanced_model, enhanced_tokenizer = load_model("fine-tuned-mbart-tl2en-best", "Enhanced")
    baseline_model, baseline_tokenizer = load_model("fine-tuned-mbart-tl2en-baseline-best", "Baseline")
    
    if enhanced_model is None and baseline_model is None:
        print("❌ No models available for comparison")
        return
    
    results = {}
    
    # Enhanced model translation
    if enhanced_model is not None:
        enhanced_translation, enhanced_time = translate_with_model(text, enhanced_model, enhanced_tokenizer, "Enhanced")
        if enhanced_translation:
            results['Enhanced'] = {
                'translation': enhanced_translation,
                'time': enhanced_time
            }
            print(f"🔹 Enhanced Model:")
            print(f"   Translation: {enhanced_translation}")
            print(f"   Time: {enhanced_time:.3f}s")
    
    # Baseline model translation
    if baseline_model is not None:
        baseline_translation, baseline_time = translate_with_model(text, baseline_model, baseline_tokenizer, "Baseline")
        if baseline_translation:
            results['Baseline'] = {
                'translation': baseline_translation,
                'time': baseline_time
            }
            print(f"🔹 Baseline Model:")
            print(f"   Translation: {baseline_translation}")
            print(f"   Time: {baseline_time:.3f}s")
    
    # Comparison summary
    if len(results) > 1:
        print(f"\n📊 Comparison Summary:")
        print(f"   Models compared: {len(results)}")
        
        if 'Enhanced' in results and 'Baseline' in results:
            enhanced_time = results['Enhanced']['time']
            baseline_time = results['Baseline']['time']
            
            if enhanced_time > 0 and baseline_time > 0:
                speed_diff = ((baseline_time - enhanced_time) / baseline_time) * 100
                print(f"   Speed difference: {speed_diff:+.1f}% ({'Enhanced' if speed_diff > 0 else 'Baseline'} is faster)")
    
    return results

def compare_with_test_samples(num_samples=10):
    """Compare models using random test samples from the dataset"""
    print(f"\n🧪 Comparing models with {num_samples} random test samples")
    print("=" * 60)
    
    # Load test data
    try:
        df = pd.read_csv("filipino_english_parallel_corpus.csv")
        df = df.dropna(subset=['preprocessed_text', 'english_translation'])
        df = df[(df['preprocessed_text'].str.strip() != '') & (df['english_translation'].str.strip() != '')]
        
        # Remove social media artifacts
        df = df[~df['preprocessed_text'].str.contains('@|#', na=False)]
        df = df[~df['english_translation'].str.contains('@|#', na=False)]
        
        # Filter by length
        df['src_length'] = df['preprocessed_text'].str.len()
        df['tgt_length'] = df['english_translation'].str.len()
        df = df[
            (df['src_length'] >= 10) & (df['src_length'] <= 200) &
            (df['tgt_length'] >= 10) & (df['tgt_length'] <= 200)
        ]
        
        if len(df) == 0:
            print("❌ No suitable test samples found")
            return
        
        # Sample random test cases
        test_samples = df.sample(min(num_samples, len(df)))
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Load models
    enhanced_model, enhanced_tokenizer = load_model("fine-tuned-mbart-tl2en-best", "Enhanced")
    baseline_model, baseline_tokenizer = load_model("fine-tuned-mbart-tl2en-baseline-best", "Baseline")
    
    if enhanced_model is None and baseline_model is None:
        print("❌ No models available for comparison")
        return
    
    # Test results
    enhanced_bleus = []
    baseline_bleus = []
    enhanced_times = []
    baseline_times = []
    
    print(f"\n📋 Test Results:")
    print("-" * 80)
    
    for i, (_, row) in enumerate(test_samples.iterrows(), 1):
        src_text = row['preprocessed_text']
        reference = row['english_translation']
        
        print(f"\n{i}. Source: {src_text}")
        print(f"   Reference: {reference}")
        
        # Enhanced model
        if enhanced_model is not None:
            enhanced_translation, enhanced_time = translate_with_model(
                src_text, enhanced_model, enhanced_tokenizer, "Enhanced"
            )
            if enhanced_translation:
                enhanced_bleu = calculate_bleu_score(enhanced_translation, reference)
                enhanced_bleus.append(enhanced_bleu)
                enhanced_times.append(enhanced_time)
                print(f"   Enhanced: {enhanced_translation} (BLEU: {enhanced_bleu:.3f}, Time: {enhanced_time:.3f}s)")
        
        # Baseline model
        if baseline_model is not None:
            baseline_translation, baseline_time = translate_with_model(
                src_text, baseline_model, baseline_tokenizer, "Baseline"
            )
            if baseline_translation:
                baseline_bleu = calculate_bleu_score(baseline_translation, reference)
                baseline_bleus.append(baseline_bleu)
                baseline_times.append(baseline_time)
                print(f"   Baseline: {baseline_translation} (BLEU: {baseline_bleu:.3f}, Time: {baseline_time:.3f}s)")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print("=" * 60)
    
    if enhanced_bleus:
        print(f"🔹 Enhanced Model:")
        print(f"   Average BLEU: {sum(enhanced_bleus)/len(enhanced_bleus):.3f}")
        print(f"   Average Time: {sum(enhanced_times)/len(enhanced_times):.3f}s")
        print(f"   Samples: {len(enhanced_bleus)}")
    
    if baseline_bleus:
        print(f"🔹 Baseline Model:")
        print(f"   Average BLEU: {sum(baseline_bleus)/len(baseline_bleus):.3f}")
        print(f"   Average Time: {sum(baseline_times)/len(baseline_times):.3f}s")
        print(f"   Samples: {len(baseline_bleus)}")
    
    if enhanced_bleus and baseline_bleus:
        bleu_diff = (sum(enhanced_bleus)/len(enhanced_bleus)) - (sum(baseline_bleus)/len(baseline_bleus))
        time_diff = (sum(enhanced_times)/len(enhanced_times)) - (sum(baseline_times)/len(baseline_times))
        
        print(f"\n🔍 Model Comparison:")
        print(f"   BLEU Difference: {bleu_diff:+.3f} ({'Enhanced' if bleu_diff > 0 else 'Baseline'} is better)")
        print(f"   Time Difference: {time_diff:+.3f}s ({'Enhanced' if time_diff < 0 else 'Baseline'} is faster)")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python compare_models.py 'Filipino text to translate'")
        print("  python compare_models.py --test_samples 10")
        print("\nExamples:")
        print("  python compare_models.py 'Kamusta ka?'")
        print("  python compare_models.py --test_samples 5")
        return
    
    if sys.argv[1] == "--test_samples":
        if len(sys.argv) >= 3:
            try:
                num_samples = int(sys.argv[2])
                compare_with_test_samples(num_samples)
            except ValueError:
                print("❌ Invalid number of samples. Please provide a valid integer.")
        else:
            print("❌ Please specify the number of test samples.")
    else:
        # Single text comparison
        text = " ".join(sys.argv[1:])
        compare_single_text(text)

if __name__ == "__main__":
    main()
