#!/usr/bin/env python3
"""
Run inference on 15 generated Filipino sentences using:
- translate_baseline.translate_text (baseline best)
- translate.translate_text (mbart50-finetuned-LoRA-best)
- translate_v2.translate_text (mbart50-finetuned-LoRA-best_v2)

Generates diverse Filipino sentences for testing.
Outputs results to stdout and saves to inference_generated_15.csv.
Optimized with model caching for faster inference.
"""

import pandas as pd
import time
import random

# Import translator modules (not functions to avoid loading models multiple times)
import translate_baseline
import translate
import translate_v2


def generate_filipino_sentences():
    """Generate 15 diverse Filipino sentences for testing."""
    
    # Different types of Filipino sentences
    sentences = [
        # Casual conversations
        "Kumusta ka na? Ang tagal na nating hindi nagkita.",
        "Saan ka pupunta mamaya? Gusto ko sumama.",
        "Ang init ng panahon ngayon, di ba?",
        "Nakita mo na ba yung bagong movie? Ang ganda daw.",
        
        # Questions with "di ba"
        "Ang ganda ng sunset kanina, di ba?",
        "Masarap yung pagkain sa restaurant na yun, di ba?",
        "Ang bilis ng oras, di ba? Parang kahapon lang tayo nagkita.",
        "Ang mahal na ng mga bilihin ngayon, di ba?",
        
        # Mixed language (Taglish)
        "Nakita ko siya sa mall yesterday, ang cute niya.",
        "Can you help me with this project? Medyo mahirap eh.",
        "I love this song so much, nakakarelax talaga.",
        
        # Emotional expressions
        "Nakakainis talaga yung traffic sa EDSA.",
        "Ang saya ng party kahapon, sobrang enjoy ako.",
        "Nakakalungkot naman yung nangyari sa kanila.",
        
        # Complex sentences
        "Kung pwede lang sana, gusto ko magbakasyon sa beach kasama ang pamilya ko.",
        "Hindi ko alam kung bakit ganito ang nangyayari, pero susubukan ko pa rin.",
    ]
    
    # If we have more than 15, randomly select 15
    if len(sentences) > 15:
        sentences = random.sample(sentences, 15)
    
    return sentences


def load_models():
    """Load all models once at startup for faster inference."""
    print("🔄 Loading models...")
    start_time = time.time()
    
    # Load baseline model
    print("  Loading baseline model...")
    baseline_start = time.time()
    try:
        baseline_translate = translate_baseline.translate_text
        print(f"  ✓ Baseline loaded in {time.time() - baseline_start:.2f}s")
    except Exception as e:
        print(f"  ✗ Baseline failed: {e}")
        baseline_translate = None
    
    # Load v1 model
    print("  Loading v1 model...")
    v1_start = time.time()
    try:
        translate_v1 = translate.translate_text
        print(f"  ✓ V1 loaded in {time.time() - v1_start:.2f}s")
    except Exception as e:
        print(f"  ✗ V1 failed: {e}")
        translate_v1 = None
    
    # Load v2 model
    print("  Loading v2 model...")
    v2_start = time.time()
    try:
        translate_v2_func = translate_v2.translate
        print(f"  ✓ V2 loaded in {time.time() - v2_start:.2f}s")
    except Exception as e:
        print(f"  ✗ V2 failed: {e}")
        translate_v2_func = None
    
    total_time = time.time() - start_time
    print(f"🎯 All models loaded in {total_time:.2f}s")
    
    return baseline_translate, translate_v1, translate_v2_func


def main():
    # Generate sentences
    print("📝 Generating 15 Filipino sentences...")
    sentences = generate_filipino_sentences()
    
    # Load all models once
    baseline_translate, translate_v1, translate_v2 = load_models()
    
    print(f"\n🚀 Starting inference on {len(sentences)} generated sentences...")
    inference_start = time.time()
    
    outputs = []
    for i, src in enumerate(sentences):
        print(f"Processing sentence {i+1}/15...")
        sentence_start = time.time()
        
        # Translate with each model
        results = {"idx": i + 1, "src": src}
        
        if baseline_translate:
            try:
                results["baseline"] = baseline_translate(src)
            except Exception as e:
                results["baseline"] = f"Error: {str(e)}"
        else:
            results["baseline"] = "Model not loaded"
        
        if translate_v1:
            try:
                results["v1"] = translate_v1(src)
            except Exception as e:
                results["v1"] = f"Error: {str(e)}"
        else:
            results["v1"] = "Model not loaded"
        
        if translate_v2:
            try:
                results["v2"] = translate_v2(src)
            except Exception as e:
                results["v2"] = f"Error: {str(e)}"
        else:
            results["v2"] = "Model not loaded"
        
        outputs.append(results)
        sentence_time = time.time() - sentence_start
        print(f"  ✓ Completed in {sentence_time:.2f}s")

    total_inference_time = time.time() - inference_start
    print(f"\n🎯 All inference completed in {total_inference_time:.2f}s")

    out_df = pd.DataFrame(outputs)
    
    # Print a concise CLI view
    for _, row in out_df.iterrows():
        print(f"\n--- Sample {int(row['idx'])}")
        print("SRC:", row["src"]) 
        print("BASE:", row["baseline"]) 
        print("V1  :", row["v1"]) 
        print("V2  :", row["v2"])
    
    # Save to CSV
    out_df.to_csv("inference_generated_15.csv", index=False)
    print(f"\n💾 Results saved to inference_generated_15.csv")


if __name__ == "__main__":
    main()
