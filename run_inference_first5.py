#!/usr/bin/env python3
"""
Run inference on 15 specific Filipino sentences using:
- translate_baseline.translate_text (baseline best)
- translate.translate_text (mbart50-finetuned-LoRA-best)
- translate_v2.translate_text (mbart50-finetuned-LoRA-best_v2)
- translate_v3.translate (mbart50-finetuned-LoRA-best_v3)

Outputs results to stdout and saves to inference_15.csv.
Optimized with model caching for faster inference.
"""

import pandas as pd
import time

# Import translator modules (not functions to avoid loading models multiple times)
import translate_baseline
import translate
import translate_v2


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
    # Load all models once
    baseline_translate, translate_v1, translate_v2 = load_models()
    
    # Define the 15 specific sentences
    sentences = [
        "Sana lahat ganito kabilis mag announce di ba.",
        "Biggest slap to that industry plant lol di ba kinaya buhatin?",
        "Good evening! Maganda sales namin these days and di po makahanap ng ligaw na victim so dito na lng po :) Comment below kung bakit deserve mo ng D-1 ticket (can choose 27 or 28). Ends 8:30am tomorrow.",
        "di ba pwede sabihin kay sir na nilindol yung lote kaya di ako makagawa ng floor plan.",
        "bat stage 3 agad may di ba aq napanood.. HAHAHAHA",
        "into the i-land nanaman? Haha di ba kayo nagsasawa? 'cause same i miss hanbin yawa.",
        "Puro coffee date yaya ng mga to amputa siguro masyadong oa pagka manifest ko ng starbucks planner ah. di ba pwedeng gcash date na lang?",
        "OohH Jivaaa!! supon ubo lang,Antibiotic agad??? di ba pwede mga OTC drugs mo na kaloka.",
        "pati b nmn ikw ireject aq s bgay cno b nmn aq 'di ba isa lang nmn aqng dinosaur rorororor.",
        "saw this one sa tl tas what if hindi pala si jungwoo yung bowl cut...si sungchan pala? t7s unit with sungtaro nga.",
        "Di daw nakapremium si jinki ahahaha charot di ko rin alam.",
        "like lahat kami bagets tas yung mama q lang yung parent na sumama pak. pati yung di invited sa gala namin, sumasama.",
        "hindi ko na rin alam navivisualize ko na kung sino sino basta ang sure tayo andyan si jungwoo di naman maitatanggi.",
        "Parang may lamay sa grammy eh noh antamlay magsalita di ba kayo naka kaen.",
        "di ba pedeng ikaw naman yung magpagaan ng loob ko? hindi yung magiging isa ka rin sa reason kung bat mabigat yung loob ko?"
    ]

    print(f"\n🚀 Starting inference on {len(sentences)} sentences...")
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
    out_df.to_csv("inference_15.csv", index=False)
    print(f"\n💾 Results saved to inference_15.csv")


if __name__ == "__main__":
    main()


