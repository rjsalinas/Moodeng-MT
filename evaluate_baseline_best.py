#!/usr/bin/env python3
"""
Evaluate the Filipino→English baseline-best model on test set using sacreBLEU, chrF, METEOR, and COMET.

- Sources: corpus-parallel-txt/test.tl (Filipino)
- References: corpus-parallel-txt/test.en (English)
"""

import os
import sys
import argparse
import json
from typing import List, Tuple

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


def clean_adapter_config(config_path: str) -> str:
    """Clean LoRA adapter config for compatibility; return path to cleaned config or original."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        problematic_fields = [
            'corda_config', 'eva_config', 'loftq_config', 'megatron_config',
            'megatron_core', 'qalora_group_size', 'use_dora', 'use_qalora', 'use_rslora'
        ]

        cleaned_config = {k: v for k, v in config.items() if k not in problematic_fields}

        cleaned_config_path = config_path.replace('.json', '_cleaned.json')
        with open(cleaned_config_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_config, f, indent=2)

        return cleaned_config_path
    except Exception:
        return config_path


def load_model(model_path: str):
    base_model_name = "facebook/mbart-large-50-many-to-many-mmt"
    base_model = MBartForConditionalGeneration.from_pretrained(base_model_name)

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        cleaned_config_path = clean_adapter_config(adapter_config_path)
        model_path = os.path.dirname(cleaned_config_path)

    try:
        model = PeftModel.from_pretrained(base_model, model_path)
    except Exception:
        model = base_model

    tokenizer = MBart50Tokenizer.from_pretrained(base_model_name)
    tokenizer.src_lang = "tl_XX"
    tokenizer.tgt_lang = "en_XX"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def read_parallel_files(src_path: str, ref_path: str) -> Tuple[List[str], List[str]]:
    with open(src_path, 'r', encoding='utf-8') as f:
        src = [line.rstrip("\n") for line in f]
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref = [line.rstrip("\n") for line in f]
    if len(src) != len(ref):
        raise ValueError(f"Mismatched lines: {len(src)} src vs {len(ref)} ref")
    return src, ref


@torch.inference_mode()
def translate_batch(texts: List[str], model, tokenizer, device: torch.device, max_length: int) -> List[str]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    bos_token_id = tokenizer.lang_code_to_id.get("en_XX", tokenizer.eos_token_id)
    gen = model.generate(
        **enc,
        forced_bos_token_id=bos_token_id,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        do_sample=False,
        no_repeat_ngram_size=3,
        length_penalty=0.8,
        repetition_penalty=1.2,
    )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def compute_metrics_sys_refs(hypotheses: List[str], references: List[str]):
    results = {}

    # sacreBLEU
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        results["sacrebleu"] = {
            "score": bleu.score,
            "precisions": list(bleu.precisions),
            "bp": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
        }
    except Exception as e:
        results["sacrebleu"] = {"error": str(e)}

    # chrF
    try:
        import sacrebleu
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        results["chrf"] = {"score": chrf.score}
    except Exception as e:
        results["chrf"] = {"error": str(e)}

    # METEOR
    try:
        # NLTK METEOR needs punkt resources sometimes; handle gracefully
        from nltk.translate.meteor_score import single_meteor_score
        # Average sentence-level METEOR to approximate corpus METEOR
        meteor_scores = []
        for hyp, ref in zip(hypotheses, references):
            try:
                meteor_scores.append(single_meteor_score(ref, hyp))
            except Exception:
                meteor_scores.append(0.0)
        results["meteor"] = {"score": float(sum(meteor_scores) / max(1, len(meteor_scores)))}
    except Exception as e:
        results["meteor"] = {"error": str(e)}

    # COMET (reference-based)
    try:
        # Try new UniCOMET or fallback to older COMET
        try:
            from comet import download_model, load_from_checkpoint  # type: ignore
            model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(model_path)
            data = [{"src": "", "mt": hyp, "ref": ref} for hyp, ref in zip(hypotheses, references)]
            comet_output = comet_model.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
            results["comet"] = {"score": float(comet_output.system_score)}
        except Exception:
            # Newer library name
            from comet import models as comet_models  # type: ignore
            comet_model = comet_models.load("Unbabel/wmt22-comet-da")
            data = [{"src": "", "mt": hyp, "ref": ref} for hyp, ref in zip(hypotheses, references)]
            scores = comet_model.predict(data, batch_size=32)
            results["comet"] = {"score": float(scores["system"]) if isinstance(scores, dict) and "system" in scores else float(scores)}
    except Exception as e:
        results["comet"] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline-best model with multiple metrics")
    parser.add_argument("--model_dir", default="fine-tuned-mbart-tl2en-baseline-best", help="Path to fine-tuned model directory")
    parser.add_argument("--src", default=os.path.join("corpus-parallel-txt", "test.tl"), help="Path to source (Filipino) test file")
    parser.add_argument("--ref", default=os.path.join("corpus-parallel-txt", "test.en"), help="Path to reference (English) test file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--out", default=None, help="Optional path to write hypotheses")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.model_dir}")
    model, tokenizer = load_model(args.model_dir)
    model = model.to(device)
    model.eval()

    print(f"Reading test set: src={args.src} ref={args.ref}")
    src_texts, ref_texts = read_parallel_files(args.src, args.ref)

    print(f"Translating {len(src_texts)} sentences with batch_size={args.batch_size}...")
    hypotheses: List[str] = []
    for i in range(0, len(src_texts), args.batch_size):
        batch = src_texts[i:i + args.batch_size]
        hyps = translate_batch(batch, model, tokenizer, device, args.max_length)
        hypotheses.extend(hyps)
        if (i // args.batch_size) % 10 == 0:
            print(f"  progress: {min(i + args.batch_size, len(src_texts))}/{len(src_texts)}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for h in hypotheses:
                f.write(h + "\n")
        print(f"Wrote hypotheses to: {args.out}")

    print("Computing metrics...")
    metrics = compute_metrics_sys_refs(hypotheses, ref_texts)

    # Pretty print summary
    def maybe(metric: str):
        m = metrics.get(metric, {})
        if "score" in m:
            return f"{m['score']:.2f}"
        if "error" in m:
            return f"error: {m['error']}"
        return "n/a"

    print("\n=== Evaluation Summary ===")
    print(f"sacreBLEU: {maybe('sacrebleu')}")
    print(f"chrF:      {maybe('chrf')}")
    print(f"METEOR:    {maybe('meteor')}")
    print(f"COMET:     {maybe('comet')}")

    # Also dump full JSON to stdout for programmatic use
    try:
        print("\nRaw metrics JSON:\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    except Exception:
        pass


if __name__ == "__main__":
    main()



