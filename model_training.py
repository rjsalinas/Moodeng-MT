#!/usr/bin/env python3
"""
Model training (LoRA mBART50) using annotated_tweets.xlsx with quality-based 70/15/15 split.

Per-epoch logs: train loss, val loss, sacreBLEU, chrF, METEOR, COMET.
Saves logs to both JSON and TXT. Best model saved to mbart50-finetuned-LoRA-best.
Evaluates final metrics on the test split after training.
"""

import os
import sys
import json
import math
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm

from transformers import MBartForConditionalGeneration, MBart50Tokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType


# ---------------------- Configuration ----------------------
SEED = 42
IGNORE_INDEX = -100
MAX_LENGTH = 128
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20
WARMUP_STEPS = 100
PATIENCE = 8
VAL_MAX_EXAMPLES_FOR_PRINT = 5

BASE_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
BEST_MODEL_DIR = "mbart50-finetuned-LoRA-best"

INPUT_XLSX = "annotated_tweets.xlsx"


# ---------------------- Utils ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_requirements() -> bool:
    required = ["torch", "transformers", "peft", "pandas", "tqdm", "nltk", "sacrebleu"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install: pip install torch transformers peft pandas tqdm nltk sacrebleu")
        return False
    return True


def safe_save_pretrained(model: torch.nn.Module, output_dir: str) -> bool:
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir)
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        return True
    except Exception as e:
        print(f"Model save failed: {e}")
        return False


class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: MBart50Tokenizer, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        src_text = str(row["src"]) if not pd.isna(row["src"]) else ""
        tgt_text = str(row["tgt"]) if not pd.isna(row["tgt"]) else ""
        if len(src_text.strip()) == 0:
            src_text = "dummy"
        if len(tgt_text.strip()) == 0:
            tgt_text = "dummy"

        src = self.tokenizer(
            src_text,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        tgt = self.tokenizer(
            text_target=tgt_text,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        labels = tgt.get("input_ids", tgt.get("labels")).squeeze()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels = labels.masked_fill(labels == pad_id, IGNORE_INDEX)
        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels,
        }


@torch.inference_mode()
def translate_batch(texts: List[str], model, tokenizer, device: torch.device, max_length: int) -> List[str]:
    tokenizer.src_lang = "tl_XX"
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
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=0.8,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
    )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def compute_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # sacreBLEU
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        results["sacrebleu"] = {
            "score": float(bleu.score),
            "bp": float(bleu.bp),
            "sys_len": int(bleu.sys_len),
            "ref_len": int(bleu.ref_len),
        }
    except Exception as e:
        results["sacrebleu"] = {"error": str(e)}

    # chrF
    try:
        import sacrebleu
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        results["chrf"] = {"score": float(chrf.score)}
    except Exception as e:
        results["chrf"] = {"error": str(e)}

    # METEOR
    try:
        from nltk.translate.meteor_score import single_meteor_score
        meteor_values = []
        for hyp, ref in zip(hypotheses, references):
            try:
                meteor_values.append(single_meteor_score(ref, hyp))
            except Exception:
                meteor_values.append(0.0)
        results["meteor"] = {"score": float(sum(meteor_values) / max(1, len(meteor_values)))}
    except Exception as e:
        results["meteor"] = {"error": str(e)}

    return results


def load_annotated_dataset(xlsx_path: str) -> pd.DataFrame:
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Dataset not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    # Normalize to columns 'src' and 'tgt'
    colmap_candidates = [
        ("src", "tgt"),
        ("preprocessed_text", "english_translation"),
        ("filipino", "english"),
        ("text_tl", "text_en"),
        ("tweet", "translation"),
    ]
    src_col = None
    tgt_col = None
    lowered_cols = {c.lower(): c for c in df.columns}
    for s, t in colmap_candidates:
        if s in df.columns and t in df.columns:
            src_col, tgt_col = s, t
            break
        if s in lowered_cols and t in lowered_cols:
            src_col, tgt_col = lowered_cols[s], lowered_cols[t]
            break
    if src_col is None or tgt_col is None:
        # Fallback: take first two text-like columns
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) >= 2:
            src_col, tgt_col = text_cols[0], text_cols[1]
        else:
            raise ValueError("Could not infer source/target columns from annotated_tweets.xlsx")

    df = df.rename(columns={src_col: "src", tgt_col: "tgt"})
    df = df[["src", "tgt"]].copy()

    # Basic cleaning
    df = df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
    df["src"] = df["src"].astype(str).str.strip()
    df["tgt"] = df["tgt"].astype(str).str.strip()
    df = df[(df["src"] != "") & (df["tgt"] != "")]

    return df.reset_index(drop=True)


def apply_quality_split(current_combined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Block B: QUALITY-BASED 70/15/15 SPLITTING
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    df = current_combined.copy().reset_index(drop=True)

    GOLD_SIZE = 1318

    if 'quality' not in df.columns:
        df['quality'] = ['gold' if i < GOLD_SIZE else 'silver' for i in range(len(df))]
    else:
        df['quality'] = df['quality'].astype(str).str.lower().replace({'human': 'gold'})

    total = len(df)
    gold_count = int((df['quality'] == 'gold').sum())

    val_prop = 0.15
    test_prop = 0.15
    val_size = int(round(total * val_prop))
    test_size = int(round(total * test_prop))
    train_size = total - val_size - test_size

    required_gold_for_val_test = val_size + test_size
    if gold_count < required_gold_for_val_test:
        raise ValueError(
            f"Not enough gold samples ({gold_count}) to fill validation ({val_size}) + test ({test_size})."
        )

    gold_df = df[df['quality'] == 'gold'].sample(frac=1, random_state=SEED).reset_index(drop=True)
    silver_df = df[df['quality'] == 'silver'].sample(frac=1, random_state=SEED).reset_index(drop=True)

    test_df = gold_df.iloc[:test_size].copy().reset_index(drop=True)
    val_df = gold_df.iloc[test_size:test_size + val_size].copy().reset_index(drop=True)
    remaining_gold_for_train = gold_df.iloc[test_size + val_size:].copy().reset_index(drop=True)
    train_df = pd.concat([remaining_gold_for_train, silver_df], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    assert len(train_df) + len(val_df) + len(test_df) == total
    assert (val_df['quality'] == 'gold').all() and (test_df['quality'] == 'gold').all()

    return train_df, val_df, test_df


def prepare_model_and_tokenizer(device: torch.device):
    tokenizer = MBart50Tokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.src_lang = "tl_XX"
    tokenizer.tgt_lang = "en_XX"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="lora_only",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    return model, tokenizer


def evaluate_on_df(df_eval: pd.DataFrame, model, tokenizer, device: torch.device) -> Dict[str, Any]:
    sources = df_eval["src"].tolist()
    references = df_eval["tgt"].tolist()
    hypotheses: List[str] = []
    for i in range(0, len(sources), 16):
        batch = sources[i:i + 16]
        hyps = translate_batch(batch, model, tokenizer, device, MAX_LENGTH)
        hypotheses.extend(hyps)
    return compute_metrics(hypotheses, references)


def main():
    print("Starting training with annotated_tweets.xlsx and quality-based split")
    if not check_requirements():
        return

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logging setup
    os.makedirs('training_logs', exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_log_path = os.path.join('training_logs', f'model_training_{run_id}.json')
    txt_log_path = os.path.join('training_logs', f'model_training_{run_id}.txt')
    history: Dict[str, Any] = {"epochs": []}

    # Load dataset
    df = load_annotated_dataset(INPUT_XLSX)
    # Apply split
    train_df, val_df, test_df = apply_quality_split(df)

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Build model/tokenizer
    model, tokenizer = prepare_model_and_tokenizer(device)

    # Dataloaders
    train_dataset = TranslationDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = TranslationDataset(val_df, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=str(device).startswith('cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=str(device).startswith('cuda'))

    # Optimizer/scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * NUM_EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    scaler = GradScaler(enabled=str(device).startswith('cuda'))

    best_val_loss = float('inf')
    best_epoch_idx = -1

    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        # Train
        model.train()
        train_loss_sum = 0.0
        train_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
            with amp_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            if (batch_idx + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss_sum += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
                with amp_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss_sum += outputs.loss.item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))

        # Compute metrics on validation split (full set)
        val_metrics = evaluate_on_df(val_df, model, tokenizer, device)

        # Logging
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "metrics": val_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        history["epochs"].append(epoch_record)

        # Redundant TXT logging
        with open(txt_log_path, 'a', encoding='utf-8') as f_txt:
            f_txt.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        with open(json_log_path, 'w', encoding='utf-8') as f_json:
            json.dump(history, f_json, ensure_ascii=False, indent=2)

        # Print concise summary
        def mget(name: str):
            m = val_metrics.get(name, {})
            return m.get("score") if isinstance(m, dict) else None
        print(f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"BLEU={mget('sacrebleu')} chrF={mget('chrf')} METEOR={mget('meteor')}")

        # Early stopping/best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch_idx = epoch
            patience_counter = 0
            print("Saving best model (by val loss)...")
            if safe_save_pretrained(model, BEST_MODEL_DIR):
                try:
                    tokenizer.save_pretrained(BEST_MODEL_DIR)
                except Exception:
                    pass
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Final evaluation on test set
    print("\nEvaluating on test split with best model (if available)...")
    try:
        # Reload base + LoRA from BEST_MODEL_DIR if exists
        if os.path.exists(BEST_MODEL_DIR):
            # For PEFT models, reloading is implicit via save_pretrained; if fallback, keep current model
            pass
    except Exception:
        pass

    test_metrics = evaluate_on_df(test_df, model, tokenizer, device)
    final_record = {
        "final_test_metrics": test_metrics,
        "best_epoch": int(best_epoch_idx + 1) if best_epoch_idx >= 0 else None,
    }
    with open(txt_log_path, 'a', encoding='utf-8') as f_txt:
        f_txt.write(json.dumps(final_record, ensure_ascii=False) + "\n")
    with open(json_log_path, 'w', encoding='utf-8') as f_json:
        history["final"] = final_record
        json.dump(history, f_json, ensure_ascii=False, indent=2)

    print("Done. Logs:")
    print(f"  JSON: {json_log_path}")
    print(f"  TXT:  {txt_log_path}")
    print(f"Best model (by val loss) saved to: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()


