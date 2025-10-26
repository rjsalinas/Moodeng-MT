#!/usr/bin/env python3
"""
Training v4: mBART50 + LoRA with regularization-focused settings.

Dataset: annotated_tweets.xlsx (src → tgt), 70/15/15 split
Hyperparameters are aligned with training_v3.py.
Logs per epoch: train_loss, val_loss, BLEU, chrF.
"""

import os
import json
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
from tqdm import tqdm

from transformers import MBartForConditionalGeneration, MBart50Tokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType


# Constants (match v3)
IGNORE_INDEX = -100
MAX_LENGTH = 128
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 200
PATIENCE = 3

BASE_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
BEST_MODEL_DIR = "mbart50-finetuned-LoRA-best_v4"
INPUT_CSV = "annotated_tweets.xlsx"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, logits, labels):
        pred, target = mask_ignore_index(logits, labels)
        log_probs = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
        true_dist.scatter_(-1, target.unsqueeze(-1), 1 - self.smoothing)
        loss = -true_dist * log_probs
        loss = loss.sum(dim=-1)
        return loss.mean()


def mask_ignore_index(logits, labels):
    mask = labels != IGNORE_INDEX
    return logits[mask], labels[mask]


class TranslationDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item["src"]
        tgt_text = item["tgt"]
        src_enc = self.tokenizer(src_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        tgt_enc = self.tokenizer(tgt_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = tgt_enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        return {"input_ids": src_enc["input_ids"].squeeze(0), "attention_mask": src_enc["attention_mask"].squeeze(0), "labels": labels.squeeze(0)}


def load_dataset() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    print("📊 Loading dataset...")
    df = pd.read_excel(INPUT_CSV)
    src_col = "src"
    tgt_col = "tgt"
    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError("Expected columns 'src' and 'tgt' not found in annotated_tweets.xlsx")
    df = df[[src_col, tgt_col]].dropna()
    data = [{"src": str(r[src_col]).strip(), "tgt": str(r[tgt_col]).strip()} for _, r in df.iterrows()]
    random.shuffle(data)
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]
    print(f"Dataset split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def compute_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        # Align with reference script: pass references list directly for chrF
        chrf = sacrebleu.corpus_chrf(hypotheses, references)
        results["sacrebleu"] = {"score": float(bleu.score)}
        results["chrf"] = {"score": float(chrf.score)}
    except Exception as e:
        results["sacrebleu"] = {"error": str(e)}
        results["chrf"] = {"error": str(e)}
    return results


def evaluate_model(model, dataloader, tokenizer, device) -> Tuple[Dict[str, Any], float]:
    model.eval()
    hyps, refs = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                gen = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LENGTH, num_beams=4, do_sample=False, no_repeat_ngram_size=3, length_penalty=1.0, repetition_penalty=1.2, early_stopping=True)
            total_loss += float(loss.item())
            n_batches += 1
            for j in range(gen.size(0)):
                hyps.append(tokenizer.decode(gen[j], skip_special_tokens=True))
                ref_tokens = labels[j]
                ref_tokens = ref_tokens[ref_tokens != IGNORE_INDEX]
                ref_tokens = ref_tokens[ref_tokens != tokenizer.pad_token_id]
                refs.append(tokenizer.decode(ref_tokens, skip_special_tokens=True) if ref_tokens.numel() > 0 else "")
    metrics = compute_metrics(hyps, refs)
    val_loss = total_loss / max(1, n_batches)
    return metrics, val_loss


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler, epoch: int) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def main():
    print("🚀 Starting mBART50 + LoRA training (v4) on annotated_tweets.xlsx")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, test_data = load_dataset()
    tokenizer = MBart50Tokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    train_ds = TranslationDataset(train_data, tokenizer)
    val_ds = TranslationDataset(val_data, tokenizer)
    test_ds = TranslationDataset(test_data, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=str(device).startswith('cuda'))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=str(device).startswith('cuda'))
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=str(device).startswith('cuda'))

    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"], lora_dropout=0.3, bias="lora_only", task_type=TaskType.SEQ_2_SEQ_LM)
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    total_steps = len(train_dl) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(500, int(0.03 * total_steps)), num_training_steps=total_steps)
    criterion = LabelSmoothingLoss(0.1)
    scaler = GradScaler() if device.type == 'cuda' else None

    best_val = float('inf')
    patience_ctr = 0
    logs = []
    os.makedirs("training_logs", exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, criterion, device, scaler, epoch)
        # Compute BLEU/chrF on full train set; keep val_loss for early stopping
        train_metrics, _ = evaluate_model(model, train_dl, tokenizer, device)
        val_metrics, val_loss = evaluate_model(model, val_dl, tokenizer, device)
        bleu = train_metrics.get('sacrebleu', {}).get('score', 0.0)
        chrf = train_metrics.get('chrf', {}).get('score', 0.0)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train BLEU: {bleu:.2f} | Train chrF: {chrf:.2f}")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "bleu": bleu, "chrf": chrf})
        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test
    test_metrics, _ = evaluate_model(model, test_dl, tokenizer, device)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"training_logs/training_v4_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({"config": {"batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "num_epochs": NUM_EPOCHS, "warmup_steps": WARMUP_STEPS, "patience": PATIENCE, "lora_r": 64, "lora_alpha": 128}, "logs": logs, "test_metrics": test_metrics}, f, indent=2)
    print("✅ Training completed!")


if __name__ == "__main__":
    main()


