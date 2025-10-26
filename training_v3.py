#!/usr/bin/env python3
"""
Training v3: mBART50 + LoRA derived from model_training_baseline.py with regularization-focused settings.

Configuration:
- Dataset: filipino_english_parallel_corpus.csv with 70/15/15 split
- Lower LR (2e-5) for smoother convergence
- Stronger regularization: dropout (model + LoRA) and L2 weight decay (1e-4)
- LoRA simplified (r=64, alpha=128) to reduce capacity
- Evaluate BLEU/chrF every epoch (200 samples for speed, full at end)
- Best model saved to mbart50-finetuned-LoRA-best_v3
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

from transformers import MBartForConditionalGeneration, MBart50Tokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType


# Constants
IGNORE_INDEX = -100
MAX_LENGTH = 128
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10  # Original epochs
WARMUP_STEPS = 200  # Original warmup
PATIENCE = 3  # Original patience

BASE_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
BEST_MODEL_DIR = "mbart50-finetuned-LoRA-best_v3"
INPUT_CSV = "filipino_english_parallel_corpus.csv"


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
    """Label smoothing loss for better generalization."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    def forward(self, logits, labels):
        # Mask out ignored tokens first
        pred, target = mask_ignore_index(logits, labels)
        
        # Calculate label smoothing loss
        log_probs = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
        true_dist.scatter_(-1, target.unsqueeze(-1), 1 - self.smoothing)
        
        loss = -true_dist * log_probs
        loss = loss.sum(dim=-1)
        return loss.mean()


def mask_ignore_index(logits, labels):
    """Mask out ignored tokens from logits and labels."""
    mask = labels != IGNORE_INDEX
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    return masked_logits, masked_labels


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

        # Tokenize source and target
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create labels (shifted target)
        labels = tgt_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": src_encoding["input_ids"].squeeze(0),
            "attention_mask": src_encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


def load_dataset() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load and split filipino_english_parallel_corpus.csv with 70/15/15 split.

    Uses 'preprocessed_text' as src and 'english_translation' as tgt.
    """
    print("📊 Loading dataset...")
    
    # Load CSV file
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    
    # Column mapping
    src_col = "preprocessed_text"
    tgt_col = "english_translation"
    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError("Expected columns 'preprocessed_text' and 'english_translation' not found in filipino_english_parallel_corpus.csv")
    
    # Filter missing
    df = df.dropna(subset=[src_col, tgt_col])
    print(f"After filtering missing data: {len(df)} rows")
    
    # Build data list
    data = []
    for _, row in df.iterrows():
        data.append({
            "src": str(row[src_col]).strip(),
            "tgt": str(row[tgt_col]).strip()
        })
    
    # Simple random split (70/15/15)
    random.shuffle(data)
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data


def compute_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, Any]:
    """Compute BLEU and chrF scores (METEOR removed for stability)."""
    results: Dict[str, Any] = {}
    
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        results["sacrebleu"] = {"score": float(bleu.score)}
        # Align with reference script: pass references list directly for chrF
        chrf = sacrebleu.corpus_chrf(hypotheses, references)
        results["chrf"] = {"score": float(chrf.score)}
    except Exception as e:
        results["sacrebleu"] = {"error": str(e)}
        results["chrf"] = {"error": str(e)}
    
    return results


def evaluate_model(model, dataloader, tokenizer, device, max_samples: int = None) -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    model.eval()
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_samples and i * BATCH_SIZE >= max_samples:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Generate translations
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_LENGTH,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode predictions and references
            for j in range(len(outputs)):
                pred_text = tokenizer.decode(outputs[j], skip_special_tokens=True)
                
                # Handle reference text properly - filter out IGNORE_INDEX tokens
                ref_tokens = labels[j]
                # Remove IGNORE_INDEX tokens
                ref_tokens = ref_tokens[ref_tokens != IGNORE_INDEX]
                # Remove padding tokens
                ref_tokens = ref_tokens[ref_tokens != tokenizer.pad_token_id]
                
                if len(ref_tokens) > 0:
                    ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                else:
                    ref_text = ""
                
                hypotheses.append(pred_text)
                references.append(ref_text)
    
    return compute_metrics(hypotheses, references)


def compute_validation_loss(
    model,
    dataloader,
    criterion,
    device,
) -> float:
    """Compute average validation loss across the full validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val loss"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = criterion(outputs.logits, labels)
            total_loss += float(loss.item())
            num_batches += 1
    return total_loss / max(1, num_batches)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler, epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = criterion(outputs.logits, labels)
        
        scaler.scale(loss).backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def main():
    """Main training function."""
    print("🚀 Starting mBART50 + LoRA training (v3) with regularization settings")
    
    # Set random seed
    set_seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    train_data, val_data, test_data = load_dataset()
    
    # Load tokenizer and model
    print("📥 Loading tokenizer and model...")
    tokenizer = MBart50Tokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    
    # Create datasets and dataloaders
    train_dataset = TranslationDataset(train_data, tokenizer)
    val_dataset = TranslationDataset(val_data, tokenizer)
    test_dataset = TranslationDataset(test_data, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=str(device).startswith('cuda'))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=str(device).startswith('cuda'))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=str(device).startswith('cuda'))
    
    # Configure LoRA
    print("🔧 Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=64,  # Reduced capacity
        lora_alpha=128,  # Reduced alpha
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.3,  # stronger dropout for regularization
        bias="lora_only",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(500, int(0.03 * total_steps)),
        num_training_steps=total_steps,
    )
    
    # Loss function
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    best_bleu = 0.0
    patience_counter = 0
    training_log = []
    
    print(f"🎯 Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n📈 Epoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, criterion, device, scaler, epoch)
        
        # Evaluate BLEU/chrF on the full TRAIN set per request
        print("📊 Evaluating (train BLEU/chrF on full train set)...")
        train_metrics_eval = evaluate_model(model, train_dataloader, tokenizer, device)
        # Compute validation loss on full validation set
        val_loss = compute_validation_loss(model, val_dataloader, criterion, device)
        
        # Log results
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics_eval
        }
        training_log.append(epoch_log)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train BLEU: {train_metrics_eval.get('sacrebleu', {}).get('score', 0):.2f}")
        print(f"Train chrF: {train_metrics_eval.get('chrf', {}).get('score', 0):.2f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Check for improvement based on validation loss
        if (best_bleu == 0.0) or (val_loss < best_bleu):
            best_bleu = val_loss
            patience_counter = 0
            
            # Save best model
            print(f"💾 Saving best model (ValLoss: {best_bleu:.4f})")
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
    
    # Final evaluation on test set (FULL evaluation for final results)
    print("\n🧪 Final evaluation on test set (FULL)...")
    test_metrics = evaluate_model(model, test_dataloader, tokenizer, device)
    
    print(f"Test BLEU: {test_metrics.get('sacrebleu', {}).get('score', 0):.2f}")
    print(f"Test chrF: {test_metrics.get('chrf', {}).get('score', 0):.2f}")
    
    # Save training log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_logs/training_v3_{timestamp}.json"
    os.makedirs("training_logs", exist_ok=True)
    
    with open(log_file, 'w') as f:
        json.dump({
            "config": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "warmup_steps": WARMUP_STEPS,
                "patience": PATIENCE,
                "lora_r": 128,
                "lora_alpha": 256,
                "dataset": "cleaned_csv"
            },
            "training_log": training_log,
            "test_metrics": test_metrics
        }, f, indent=2)
    
    print(f"📝 Training log saved to {log_file}")
    print("✅ Training completed!")


if __name__ == "__main__":
    main()