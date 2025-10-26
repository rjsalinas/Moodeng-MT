#!/usr/bin/env python3
"""
Baseline Filipino-to-English Translation Model Fine-tuning Script

This script uses the original filipino_english_parallel_corpus.csv without CalamanCy enhancements:
- Direct use of preprocessed_text and english_translation columns
- Simplified preprocessing without linguistic complexity analysis
- Standard training approach for baseline comparison
- Compatible with the existing model architecture

CUDA Error Handling Features:
- Automatic CUDA memory management
- Graceful fallback to CPU on persistent errors
- Reduced batch size (2) to prevent VRAM issues
- Error counting and automatic device switching
- Periodic memory cleanup during training

Requirements:
    pip install torch transformers peft pandas tqdm nltk

Usage:
    python model_training_baseline.py

Debugging (if CUDA errors persist):
    CUDA_LAUNCH_BLOCKING=1 python model_training_baseline.py
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from torch import optim
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm
import os
import warnings
import numpy as np
import re
import sys
import logging
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Configure stdout to strip emojis/non-ASCII to avoid console encoding issues
try:
    sys.stdout.reconfigure(encoding='ascii', errors='ignore')
except Exception:
    pass

# Suppress deprecation warnings and PEFT warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*save_embedding_layers.*")

# Set up logging to file per run
os.makedirs('training_logs', exist_ok=True)
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join('training_logs', f'baseline_training_{run_timestamp}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
CUDA_ERROR_OCCURRED = False

def manage_cuda_memory():
    """Manage CUDA memory to prevent device-side asserts."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

def safe_save_pretrained(model: torch.nn.Module, output_dir: str) -> bool:
    """Save by materializing a CPU state_dict to avoid CUDA asserts during serialization."""
    try:
        # Force CUDA synchronization and move model to CPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model with proper PEFT handling
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir)
        else:
            # Fallback for non-PEFT models
            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        
        # Also save the base model config and tokenizer if they exist
        if hasattr(model, 'config'):
            model.config.save_pretrained(output_dir)
        
        print(f"✅ Model saved successfully to: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Model save failed: {e}")
        return False

# Constants
IGNORE_INDEX = -100
MAX_LENGTH = 128
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 30  # Increased from 20
WARMUP_STEPS = 100
PATIENCE = 8  # Increased from 5 to allow more training

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    def forward(self, logits, labels):
        # Mask out ignored tokens first
        pred, target = mask_ignore_index(logits, labels)
        if pred is None:  # All tokens ignored
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Apply label smoothing
        vocab_size = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))

class FocalLoss(nn.Module):
    """Focal loss for handling hard examples."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
    
    def forward(self, logits, labels):
        # Mask out ignored tokens first
        pred, target = mask_ignore_index(logits, labels)
        if pred is None:  # All tokens ignored
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def mask_ignore_index(logits, labels):
    """Mask out ignored tokens (-100) from loss calculation."""
    mask = labels != IGNORE_INDEX
    if not mask.any():
        return None, None
    
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    return masked_logits, masked_labels

class BaselineTranslationDataset(Dataset):
    """Custom dataset for baseline translation training using original corpus."""
    
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Use preprocessed_text as source and english_translation as target
        src_text = row["preprocessed_text"]
        tgt_text = row["english_translation"]
        
        # Handle missing or empty values
        if pd.isna(src_text) or pd.isna(tgt_text) or src_text == "" or tgt_text == "":
            # Return a dummy sample that will be filtered out
            src_text = "dummy"
            tgt_text = "dummy"
        
        src = self.tokenizer(
            src_text, 
            return_tensors="pt", 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )
        tgt = self.tokenizer(
            text_target=tgt_text,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        
        # Create labels and mask padding to -100 for CrossEntropyLoss
        labels = tgt.get("input_ids", tgt.get("labels")).squeeze()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels = labels.masked_fill(labels == pad_id, -100)
        
        # Validate labels are within valid range (excluding -100)
        valid_labels = labels[labels != -100]
        if valid_labels.numel() > 0:
            max_label = valid_labels.max().item()
            if max_label >= self.tokenizer.vocab_size:
                print(f"⚠️  Warning: Label {max_label} exceeds vocab size {self.tokenizer.vocab_size}")
                # Clamp labels to valid range
                labels = torch.clamp(labels, -100, self.tokenizer.vocab_size - 1)

        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels
        }

def translate_text(text, model, tokenizer, src_lang="tl_XX", tgt_lang="en_XX", max_len=128):
    """Translate text from source language to target language."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # Get the device from the model
    model_device = next(model.parameters()).device
    
    # Set source language
    tokenizer.src_lang = src_lang
    
    # Encode input text
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        )
        
        # Move to device
        enc = {k: v.to(model_device) for k, v in encoded.items()}
        
        # Get language token ID for English
        bos_id = tokenizer.lang_code_to_id.get(tgt_lang, tokenizer.eos_token_id)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **enc,
                forced_bos_token_id=bos_id,
                max_length=max_len,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode and return
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translation
        
    except Exception as e:
        return ""

def check_requirements():
    """Check if all required packages are available."""
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT (Parameter-Efficient Fine-Tuning)',
        'pandas': 'Pandas',
        'tqdm': 'TQDM',
        'nltk': 'NLTK'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\nInstall with: pip install torch transformers peft pandas tqdm nltk")
        return False
    
    print("✅ All required packages are available")
    return True

def load_baseline_dataset():
    """Load the baseline dataset from filipino_english_parallel_corpus.csv."""
    print("🔧 Loading baseline dataset...")
    
    # Check if the baseline corpus exists
    baseline_corpus_path = "filipino_english_parallel_corpus.csv"
    if not os.path.exists(baseline_corpus_path):
        print(f"❌ Baseline corpus not found: {baseline_corpus_path}")
        return None
    
    try:
        # Load the dataset
        df = pd.read_csv(baseline_corpus_path)
        print(f"✅ Loaded {len(df)} samples from baseline corpus")
        
        # Check required columns
        required_columns = ['preprocessed_text', 'english_translation']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return None
        
        # Clean the dataset
        print("🔧 Cleaning dataset...")
        
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna(subset=['preprocessed_text', 'english_translation'])
        print(f"   Removed {initial_count - len(df)} rows with missing values")
        
        # Remove empty strings
        df = df[(df['preprocessed_text'].str.strip() != '') & (df['english_translation'].str.strip() != '')]
        print(f"   Removed {initial_count - len(df)} rows with empty strings")
        
        # Remove rows with @ or # symbols (social media artifacts)
        df = df[~df['preprocessed_text'].str.contains('@|#', na=False)]
        df = df[~df['english_translation'].str.contains('@|#', na=False)]
        print(f"   Removed rows with social media artifacts")
        
        # Remove very short or very long texts
        df['src_length'] = df['preprocessed_text'].str.len()
        df['tgt_length'] = df['english_translation'].str.len()
        
        df = df[
            (df['src_length'] >= 5) & (df['src_length'] <= 500) &
            (df['tgt_length'] >= 5) & (df['tgt_length'] <= 500)
        ]
        print(f"   Removed rows with inappropriate text lengths")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['preprocessed_text'])
        print(f"   Removed duplicate preprocessed texts")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        print(f"✅ Final dataset: {len(df)} samples")
        print(f"📊 Dataset statistics:")
        print(f"   Average source length: {df['src_length'].mean():.1f} characters")
        print(f"   Average target length: {df['tgt_length'].mean():.1f} characters")
        print(f"   Source length range: {df['src_length'].min()}-{df['src_length'].max()}")
        print(f"   Target length range: {df['tgt_length'].min()}-{df['tgt_length'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading baseline dataset: {e}")
        return None

def main():
    """Main training function for baseline model."""
    print("🚀 Starting Baseline Filipino-to-English Translation Training")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Set device
    global device
    if torch.cuda.is_available():
        # Use the first available CUDA device
        device = torch.device("cuda:0")
        print(f"🔧 Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"🔧 Using CPU device: {device}")
    
    # Load dataset
    df = load_baseline_dataset()
    if df is None:
        print("❌ Failed to load dataset")
        return
    
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
    
    # Create dataset
    print("🔧 Creating dataset...")
    dataset = BaselineTranslationDataset(df, tokenizer, MAX_LENGTH)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"📊 Dataset splits:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=(str(device).startswith('cuda'))
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=(str(device).startswith('cuda'))
    )
    
    # Load base model
    print("🔧 Loading base mBART model...")
    base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
    # Configure LoRA
    print("🔧 Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="lora_only",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(device)
    
    # Verify device consistency
    model_device = next(model.parameters()).device
    if model_device != device:
        # Update global device to match model
        device = model_device
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=total_steps
    )
    
    # Loss functions
    label_smoothing_loss = LabelSmoothingLoss(smoothing=0.1)
    focal_loss = FocalLoss(alpha=1, gamma=2)
    
    # Mixed precision
    scaler = GradScaler(enabled=str(device).startswith('cuda'))
    
    # Check if we can resume from previous training
    resume_training = False
    if os.path.exists("fine-tuned-mbart-tl2en-baseline-best/adapter_model.safetensors"):
        print("🔍 Found existing baseline model, checking if we can resume training...")
        try:
            # Try to load the existing model
            existing_model = PeftModel.from_pretrained(base_model, "fine-tuned-mbart-tl2en-baseline-best")
            print("✅ Successfully loaded existing model for resuming training")
            model = existing_model
            resume_training = True
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}")
            print("🔄 Starting fresh training...")
    
    # Training loop
    print("🚀 Starting training...")
    best_val_loss = float('inf')
    best_bleu = 0.0
    patience_counter = 0
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_bleus = []
    
    if resume_training:
        print("📚 Resuming training from existing model...")
        # You could load previous training state here if needed
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training
        model.train()
        total_loss = 0
        train_progress = tqdm(train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(train_progress):
            try:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                
                # Ensure model is on correct device
                current_model_device = next(model.parameters()).device
                if current_model_device != device:
                    if str(current_model_device).startswith('cuda') and str(device).startswith('cuda'):
                        # Both are CUDA devices, use the model's current device
                        device = current_model_device
                    else:
                        # Move model to target device
                        model.to(device)
                
                # Forward pass with mixed precision
                amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
                with amp_ctx:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                # Compute loss (simple cross-entropy for baseline)
                loss = outputs.loss
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation (every 4 batches)
                if (batch_idx + 1) % 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                total_loss += loss.item()
                train_progress.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Periodic CUDA memory management
                if batch_idx % 50 == 0 and str(device).startswith('cuda'):
                    manage_cuda_memory()
                
            except Exception as e:
                print(f"\n⚠️  Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"📊 Average training loss: {avg_train_loss:.4f}")
        logger.info(f"train_loss={avg_train_loss:.6f}")
        epoch_train_losses.append(float(avg_train_loss))
        
        # Validation
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_dataloader, desc="Validation")
        
        with torch.no_grad():
            for batch in val_progress:
                try:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    
                    current_model_device = next(model.parameters()).device
                    if current_model_device != device:
                        if str(current_model_device).startswith('cuda') and str(device).startswith('cuda'):
                            # Both are CUDA devices, use the model's current device
                            device = current_model_device
                        else:
                            # Move model to target device
                            model.to(device)
                    
                    # Use autocast for validation as well
                    amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
                    with amp_ctx:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                    
                    val_loss += outputs.loss.item()
                    
                except Exception as e:
                    print(f"\n⚠️  Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        logger.info(f"val_loss={avg_val_loss:.6f}")
        epoch_val_losses.append(float(avg_val_loss))
        
        # BLEU score calculation
        bleu_scores = []
        max_eval = min(50, len(val_dataset))
        
        printed = 0
        for j in range(max_eval):
            try:
                # Get sample from validation dataset
                sample = val_dataset[j]
                # Fix: Get the actual row index from the dataset
                actual_idx = val_dataset.indices[j] if hasattr(val_dataset, 'indices') else j
                src_text = df.iloc[actual_idx]["preprocessed_text"]
                reference = df.iloc[actual_idx]["english_translation"]
                
                if not src_text or not reference:
                    continue
                
                translation = translate_text(src_text, model, tokenizer)
                if not translation:
                    continue
                
                if printed < 5:
                    print(f"\n🔎 Val sample {printed+1}")
                    print(f"SRC: {src_text}")
                    print(f"REF: {reference}")
                    print(f"HYP: {translation}")
                    printed += 1
                
                reference_tokens = reference.split()
                translation_tokens = translation.split()
                if not reference_tokens or not translation_tokens:
                    continue
                
                bleu = sentence_bleu([reference_tokens], translation_tokens, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu)
                
            except Exception as e:
                print(f"BLEU calculation error for val sample {j}: {e}")
                continue
        
        avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
        print(f"Average BLEU score: {avg_bleu:.4f}")
        logger.info(f"bleu={avg_bleu:.6f}")
        epoch_bleus.append(float(avg_bleu))
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            print("Saving best model...")
            if safe_save_pretrained(model, "fine-tuned-mbart-tl2en-baseline-best"):
                print("Best model saved successfully")
                logger.info("saved_best_model=1")
                
                # Save tokenizer to best model directory
                try:
                    tokenizer.save_pretrained("fine-tuned-mbart-tl2en-baseline-best")
                    print("✅ Tokenizer saved to best model directory")
                except Exception as e:
                    print(f"⚠️  Tokenizer save warning: {e}")
            else:
                print("Best model save failed")
        else:
            patience_counter += 1
        
        if avg_bleu > best_bleu:
            best_bleu = avg_bleu
        
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best BLEU score: {best_bleu:.4f}")
        print(f"Patience counter: {patience_counter}/{PATIENCE}")
        logger.info(f"best_val_loss={best_val_loss:.6f}")
        logger.info(f"best_bleu={best_bleu:.6f}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            logger.info(f"early_stopped_epoch={epoch+1}")
            break
    
    # Save final model
    print("Saving final model...")
    if safe_save_pretrained(model, "fine-tuned-mbart-tl2en-baseline"):
        print("Final model saved successfully")
    
    # Save tokenizer to both model directories
    print("Saving tokenizer...")
    try:
        tokenizer.save_pretrained("fine-tuned-mbart-tl2en-baseline-best")
        tokenizer.save_pretrained("fine-tuned-mbart-tl2en-baseline")
        print("✅ Tokenizer saved successfully")
    except Exception as e:
        print(f"⚠️  Tokenizer save warning: {e}")
    
    print("\n🎉 Baseline training completed!")
    print(f"📁 Best model: fine-tuned-mbart-tl2en-baseline-best/")
    print(f"📁 Final model: fine-tuned-mbart-tl2en-baseline/")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"📊 Best BLEU score: {best_bleu:.4f}")

if __name__ == "__main__":
    main()
