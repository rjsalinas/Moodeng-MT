#!/usr/bin/env python3
"""
Enhanced Filipino-to-English Translation Model Fine-tuning Script

This script integrates CalamanCy for enhanced Filipino text preprocessing:
- Linguistic complexity calculation
- Advanced data augmentation
- Quality validation
- Filipino-aware tokenization

Requirements:
    pip install torch transformers peft pandas tqdm calamancy spacy[transformers]

Usage:
    python model_training_enhanced.py
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
    sys.stdout.reconfigure(encoding='ascii', errors='ignore')  # remove emojis in console
except Exception:
    pass

# Import enhanced preprocessing
try:
    from enhanced_preprocessing import enhance_filipino_dataset
    CALAMANCY_AVAILABLE = True
    print("CalamanCy integration available")
except ImportError:
    CALAMANCY_AVAILABLE = False
    print("CalamanCy not available, using basic preprocessing")

# Suppress deprecation warnings and PEFT warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*save_embedding_layers.*")

# Set up logging to file per run
os.makedirs('training_logs', exist_ok=True)
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join('training_logs', f'training_{run_timestamp}.log')
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

def safe_save_pretrained(model: torch.nn.Module, output_dir: str) -> bool:
    """Save by materializing a CPU state_dict to avoid CUDA asserts during serialization."""
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        model.eval()
        state = {}
        for k, v in model.state_dict().items():
            try:
                state[k] = v.detach().cpu()
            except Exception:
                state[k] = v.clone().detach().cpu()
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))
        try:
            if hasattr(model, "config"):
                model.config.save_pretrained(output_dir)
        except Exception:
            pass
        return True
    except Exception as e:
        logger.exception("error_saving_model_safetensors")
        print(f"Error saving model to {output_dir}: {e}")
        return False

# Download NLTK data for BLEU calculation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Advanced loss functions with R-Drop
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for better generalization."""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class RDropLoss(nn.Module):
    """R-Drop for better generalization and regularization."""
    def __init__(self, alpha=4.0):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, logits1, logits2, labels):
        ce_loss = (self.ce(logits1, labels) + self.ce(logits2, labels)) / 2
        kl_loss = F.kl_div(
            F.log_softmax(logits1, dim=-1), 
            F.softmax(logits2, dim=-1), 
            reduction='batchmean'
        )
        return ce_loss + self.alpha * kl_loss

class TranslationDataset(Dataset):
    """Custom dataset for translation training."""
    
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Use enhanced source if available, otherwise fall back to original
        src_text = row.get("src_enhanced", row["src"])
        tgt_text = row.get("tgt_enhanced", row["tgt"])
        
        src = self.tokenizer(
            src_text, 
            return_tensors="pt", 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )
        tgt = self.tokenizer(
            tgt_text, 
            return_tensors="pt", 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )
        # Create labels and mask padding to -100 for CrossEntropyLoss
        labels = tgt["input_ids"].squeeze()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels = labels.masked_fill(labels == pad_id, -100)

        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels
        }

def translate_text(text, model, tokenizer, src_lang="tl_XX", tgt_lang="en_XX", max_len=128):
    """Translate text from source language to target language.
    Includes a CPU fallback to avoid CUDA device-side asserts crashing the run.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    tokenizer.src_lang = src_lang
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    # Encode on active device
    enc = {k: v.to(device) for k, v in encoded.items()}
    bos_id = tokenizer.lang_code_to_id.get(tgt_lang, None)
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    try:
        with torch.no_grad():
            generated_tokens = model.generate(
                **enc,
                forced_bos_token_id=bos_id,
                decoder_start_token_id=bos_id,
                max_length=max_len
            )
        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except RuntimeError as e:
        # Fallback to CPU if CUDA asserts occur
        if "CUDA error" in str(e) or "device-side assert" in str(e):
            CUDA_ERROR_OCCURRED = True
            try:
                cpu_model = model.to("cpu")
                enc_cpu = {k: v.to("cpu") for k, v in encoded.items()}
                with torch.no_grad():
                    generated_tokens = cpu_model.generate(
                        **enc_cpu,
                        forced_bos_token_id=bos_id,
                        decoder_start_token_id=bos_id,
                        max_length=max_len
                    )
                return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            except Exception:
                return ""
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

def load_and_enhance_dataset():
    """Load and enhance the dataset using CalamanCy if available."""
    
    print("\n📊 Loading and enhancing dataset...")
    
    try:
        # 0) Prefer a pre-computed enhanced dataset to skip CalamanCy at train time
        if os.path.exists("full_enhanced_parallel_corpus.csv"):
            print("📁 Found precomputed enhanced dataset: full_enhanced_parallel_corpus.csv")
            df = pd.read_csv("full_enhanced_parallel_corpus.csv")
            expected_cols = {"src", "tgt", "complexity_score", "quality_score"}
            if not {"src", "tgt"}.issubset(df.columns):
                print("⚠️  Enhanced file missing src/tgt; attempting to map columns...")
                if "text" in df.columns:
                    df = df.rename(columns={"text": "src"})
                if "english_translation" in df.columns:
                    df = df.rename(columns={"english_translation": "tgt"})
            # Basic validation
            df = df.dropna(subset=["src", "tgt"]).copy()
            df = df[(df["src"].astype(str).str.strip() != "") & (df["tgt"].astype(str).str.strip() != "")]
            print(f"✅ Using enhanced dataset directly: {len(df)} samples")
            # Return as-is without running CalamanCy again
            return df

        # Check if filipino_english_parallel_corpus.csv exists
        if os.path.exists("filipino_english_parallel_corpus.csv"):
            print("📁 Loading filipino_english_parallel_corpus.csv...")
            df = pd.read_csv("filipino_english_parallel_corpus.csv")
            
            # Map your columns to the expected format
            df = df.rename(columns={
                "preprocessed_text": "src",  # Use preprocessed Filipino text as source
                "english_translation": "tgt"  # Use English translation as target
            })
            
            # Basic data cleaning
            df = df.dropna(subset=["src", "tgt"])
            df = df[df["src"].str.strip() != ""]
            df = df[df["tgt"].str.strip() != ""]
            
            print(f"✅ Loaded {len(df)} translation pairs")
            
            # Apply CalamanCy enhancement if available
            if CALAMANCY_AVAILABLE:
                print("🚀 Applying CalamanCy enhancements...")
                try:
                    enhanced_df = enhance_filipino_dataset(df)
                    print(f"✅ CalamanCy enhancement completed: {len(enhanced_df)} enhanced samples")
                    
                    # Show enhancement statistics
                    if 'complexity_score' in enhanced_df.columns:
                        print(f"📊 Complexity range: {enhanced_df['complexity_score'].min():.1f} - {enhanced_df['complexity_score'].max():.1f}")
                        print(f"📝 Average complexity: {enhanced_df['complexity_score'].mean():.1f}")
                    
                    if 'quality_score' in enhanced_df.columns:
                        print(f"📝 Average quality score: {enhanced_df['quality_score'].mean():.2f}")
                        # Some enhanced datasets may not include an explicit boolean 'quality_valid' column.
                        # Fall back to a threshold on quality_score when it's missing.
                        if 'quality_valid' in enhanced_df.columns:
                            quality_counts = enhanced_df['quality_valid'].value_counts()
                            print(f"✅ Quality validation: {quality_counts.get(True, 0)} valid, {quality_counts.get(False, 0)} invalid")
                        else:
                            valid = int((enhanced_df['quality_score'] >= 0.7).sum())
                            invalid = int(len(enhanced_df) - valid)
                            print(f"✅ Quality validation (by score≥0.7): {valid} valid, {invalid} invalid")
                    
                    return enhanced_df
                    
                except Exception as e:
                    print(f"⚠️  CalamanCy enhancement failed: {e}")
                    print("Falling back to basic preprocessing...")
                    return df
            else:
                print("⚠️  CalamanCy not available, using basic preprocessing")
                return df
                
        else:
            print("⚠️  filipino_english_parallel_corpus.csv not found. Creating sample dataset...")
            # Create sample Filipino-English pairs
            sample_data = {
                "src": [
                    "Kamusta ka?",
                    "Salamat sa tulong mo.",
                    "Magandang umaga.",
                    "Paalam na.",
                    "Gusto ko ng kape."
                ],
                "tgt": [
                    "How are you?",
                    "Thank you for your help.",
                    "Good morning.",
                    "Goodbye.",
                    "I want coffee."
                ]
            }
            return pd.DataFrame(sample_data)
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def main():
    """Main training and inference function."""
    global device
    
    # Check requirements first
    if not check_requirements():
        return
    
    print("Starting Enhanced Filipino-to-English Translation Model Fine-tuning")
    print("=" * 70)
    
    if CALAMANCY_AVAILABLE:
        print("CalamanCy Integration: ENABLED")
        print("   - Enhanced complexity calculation")
        print("   - Advanced data augmentation")
        print("   - Linguistic quality validation")
        print("   - Filipino-aware tokenization")
    else:
        print("CalamanCy Integration: DISABLED")
        print("   - Basic preprocessing only")
        print("   - Install with: pip install calamancy spacy[transformers]")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("Warning: CUDA not available. Training will be slower on CPU.")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer.src_lang = "tl_XX"  # Filipino
        target_lang = "en_XX"  # English
        # Ensure pad token exists (fallback to eos) to avoid device-side asserts from -1 padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Load and enhance dataset
    df = load_and_enhance_dataset()
    if df is None or df.empty:
        print("❌ No valid dataset available")
        return
    
    # Prepare DataLoader with train/validation/test split
    print("\nPreparing data loader with train/validation/test split...")
    try:
        # Create full dataset
        full_dataset = TranslationDataset(df, tokenizer)

        # Use a CPU generator for deterministic splits (avoids device mismatch errors)
        g = torch.Generator()

        # Split into train/val/test: 75/15/10 (but you asked 75/15/15; adjust exact counts)
        total = len(full_dataset)
        train_size = int(0.75 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        train_dataset, temp_dataset = random_split(full_dataset, [train_size, total - train_size], generator=g)
        val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], generator=g)

        # Create data loaders
        pin = (device.type == "cuda")
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=pin)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=pin)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, pin_memory=pin)

        print(f"DataLoader prepared:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Training batches: {len(train_dataloader)}")
        print(f"   Validation batches: {len(val_dataloader)}")
        print(f"   Test batches: {len(test_dataloader)}")
    except Exception as e:
        print(f"❌ Error preparing DataLoader: {e}")
        return
    
    # Load model and apply LoRA
    print("\n🤖 Loading model and applying LoRA...")
    try:
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Enhanced LoRA configuration with only supported modules
        lora_config = LoraConfig(
            r=64,  # Increased rank for maximum capacity
            lora_alpha=128,  # Increased alpha proportionally
            target_modules=[
                "q_proj", "v_proj", "k_proj", "out_proj",  # Attention components
                "fc1", "fc2",  # Feed-forward components
                "self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.out_proj",  # Encoder attention
                "encoder_attn.q_proj", "encoder_attn.v_proj", "encoder_attn.k_proj", "encoder_attn.out_proj",  # Cross attention
                "embed_tokens"  # Embedding layer
            ],
            lora_dropout=0.05,  # Reduced dropout for better training
            bias="lora_only",  # Train bias terms for better adaptation
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False
        )
        
        print("🔧 Applying LoRA configuration...")
        model = get_peft_model(model, lora_config)
        
        print("📱 Moving model to device...")
        model = model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model loaded successfully")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"💾 Memory efficient: {trainable_params/total_params*100:.1f}%")
        
        # Verify LoRA was applied correctly
        lora_modules = 0
        for name, module in model.named_modules():
            if "lora" in name.lower():
                lora_modules += 1
        
        print(f"🔧 LoRA modules applied: {lora_modules}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔍 Troubleshooting tips:")
        print("   • Check if PEFT version is compatible with transformers")
        print("   • Try updating: pip install --upgrade peft transformers")
        print("   • Ensure torch version is compatible")
        return
    
    # Advanced Training Setup with multiple loss functions and scheduling
    print("\n🎓 Setting up advanced training configuration...")
    
    # Initialize loss functions
    focal_loss = FocalLoss(alpha=1, gamma=2)
    label_smoothing_loss = LabelSmoothingLoss(classes=tokenizer.vocab_size, smoothing=0.1)
    r_drop_loss = RDropLoss(alpha=4.0)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_dataloader) * 2,  # 2 epochs warmup
        num_training_steps=len(train_dataloader) * 20  # 20 epochs total
    )
    
    # Mixed precision training
    # Enable AMP scaler only on CUDA
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    # Training loop with curriculum learning
    print("\n🎯 Starting training with curriculum learning...")
    
    num_epochs = 20
    best_val_loss = float('inf')
    best_bleu = 0.0
    patience_counter = 0
    patience = 5
    
    # Curriculum phases
    curriculum_phases = [
        {"name": "Simple", "complexity_threshold": 5, "epochs": 5},
        {"name": "Medium", "complexity_threshold": 10, "epochs": 5},
        {"name": "Complex", "complexity_threshold": 15, "epochs": 5},
        {"name": "Mixed", "complexity_threshold": float('inf'), "epochs": 5}
    ]
    
    current_phase = 0
    phase_epoch = 0
    
    for epoch in range(num_epochs):
        # Determine current curriculum phase
        if phase_epoch >= curriculum_phases[current_phase]["epochs"]:
            current_phase = min(current_phase + 1, len(curriculum_phases) - 1)
            phase_epoch = 0
            print(f"\n🔄 Advancing to {curriculum_phases[current_phase]['name']} phase")
        
        phase_epoch += 1
        current_threshold = curriculum_phases[current_phase]["complexity_threshold"]
        
        print(f"\n📚 Epoch {epoch+1}/{num_epochs} - {curriculum_phases[current_phase]['name']} Phase (Threshold: {current_threshold})")
        print("=" * 60)
        
        # Filter data based on current curriculum phase
        if 'complexity_score' in df.columns:
            phase_data = df[df['complexity_score'] <= current_threshold]
            phase_dataset = TranslationDataset(phase_data, tokenizer)
            phase_dataloader = DataLoader(phase_dataset, batch_size=4, shuffle=True)
            print(f"📊 Phase data: {len(phase_data)} samples (complexity ≤ {current_threshold})")
        else:
            phase_dataloader = train_dataloader
            print(f"📊 Using full training data (no complexity scores available)")
        
        # Training
        model.train()
        total_loss = 0
        train_progress = tqdm(phase_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(train_progress):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                if next(model.parameters()).device != device:
                    model.to(device)
                
                # Forward pass with mixed precision (CUDA only)
                from contextlib import nullcontext
                amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                with amp_ctx:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Dynamic loss selection based on curriculum phase
                    if current_phase == 0:  # Simple phase
                        loss = outputs.loss
                    elif current_phase == 1:  # Medium phase
                        loss = 0.7 * outputs.loss + 0.3 * label_smoothing_loss(outputs.logits, labels)
                    elif current_phase == 2:  # Complex phase
                        loss = 0.5 * outputs.loss + 0.3 * label_smoothing_loss(outputs.logits, labels) + 0.2 * focal_loss(outputs.logits, labels)
                    else:  # Mixed phase
                        loss = 0.4 * outputs.loss + 0.3 * label_smoothing_loss(outputs.logits, labels) + 0.2 * focal_loss(outputs.logits, labels) + 0.1 * r_drop_loss(outputs.logits, outputs.logits, labels)
                
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
                    'Phase': curriculum_phases[current_phase]['name'],
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
            except Exception as e:
                print(f"\n⚠️  Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = total_loss / len(phase_dataloader)
        print(f"📊 Average training loss: {avg_train_loss:.4f}")
        
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
                    if next(model.parameters()).device != device:
                        model.to(device)
                    
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
        
        # BLEU score calculation (robust)
        bleu_scores = []
        sample_count = min(20, len(val_dataset))
        
        for i in range(sample_count):
            try:
                if i >= len(df):
                    break
                src_text = str(df.iloc[i].get("src", "")).strip()
                reference = str(df.iloc[i].get("tgt", "")).strip()
                if not src_text or not reference:
                    continue
                translation = translate_text(src_text, model, tokenizer)
                if not translation:
                    continue
                reference_tokens = reference.split()
                translation_tokens = translation.split()
                if not reference_tokens or not translation_tokens:
                    continue
                bleu = sentence_bleu([reference_tokens], translation_tokens, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu)
            except Exception as e:
                print(f"BLEU calculation error for sample {i}: {e}")
                continue
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        print(f"Average BLEU score: {avg_bleu:.4f}")
        logger.info(f"bleu={avg_bleu:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            print("Saving best model...")
            if safe_save_pretrained(model, "fine-tuned-mbart-tl2en-best"):
                print("Best model saved successfully")
                logger.info("saved_best_model=1")
            else:
                print("Best model save failed")
        else:
            patience_counter += 1
        
        if avg_bleu > best_bleu:
            best_bleu = avg_bleu
        
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best BLEU score: {best_bleu:.4f}")
        print(f"Patience counter: {patience_counter}/{patience}")
        logger.info(f"best_val_loss={best_val_loss:.6f}")
        logger.info(f"best_bleu={best_bleu:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            logger.info(f"early_stopped_epoch={epoch+1}")
            break
        
        print("-" * 60)
    
    # Save final model
    print("\nSaving final model...")
    # If a CUDA error occurred earlier (e.g., during BLEU), checkpoint CPU weights only to avoid asserts
    if CUDA_ERROR_OCCURRED:
        try:
            model_cpu = model.to("cpu")
            torch.cuda.empty_cache()
        except Exception:
            model_cpu = model
        try:
            torch.save(model_cpu.state_dict(), os.path.join("fine-tuned-mbart-tl2en", "pytorch_model.bin"))
            os.makedirs("fine-tuned-mbart-tl2en", exist_ok=True)
            print("Final model state_dict saved (CPU)")
            logger.info("saved_final_model_state_dict=1")
        except Exception as e:
            print(f"Final model state_dict save failed: {e}")
            logger.exception("error_saving_final_state_dict")
    else:
        if safe_save_pretrained(model, "fine-tuned-mbart-tl2en"):
            print("Final model saved successfully")
            logger.info("saved_final_model=1")
        else:
            print("Final model save failed")
    
    print("\nTraining completed!")
    print(f"Final validation loss: {avg_val_loss:.4f}")
    print(f"Final BLEU score: {avg_bleu:.4f}")
    print(f"Model saved to: fine-tuned-mbart-tl2en/")
    logger.info(f"final_val_loss={avg_val_loss:.6f}")
    logger.info(f"final_bleu={avg_bleu:.6f}")
    
    # Test translation
    print("\n🧪 Testing translation...")
    test_texts = [
        "Kamusta ka?",
        "Salamat sa tulong mo.",
        "Magandang umaga."
    ]
    
    for text in test_texts:
        try:
            translation = translate_text(text, model, tokenizer)
            print(f"🇵🇭 {text}")
            print(f"🇺🇸 {translation}")
            print()
        except Exception as e:
            print(f"⚠️  Translation error for '{text}': {e}")

if __name__ == "__main__":
    main()
