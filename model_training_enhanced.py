#!/usr/bin/env python3
"""
Enhanced Filipino-to-English Translation Model Fine-tuning Script

This script integrates CalamanCy for enhanced Filipino text preprocessing:
- Linguistic complexity calculation
- Advanced data augmentation
- Quality validation
- Filipino-aware tokenization

CUDA Error Handling Features:
- Automatic CUDA memory management
- Graceful fallback to CPU on persistent errors
- Reduced batch size (2) to prevent VRAM issues
- Error counting and automatic device switching
- Periodic memory cleanup during training
- Fixed custom loss functions to handle -100 ignore index
- Improved language token ID handling for mBART-50
- Enhanced generation parameters to prevent repetition
- Better device mismatch detection and resolution

Requirements:
    pip install torch transformers peft pandas tqdm calamancy spacy[transformers]

Usage:
    python model_training_enhanced.py

Debugging (if CUDA errors persist):
    CUDA_LAUNCH_BLOCKING=1 python model_training_enhanced.py
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
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        # Move model to CPU for safe saving
        model_cpu = model.cpu()
        model_cpu.eval()
        
        # Create CPU state dict
        state = {}
        for k, v in model_cpu.state_dict().items():
            try:
                state[k] = v.detach().cpu()
            except Exception:
                state[k] = v.clone().detach().cpu()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save using torch.save instead of safetensors to avoid CUDA issues
        torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save config separately
        try:
            if hasattr(model_cpu, "config"):
                model_cpu.config.save_pretrained(output_dir)
        except Exception:
            pass
        
        # Move model back to original device
        if torch.cuda.is_available():
            model.to("cuda")
        else:
            model.to("cpu")
            
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

# Advanced loss functions with R-Drop - Fixed to handle -100 ignore index
IGNORE_INDEX = -100

def mask_ignore_index(pred, target):
    """Mask out ignored tokens (-100) from predictions and targets."""
    mask = target != IGNORE_INDEX
    if mask.sum() == 0:  # All tokens are ignored
        return None, None
    return pred[mask], target[mask]

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Mask out ignored tokens first
        pred, target = mask_ignore_index(inputs, targets)
        if pred is None:  # All tokens ignored
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
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
        # Mask out ignored tokens first
        pred, target = mask_ignore_index(pred, target)
        if pred is None:  # All tokens ignored
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class RDropLoss(nn.Module):
    """R-Drop for better generalization and regularization."""
    def __init__(self, alpha=4.0):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    def forward(self, logits1, logits2, labels):
        # Mask out ignored tokens first
        pred1, target = mask_ignore_index(logits1, labels)
        pred2, _ = mask_ignore_index(logits2, labels)
        
        if pred1 is None:  # All tokens ignored
            return torch.tensor(0.0, device=logits1.device, requires_grad=True)
        
        ce_loss = (self.ce(pred1, target) + self.ce(pred2, target)) / 2
        kl_loss = F.kl_div(
            F.log_softmax(pred1, dim=-1), 
            F.softmax(pred2, dim=-1), 
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
    """Translate text from source language to target language.
    Includes a CPU fallback to avoid CUDA device-side asserts crashing the run.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # Ensure model is on the correct device
    current_device = next(model.parameters()).device
    if current_device != device:
        print(f"⚠️  Model device mismatch: {current_device} vs {device}")
        model = model.to(device)
    
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
        enc = {k: v.to(device) for k, v in encoded.items()}
        
        # Get language token IDs from tokenizer (no hardcoding)
        bos_id = tokenizer.lang_code_to_id.get(tgt_lang, tokenizer.eos_token_id)
        
        print(f" Debug: src_lang={src_lang}, tgt_lang={tgt_lang}, bos_id={bos_id}")
        print(f" Debug: Input shape: {enc['input_ids'].shape}")
        
        # Generate translation with better parameters
        with torch.no_grad():
            generated_tokens = model.generate(
                **enc,
                forced_bos_token_id=bos_id,
                max_length=max_len,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                do_sample=False,  # Use deterministic generation for better quality
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2  # Prevent repetition
            )
        
        # Decode and return
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"🔍 Debug: Generated tokens: {generated_tokens[0]}")
        print(f"🔍 Debug: Translation: '{translation}'")
        
        return translation
        
    except RuntimeError as e:
        # Fallback to CPU if CUDA asserts occur
        if "CUDA error" in str(e) or "device-side assert" in str(e):
            CUDA_ERROR_OCCURRED = True
            print(f"⚠️  CUDA error in translation, falling back to CPU: {e}")
            try:
                cpu_model = model.to("cpu")
                enc_cpu = {k: v.to("cpu") for k, v in encoded.items()}
                with torch.no_grad():
                    generated_tokens = cpu_model.generate(
                        **enc_cpu,
                        forced_bos_token_id=bos_id,
                        max_length=max_len,
                        num_beams=3,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        length_penalty=0.8,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2
                    )
                translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                # Move model back to original device
                model = model.to(device)
                return translation
            except Exception as cpu_e:
                print(f"⚠️  CPU fallback also failed: {cpu_e}")
                # Move model back to original device
                model = model.to(device)
                return ""
        else:
            print(f"⚠️  Runtime error in translation: {e}")
            return ""
    except Exception as e:
        print(f"⚠️  Unexpected error in translation: {e}")
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
            # Strict filter: drop rows containing @ or # (should be preprocessed away)
            before_ct = len(df)
            mask_clean = ~df["src"].astype(str).str.contains(r"[@#]") & ~df["tgt"].astype(str).str.contains(r"[@#]")
            df = df[mask_clean]
            removed_ct = before_ct - len(df)
            if removed_ct > 0:
                print(f"🧹 Removed {removed_ct} rows containing @ or # from enhanced dataset")
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
            df = df.dropna(subset=["src", "tgt"]).copy()
            df = df[df["src"].astype(str).str.strip() != ""]
            df = df[df["tgt"].astype(str).str.strip() != ""]
            # Strict filter: drop any rows containing @ or # (should be preprocessed away)
            before = len(df)
            mask_clean = ~df["src"].astype(str).str.contains(r"[@#]") & ~df["tgt"].astype(str).str.contains(r"[@#]")
            df = df[mask_clean]
            removed = before - len(df)
            if removed > 0:
                print(f"🧹 Removed {removed} rows containing @ or # from dataset")
            
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
                    
                    # Strict filter: drop rows containing @ or #
                    try:
                        enhanced_df = enhanced_df.dropna(subset=["src", "tgt"]).copy()
                        before_ct2 = len(enhanced_df)
                        mask_clean2 = ~enhanced_df["src"].astype(str).str.contains(r"[@#]") & ~enhanced_df["tgt"].astype(str).str.contains(r"[@#]")
                        enhanced_df = enhanced_df[mask_clean2]
                        removed_ct2 = before_ct2 - len(enhanced_df)
                        if removed_ct2 > 0:
                            print(f"🧹 Removed {removed_ct2} rows containing @ or # from enhanced data")
                    except Exception:
                        pass
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
    
    # Device configuration for GPU training (standardize to cuda:0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.cuda.set_device(0)
            print(f"🚀 CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            print(f"⚠️  CUDA setup warning: {e}")
    print(f"📱 Using device: {device}")
    
    if device.type == "cpu":
        print("⚠️  Warning: CUDA not available. Training will be slower on CPU.")
        print("   Consider installing PyTorch with CUDA support.")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer.src_lang = "tl_XX"  # Filipino
        target_lang = "en_XX"  # English
        tokenizer.tgt_lang = target_lang  # Ensure target language is set for text_target
        
        # Check available language codes
        print(f"Available language codes: {list(tokenizer.lang_code_to_id.keys())}")
        print(f"Filipino (tl_XX) ID: {tokenizer.lang_code_to_id.get('tl_XX', 'NOT FOUND')}")
        print(f"English (en_XX) ID: {tokenizer.lang_code_to_id.get('en_XX', 'NOT FOUND')}")
        
        # Validate language codes exist
        if 'tl_XX' not in tokenizer.lang_code_to_id:
            print("⚠️  Warning: Filipino (tl_XX) language code not found in tokenizer!")
            print("   This may cause translation issues.")
        if 'en_XX' not in tokenizer.lang_code_to_id:
            print("⚠️  Warning: English (en_XX) language code not found in tokenizer!")
            print("   This may cause translation issues.")
        
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

        # Split into train/val/test: 70/15/15
        total = len(full_dataset)
        train_size = int(0.70 * total)
        val_size = int(0.15 * total)
        test_size = int(0.15 * total)
        
        # Ensure exact split by adjusting for rounding
        remaining = total - train_size - val_size - test_size
        if remaining > 0:
            train_size += remaining  # Add any remaining samples to training set
        
        train_dataset, temp_dataset = random_split(full_dataset, [train_size, total - train_size], generator=g)
        val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], generator=g)

        # Create data loaders (GPU optimized with reduced batch size)
        pin = (device.type == "cuda")
        # Reduce batch size to prevent CUDA errors
        batch_size = 2 if device.type == "cuda" else 4
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)

        print(f"DataLoader prepared (70/15/15 split):")
        print(f"   Training samples: {len(train_dataset)} (70%)")
        print(f"   Validation samples: {len(val_dataset)} (15%)")
        print(f"   Test samples: {len(test_dataset)} (15%)")
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
                "encoder_attn.q_proj", "encoder_attn.v_proj", "encoder_attn.k_proj", "encoder_attn.out_proj"  # Cross attention
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
        # Make target language explicit for generation globally
        try:
            model.config.forced_bos_token_id = tokenizer.lang_code_to_id.get("en_XX")
            if hasattr(model, "generation_config"):
                model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id.get("en_XX")
        except Exception:
            pass
        
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
    
    num_epochs = 20  # Full training for GPU
    best_val_loss = float('inf')
    best_bleu = 0.0
    patience_counter = 0
    patience = 5
    
    # CUDA error tracking
    cuda_error_count = 0
    max_cuda_errors = 100  # Maximum CUDA errors before switching to CPU fallback
    
    # Curriculum phases
    curriculum_phases = [
        {"name": "Simple", "complexity_threshold": 5, "epochs": 5},
        {"name": "Medium", "complexity_threshold": 10, "epochs": 5},
        {"name": "Complex", "complexity_threshold": 15, "epochs": 5},
        {"name": "Mixed", "complexity_threshold": float('inf'), "epochs": 5}
    ]
    
    current_phase = 0
    phase_epoch = 0
    
    # Tracking per-epoch metrics for final averages
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_bleus = []

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
            # Use same reduced batch size for curriculum phases
            phase_batch_size = 2 if device.type == "cuda" else 4
            phase_dataloader = DataLoader(phase_dataset, batch_size=phase_batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
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
                # Move batch to device with error handling
                try:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                except Exception as e:
                    print(f"\n⚠️  Device transfer error in batch {batch_idx}: {e}")
                    continue
                if next(model.parameters()).device != device:
                    model.to(device)
                
                # Forward pass with mixed precision and error handling
                from contextlib import nullcontext
                try:
                    amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                    with amp_ctx:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                except RuntimeError as e:
                    if "CUDA error" in str(e) or "device-side assert" in str(e):
                        cuda_error_count += 1
                        print(f"\n⚠️  CUDA error in forward pass (batch {batch_idx}): {e}")
                        print(f"   CUDA errors so far: {cuda_error_count}/{max_cuda_errors}")
                        
                        # Try to clear CUDA cache and continue
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Switch to CPU if too many CUDA errors
                        if cuda_error_count >= max_cuda_errors:
                            print(f"\n🚨 Too many CUDA errors ({cuda_error_count}). Switching to CPU fallback...")
                            device = torch.device("cpu")
                            model = model.to(device)
                            scaler = GradScaler(enabled=False)
                            break
                        
                        continue
                    else:
                        raise e
                
                # Dynamic loss selection based on curriculum phase
                # Check if we have valid labels (not all -100)
                valid_labels = (labels != IGNORE_INDEX).any()
                
                if current_phase == 0:  # Simple phase
                    loss = outputs.loss
                elif current_phase == 1:  # Medium phase
                    if valid_labels:
                        loss = 0.7 * outputs.loss + 0.3 * label_smoothing_loss(outputs.logits, labels)
                    else:
                        loss = outputs.loss  # Fallback to standard loss
                elif current_phase == 2:  # Complex phase
                    if valid_labels:
                        loss = 0.5 * outputs.loss + 0.3 * label_smoothing_loss(outputs.logits, labels) + 0.2 * focal_loss(outputs.logits, labels)
                    else:
                        loss = outputs.loss  # Fallback to standard loss
                else:  # Mixed phase
                    if valid_labels:
                        loss = 0.4 * outputs.loss + 0.3 * label_smoothing_loss(outputs.logits, labels) + 0.2 * focal_loss(outputs.logits, labels) + 0.1 * r_drop_loss(outputs.logits, outputs.logits, labels)
                    else:
                        loss = outputs.loss  # Fallback to standard loss
                
                # Backward pass with mixed precision and error handling
                try:
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation (every 4 batches)
                    if (batch_idx + 1) % 4 == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                except RuntimeError as e:
                    if "CUDA error" in str(e) or "device-side assert" in str(e):
                        cuda_error_count += 1
                        print(f"\n⚠️  CUDA error in backward pass (batch {batch_idx}): {e}")
                        print(f"   CUDA errors so far: {cuda_error_count}/{max_cuda_errors}")
                        
                        # Clear gradients and continue
                        optimizer.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Switch to CPU if too many CUDA errors
                        if cuda_error_count >= max_cuda_errors:
                            print(f"\n🚨 Too many CUDA errors ({cuda_error_count}). Switching to CPU fallback...")
                            device = torch.device("cpu")
                            model = model.to(device)
                            scaler = GradScaler(enabled=False)
                            break
                        
                        continue
                    else:
                        raise e
                
                total_loss += loss.item()
                train_progress.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Phase': curriculum_phases[current_phase]['name'],
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Periodic CUDA memory management
                if batch_idx % 50 == 0 and device.type == "cuda":
                    manage_cuda_memory()
                
            except Exception as e:
                print(f"\n⚠️  Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = total_loss / len(phase_dataloader)
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
                    if next(model.parameters()).device != device:
                        model.to(device)
                    
                    # Use autocast for validation as well
                    amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
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
        
        # BLEU score calculation on validation split with examples
        bleu_scores = []
        max_eval = min(50, len(val_dataset))
        # Resolve subset indices back to original df rows
        try:
            val_indices = list(getattr(val_dataset, 'indices', range(len(val_dataset))))
        except Exception:
            val_indices = list(range(len(val_dataset)))
        
        printed = 0
        for j, subset_idx in enumerate(val_indices[:max_eval]):
            try:
                # Map to original row index in df
                orig_idx = subset_idx
                src_text = str(df.iloc[orig_idx].get("src", "")).strip()
                reference = str(df.iloc[orig_idx].get("tgt", "")).strip()
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

    # Log final averages over all completed epochs
    try:
        if epoch_train_losses:
            logger.info(f"final_avg_train_loss={float(np.mean(epoch_train_losses)):.6f}")
        if epoch_val_losses:
            logger.info(f"final_avg_val_loss={float(np.mean(epoch_val_losses)):.6f}")
        if epoch_bleus:
            logger.info(f"final_avg_bleu={float(np.mean(epoch_bleus)):.6f}")
    except Exception:
        pass
    
    # Test translation
    print("\n🧪 Testing translation...")
    
    # First, test if model can generate anything at all
    print("🔍 Testing basic model generation...")
    try:
        # Simple forward pass test
        test_input = torch.randint(0, tokenizer.vocab_size, (1, 5)).to(device)
        with torch.no_grad():
            test_output = model(input_ids=test_input)
        print(f"✅ Model forward pass works. Output shape: {test_output.logits.shape}")
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return
    
    # Validate language token IDs
    print("🔍 Validating language token IDs...")
    try:
        # Check if the language codes exist in the tokenizer
        filipino_id = tokenizer.lang_code_to_id.get('tl_XX')
        english_id = tokenizer.lang_code_to_id.get('en_XX')
        
        if filipino_id is not None:
            print(f"✅ Filipino (tl_XX) token ID: {filipino_id}")
        else:
            print("⚠️  Filipino (tl_XX) not found in tokenizer")
            
        if english_id is not None:
            print(f"✅ English (en_XX) token ID: {english_id}")
        else:
            print("⚠️  English (en_XX) not found in tokenizer")
            
        # Test tokenization of language codes
        if filipino_id is not None:
            filipino_tokens = tokenizer.encode('tl_XX', add_special_tokens=False)
            print(f"✅ Filipino language code tokenization: {filipino_tokens}")
            
        if english_id is not None:
            english_tokens = tokenizer.encode('en_XX', add_special_tokens=False)
            print(f"✅ English language code tokenization: {english_tokens}")
            
    except Exception as e:
        print(f"⚠️  Language validation error: {e}")
    
    # Test actual translations
    test_texts = [
        "Kamusta ka?",
        "Salamat sa tulong mo.",
        "Magandang umaga."
    ]
    
    for text in test_texts:
        try:
            print(f"\n🔍 Testing: '{text}'")
            translation = translate_text(text, model, tokenizer)
            print(f"🇵🇭 {text}")
            print(f"🇺🇸 {translation}")
            if not translation.strip():
                print("⚠️  Warning: Empty translation generated")
            print()
        except Exception as e:
            print(f"⚠️  Translation error for '{text}': {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
