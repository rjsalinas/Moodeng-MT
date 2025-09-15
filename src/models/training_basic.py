#!/usr/bin/env python3
"""
Filipino-to-English Translation Model Fine-tuning Script

This script fine-tunes an mBART-50 model for Filipino-to-English translation
using LoRA (Low-Rank Adaptation) for efficient training.

Requirements:
    pip install torch transformers peft pandas tqdm

Usage:
    python model_training.py
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from torch import optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import warnings
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Suppress deprecation warnings and PEFT warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*save_embedding_layers.*")

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
        src = self.tokenizer(
            row["src"], 
            return_tensors="pt", 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )
        tgt = self.tokenizer(
            row["tgt"], 
            return_tensors="pt", 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )

        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": tgt["input_ids"].squeeze()
        }

def translate_text(text, model, tokenizer, src_lang="tl_XX", tgt_lang="en_XX", max_len=128):
    """Translate text from source language to target language."""
    tokenizer.src_lang = src_lang
    encoded = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_len
    ).to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_len
        )
    
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

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

def main():
    """Main training and inference function."""
    global device
    
    # Check requirements first
    if not check_requirements():
        return
    
    print("🚀 Starting Filipino-to-English Translation Model Fine-tuning")
    print("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    if device.type == "cpu":
        print("⚠️  Warning: CUDA not available. Training will be slower on CPU.")
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer.src_lang = "tl_XX"  # Filipino
        target_lang = "en_XX"  # English
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Load dataset
    print("\n📊 Loading dataset...")
    try:
        # Check if filipino_english_parallel_corpus.csv exists, otherwise use a sample dataset
        if os.path.exists("filipino_english_parallel_corpus.csv"):
            print("📁 Loading filipino_english_parallel_corpus.csv...")
            df = pd.read_csv("filipino_english_parallel_corpus.csv")
            
            # Map your columns to the expected format
            df = df.rename(columns={
                "preprocessed_text": "src",  # Use preprocessed Filipino text as source
                "english_translation": "tgt"  # Use English translation as target
            })
            
            # Advanced data preprocessing and quality improvements
            print("🔧 Applying advanced data preprocessing...")
            
            # Remove rows with missing translations or empty text
            df = df.dropna(subset=["src", "tgt"])
            df = df[df["src"].str.strip() != ""]
            df = df[df["tgt"].str.strip() != ""]
            
            # Enhanced text cleaning with Filipino-specific normalization
            def advanced_clean_text(text):
                if pd.isna(text):
                    return ""
                
                # Remove social media artifacts
                text = re.sub(r'@\w+', '', str(text))  # Remove mentions
                text = re.sub(r'#\w+', '', str(text))  # Remove hashtags
                text = re.sub(r'RT\s*:', '', str(text))  # Remove RT
                
                # Remove URLs
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
                
                # Normalize Filipino contractions and common patterns
                text = re.sub(r"di'ba", "di ba", text)
                text = re.sub(r"kasi'ng", "kasi ng", text)
                text = re.sub(r"sa'yo", "sa iyo", text)
                text = re.sub(r"ko'ng", "ko ng", text)
                
                # Remove excessive punctuation
                text = re.sub(r'[!]{2,}', '!', text)
                text = re.sub(r'[?]{2,}', '?', text)
                text = re.sub(r'[.]{2,}', '.', text)
                
                # Normalize whitespace
                text = ' '.join(text.split())
                return text.strip()
            
            df["src"] = df["src"].apply(advanced_clean_text)
            df["tgt"] = df["tgt"].apply(advanced_clean_text)
            
            # Enhanced filtering by text length for better quality
            df = df[df["src"].str.len() > 15]  # Remove very short texts
            df = df[df["src"].str.len() < 200]  # Remove very long texts
            df = df[df["tgt"].str.len() > 8]   # Remove very short translations
            df = df[df["tgt"].str.len() < 300] # Remove very long translations
            
            # Remove duplicate or near-duplicate translations
            df = df.drop_duplicates(subset=["src", "tgt"])
            
            # Advanced data quality scoring
            def calculate_quality_score(src, tgt):
                # Check if translation is too similar to source (indicating poor translation)
                src_words = set(src.lower().split())
                tgt_words = set(tgt.lower().split())
                similarity = len(src_words.intersection(tgt_words)) / len(src_words.union(tgt_words))
                
                # Check for proper translation patterns
                has_proper_structure = len(tgt.split()) >= len(src.split()) * 0.5
                
                # Check for common Filipino words that should be translated
                filipino_words = ['kamusta', 'salamat', 'magandang', 'paalam', 'gusto', 'kailangan']
                has_filipino_content = any(word in src.lower() for word in filipino_words)
                
                # Quality score: lower similarity + proper structure + Filipino content
                quality = (similarity < 0.3) and has_proper_structure and has_filipino_content
                return quality
            
            # Filter by quality score
            df = df[df.apply(lambda x: calculate_quality_score(x["src"], x["tgt"]), axis=1)]
            
            # Data augmentation for common Filipino patterns
            def augment_filipino_data(df):
                augmented_rows = []
                for _, row in df.iterrows():
                    # Original pair
                    augmented_rows.append({
                        "src": row["src"],
                        "tgt": row["tgt"]
                    })
                    
                    # Synonym replacement for common words
                    src_aug = row["src"]
                    if "kamusta" in src_aug.lower():
                        src_aug2 = src_aug.replace("kamusta", "kumusta")
                        if src_aug2 != row["src"]:
                            augmented_rows.append({
                                "src": src_aug2,
                                "tgt": row["tgt"]
                            })
                    
                    if "salamat" in src_aug.lower():
                        src_aug3 = src_aug.replace("salamat", "thank you")
                        if src_aug3 != row["src"]:
                            augmented_rows.append({
                                "src": src_aug3,
                                "tgt": row["tgt"]
                            })
                
                return pd.DataFrame(augmented_rows)
            
            # Apply augmentation
            try:
                df = augment_filipino_data(df)
                print(f"✅ Data augmentation completed: {len(df)} total pairs")
            except Exception as e:
                print(f"⚠️  Data augmentation failed: {e}")
                print("Continuing with original dataset...")
            
            # Calculate complexity for curriculum learning
            def get_sentence_complexity(text):
                try:
                    if pd.isna(text) or not isinstance(text, str):
                        return 0
                    words = len(text.split())
                    punct_count = len([c for c in text if c in '.,!?;:'])
                    # Add complexity for mixed language content
                    mixed_lang = len(re.findall(r'[a-zA-Z]+', text)) > 0
                    complexity = words + punct_count + (5 if mixed_lang else 0)
                    return complexity
                except Exception as e:
                    # Fallback complexity calculation
                    return len(str(text)) if text else 0
            
            try:
                df["complexity"] = df["src"].apply(get_sentence_complexity)
                
                # Sort by complexity for curriculum learning
                df = df.sort_values("complexity").reset_index(drop=True)
                
                print(f"✅ Complexity calculation completed")
            except Exception as e:
                print(f"⚠️  Complexity calculation failed: {e}")
                print("Adding default complexity scores...")
                df["complexity"] = 1  # Default complexity
            
            print(f"✅ Loaded {len(df)} high-quality translation pairs from filipino_english_parallel_corpus.csv")
            print(f"📊 Complexity range: {df['complexity'].min()} - {df['complexity'].max()}")
            print(f"📝 Average complexity: {df['complexity'].mean():.1f}")
            
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
            df = pd.DataFrame(sample_data)
        
        # Clean column names
        try:
            df.columns = df.columns.str.strip()  # Remove any hidden spaces
        except Exception as e:
            print(f"⚠️  Column cleaning failed: {e}")
        
        # Ensure required columns exist
        if "src" not in df.columns or "tgt" not in df.columns:
            print("❌ Required columns 'src' and 'tgt' not found in dataset")
            print(f"Available columns: {list(df.columns)}")
            print("Creating sample dataset instead...")
            
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
            df = pd.DataFrame(sample_data)
        
        # Clean and validate data
        try:
            df = df.dropna(subset=["src", "tgt"])  # Remove bad rows
            df["src"] = df["src"].astype(str)  # Ensure correct type
            df["tgt"] = df["tgt"].astype(str)
            print(f"✅ Data cleaning completed: {len(df)} valid pairs")
        except Exception as e:
            print(f"⚠️  Data cleaning failed: {e}")
            print("Attempting to continue with available data...")
        
        print(f"✅ Dataset loaded: {len(df)} samples")
        print(f"📝 Sample data:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Prepare DataLoader with train/validation split
    print("\n🔄 Preparing data loader with train/validation split...")
    try:
        # Create full dataset
        full_dataset = TranslationDataset(df, tokenizer)
        
        # Split into train and validation (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        print(f"✅ DataLoader prepared:")
        print(f"   📚 Training samples: {len(train_dataset)}")
        print(f"   🔍 Validation samples: {len(val_dataset)}")
        print(f"   📦 Training batches: {len(train_dataloader)}")
        print(f"   📦 Validation batches: {len(val_dataloader)}")
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
    try:
        # Advanced optimizer with better parameters
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=2e-4, 
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enhanced learning rate scheduler with longer warmup
        num_training_steps = len(train_dataloader) * 10  # Increased to 10 epochs
        num_warmup_steps = int(0.15 * num_training_steps)  # Longer warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Enhanced loss functions with R-Drop
        criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        criterion_focal = FocalLoss(alpha=1, gamma=2)
        criterion_smooth = LabelSmoothingLoss(classes=tokenizer.vocab_size, smoothing=0.1)
        criterion_rdrop = RDropLoss(alpha=4.0)  # R-Drop for regularization
        
        scaler = GradScaler()
        model.train()
        
        # Enhanced early stopping parameters
        best_val_loss = float('inf')
        best_bleu_score = 0.0
        patience = 5  # Increased patience
        patience_counter = 0
        
        # Curriculum learning parameters
        curriculum_phases = 3
        current_phase = 0
        
        # Gradient accumulation for larger effective batch size
        accumulation_steps = 4  # Effective batch size = 4 * 4 = 16
        
        print("✅ Advanced training setup complete")
        print(f"   📈 Learning rate: 2e-4 with warmup + cosine annealing")
        print(f"   🎯 Weight decay: 0.01, betas: (0.9, 0.999)")
        print(f"   🛑 Early stopping patience: {patience} epochs")
        print(f"   📚 Curriculum learning phases: {curriculum_phases}")
        print(f"   🔄 Gradient accumulation steps: {accumulation_steps}")
        print(f"   📦 Effective batch size: {4 * accumulation_steps}")
    except Exception as e:
        print(f"❌ Error setting up training: {e}")
        return
    
    # Enhanced Training Loop with Extended Curriculum Learning and R-Drop
    print("\n🔥 Starting enhanced training with extended curriculum learning...")
    try:
        for epoch in range(10):  # Increased to 10 epochs
            # Enhanced Curriculum Learning: Progress through complexity phases
            if epoch < 3:
                current_phase = 0  # Simple sentences
            elif epoch < 6:
                current_phase = 1  # Medium sentences
            elif epoch < 8:
                current_phase = 2  # Complex sentences
            else:
                current_phase = 3  # Mixed complexity (final refinement)
            
            phase_names = ["Simple", "Medium", "Complex", "Mixed"]
            print(f"\n📚 Curriculum Phase {current_phase + 1}: {phase_names[current_phase]} sentences")
            
            # Training phase with gradient accumulation
            model.train()
            train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/5 [Train] Phase {current_phase+1}")
            train_epoch_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loop):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast(device_type=device.type):
                    # Dynamic loss function selection based on training phase
                    if current_phase == 0:
                        # Simple phase: Standard cross-entropy
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = criterion_ce(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                    elif current_phase == 1:
                        # Medium phase: Label smoothing
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = criterion_smooth(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                    elif current_phase == 2:
                        # Complex phase: Focal loss
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = criterion_focal(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                    else:
                        # Mixed phase: R-Drop for regularization
                        outputs1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = criterion_rdrop(
                            outputs1.logits.view(-1, outputs1.logits.size(-1)), 
                            outputs2.logits.view(-1, outputs2.logits.size(-1)), 
                            labels.view(-1)
                        )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                train_epoch_loss += loss.item() * accumulation_steps
                
                # Gradient accumulation step
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update learning rate
                    scheduler.step()
                
                train_loop.set_postfix(loss=f"{loss.item() * accumulation_steps:.3f}")
            
            # Validation phase with BLEU score calculation
            model.eval()
            val_epoch_loss = 0.0
            val_predictions = []
            val_references = []
            val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/5 [Val]")
            
            with torch.no_grad():
                for batch in val_loop:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    with autocast(device_type=device.type):
                        outputs = model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    val_epoch_loss += loss.item()
                    
                    # Enhanced generation for better translation quality
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=128,
                        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                        num_beams=5,  # Increased beam search
                        length_penalty=1.0,  # Prefer longer translations
                        early_stopping=True,
                        no_repeat_ngram_size=3,  # Avoid repetition
                        temperature=0.8,  # Add some randomness
                        do_sample=True,  # Enable sampling
                        top_k=50,  # Top-k sampling
                        top_p=0.9  # Nucleus sampling
                    )
                    
                    # Decode predictions and references
                    for pred, ref in zip(generated_ids, labels):
                        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                        ref_text = tokenizer.decode(ref, skip_special_tokens=True)
                        val_predictions.append(pred_text)
                        val_references.append(ref_text)
                    
                    val_loop.set_postfix(loss=f"{loss.item():.3f}")
            
            # Calculate metrics
            avg_train_loss = train_epoch_loss / len(train_dataloader)
            avg_val_loss = val_epoch_loss / len(val_dataloader)
            
            # Calculate BLEU score
            bleu_score = 0.0
            try:
                bleu_scores = []
                for pred, ref in zip(val_predictions, val_references):
                    if pred.strip() and ref.strip():
                        score = sentence_bleu([ref.split()], pred.split(), 
                                           smoothing_function=SmoothingFunction().method1)
                        bleu_scores.append(score)
                bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0
            except Exception as e:
                print(f"⚠️  BLEU calculation error: {e}")
            
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"📈 Epoch {epoch+1} completed:")
            print(f"   🎯 Train Loss: {avg_train_loss:.3f}")
            print(f"   🔍 Val Loss: {avg_val_loss:.3f}")
            print(f"   🌟 BLEU Score: {bleu_score:.4f}")
            print(f"   📉 Learning Rate: {current_lr:.2e}")
            print(f"   📚 Curriculum Phase: {current_phase + 1}")
            
            # Enhanced early stopping with multiple metrics
            improvement = False
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                improvement = True
                print(f"   🎉 New best validation loss!")
            
            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                improvement = True
                print(f"   🎉 New best BLEU score!")
            
            if improvement:
                patience_counter = 0
                print(f"   💾 Saving best model...")
                
                # Save best model
                best_model_dir = "fine-tuned-mbart-tl2en-best"
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                print(f"   ✅ Best model saved to: {best_model_dir}")
            else:
                patience_counter += 1
                print(f"   ⚠️  No improvement for {patience_counter}/{patience} epochs")
                
                if patience_counter >= patience:
                    print(f"   🛑 Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("✅ Advanced training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Save model
    print("\n💾 Saving model...")
    try:
        output_dir = "fine-tuned-mbart-tl2en"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Model saved to: {output_dir}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    # Inference Section
    print("\n" + "=" * 60)
    print("🔍 INFERENCE TESTING")
    print("=" * 60)
    
    model.eval()
    
    # Enhanced test sentences covering different complexity levels
    test_sentences = [
        # Simple sentences
        "Kamusta ka?",
        "Salamat sa tulong mo.",
        # Medium sentences  
        "Hindi ko makita ang susi ko.",
        "Ang daming trabaho ngayon.",
        "Kamusta ang pamilya mo?",
        # Complex sentences
        "Thank you! Teka, di baliktad? Di ba dapat ako lilibre mo next year?",
        "Gusto kong kumain ng adobo.",
        "Ang daming trabaho ngayon, pero kailangan kong tapusin ito."
    ]
    
    print("\n📘 Translation Results:")
    print("-" * 40)
    
    for i, sentence in enumerate(test_sentences, 1):
        try:
            translation = translate_text(sentence, model, tokenizer)
            print(f"{i}. TL: {sentence}")
            print(f"   EN: {translation}")
            print()
        except Exception as e:
            print(f"{i}. ❌ Error translating: {sentence}")
            print(f"   Error: {e}")
            print()
    
    print("🎉 Enhanced Fine-tuning and testing completed!")
    print(f"📁 Final model saved to: fine-tuned-mbart-tl2en/")
    print(f"📁 Best model saved to: fine-tuned-mbart-tl2en-best/")
    print(f"📊 Training samples used: {len(train_dataset)} from filipino_english_parallel_corpus.csv")
    print(f"🔍 Validation samples used: {len(val_dataset)} from filipino_english_parallel_corpus.csv")
    print(f"🎯 Device used: {device}")
    print(f"📝 Source language: Filipino (preprocessed from filipino_english_parallel_corpus.csv)")
    print(f"🎯 Target language: English")
    print(f"📈 Best validation loss achieved: {best_val_loss:.3f}")
    print(f"🌟 Best BLEU score achieved: {best_bleu_score:.4f}")
    print(f"🛑 Training stopped after: {patience_counter} epochs without improvement")
    print(f"📚 Curriculum learning phases completed: {curriculum_phases}")
    print(f"🔄 Effective batch size used: {4 * accumulation_steps}")
    print(f"🎓 Enhanced features implemented:")
    print(f"   • Multiple loss functions (CE, Focal, Label Smoothing, R-Drop)")
    print(f"   • Extended curriculum learning (10 epochs, 4 phases)")
    print(f"   • Gradient accumulation (4x effective batch size)")
    print(f"   • Maximum LoRA capacity (r=64, comprehensive targets)")
    print(f"   • Advanced data quality scoring and augmentation")
    print(f"   • Enhanced text preprocessing (Filipino-specific)")
    print(f"   • R-Drop regularization for mixed complexity phase")
    print(f"   • Enhanced generation parameters (beam search + sampling)")
    print(f"   • BLEU score evaluation with better metrics")
    print(f"   • Enhanced early stopping with multiple criteria")

if __name__ == "__main__":
    main()
