# Baseline Filipino-to-English Translation Training

This directory contains the baseline training system for Filipino-to-English translation, trained on the original `filipino_english_parallel_corpus.csv` without CalamanCy enhancements.

## 🎯 **Purpose**

The baseline system provides a comparison point to evaluate the effectiveness of CalamanCy-enhanced preprocessing. It uses the same architecture (mBART-50 + LoRA) but with simpler data preprocessing.

## 📁 **Files**

### **Core Training Scripts**
- **`model_training_baseline.py`** - Main baseline training script
- **`simple_translate_baseline.py`** - Translation script for baseline model
- **`compare_models.py`** - Compare enhanced vs baseline model performance

### **Testing & Documentation**
- **`test_baseline_script.py`** - Test script to verify setup
- **`README_BASELINE.md`** - This documentation file

## 🚀 **Quick Start**

### **1. Test Your Setup**
```bash
python test_baseline_script.py
```

### **2. Train the Baseline Model**
```bash
python model_training_baseline.py
```

### **3. Translate with Baseline Model**
```bash
python simple_translate_baseline.py "Kamusta ka?"
```

### **4. Compare Models**
```bash
python compare_models.py "Kamusta ka?"
python compare_models.py --test_samples 10
```

## 🔧 **Dataset Structure**

The baseline system uses `filipino_english_parallel_corpus.csv` with these columns:
- **`id`** - Unique identifier
- **`text`** - Original Filipino text
- **`preprocessed_text`** - **Source** (Filipino text after basic preprocessing)
- **`english_translation`** - **Target** (English translation)

**Key Difference**: Uses `preprocessed_text` → `english_translation` mapping instead of CalamanCy-enhanced features.

## 🏗️ **Architecture**

- **Base Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Target Languages**: Filipino (`tl_XX`) → English (`en_XX`)
- **Training**: 20 epochs with early stopping (patience=5)

## 📊 **Training Configuration**

```python
BATCH_SIZE = 2              # Reduced for VRAM efficiency
LEARNING_RATE = 5e-5        # Conservative learning rate
NUM_EPOCHS = 20             # Maximum training epochs
WARMUP_STEPS = 100          # Learning rate warmup
PATIENCE = 5                 # Early stopping patience
MAX_LENGTH = 128            # Maximum sequence length
```

## 🔍 **Model Outputs**

After training, the script creates:
- **`fine-tuned-mbart-tl2en-baseline-best/`** - Best model (lowest validation loss)
- **`fine-tuned-mbart-tl2en-baseline/`** - Final model (after all epochs)

## 📈 **Expected Performance**

**Baseline Model Characteristics**:
- **Pros**: Simpler, faster training, no external dependencies
- **Cons**: Lower translation quality, less linguistic sophistication
- **Use Case**: Baseline comparison, quick prototyping, resource-constrained environments

**Enhanced Model Characteristics**:
- **Pros**: Higher translation quality, linguistic analysis, better handling of code-switching
- **Cons**: Slower training, CalamanCy dependency, more complex

## 🐛 **Troubleshooting**

### **Common Issues**

1. **CUDA Memory Errors**
   - Reduce `BATCH_SIZE` to 1
   - Use gradient accumulation (already implemented)
   - Monitor VRAM usage

2. **Device Mismatch Errors**
   - The script automatically handles CUDA device consistency
   - Check CUDA installation: `nvidia-smi`

3. **Import Errors**
   - Install requirements: `pip install torch transformers peft pandas tqdm nltk`
   - Activate virtual environment: `.\activate_env.ps1`

### **Performance Tips**

1. **For Training**:
   - Use CUDA if available (8GB+ VRAM recommended)
   - Monitor training logs in `training_logs/` directory
   - Early stopping prevents overfitting

2. **For Inference**:
   - Baseline model loads faster than enhanced model
   - Use `simple_translate_baseline.py` for quick translations
   - Use `compare_models.py` for quality comparison

## 📝 **Training Logs**

Training progress is logged to:
- **Console**: Real-time progress with emojis and metrics
- **File**: `training_logs/baseline_training_YYYYMMDD_HHMMSS.log`
- **Metrics**: Loss, BLEU scores, validation performance

## 🔄 **Comparison with Enhanced System**

| Aspect | Baseline | Enhanced |
|--------|----------|----------|
| **Dataset** | `filipino_english_parallel_corpus.csv` | `full_enhanced_parallel_corpus.csv` |
| **Preprocessing** | Basic text cleaning | CalamanCy + linguistic analysis |
| **Training Time** | Faster | Slower (more complex) |
| **Translation Quality** | Lower | Higher |
| **Resource Usage** | Lower | Higher |
| **Dependencies** | Minimal | CalamanCy + additional packages |

## 💡 **Use Cases**

### **Baseline Model Best For**:
- Quick prototyping and testing
- Resource-constrained environments
- Baseline performance comparison
- Simple translation tasks
- Educational purposes

### **Enhanced Model Best For**:
- Production translation systems
- High-quality requirements
- Research and evaluation
- Complex Filipino text (Taglish, slang, etc.)

## 🎓 **Academic Use**

This baseline system is perfect for:
- **Thesis Research**: Compare preprocessing approaches
- **Performance Analysis**: Measure CalamanCy impact
- **Methodology Comparison**: Evaluate enhancement effectiveness
- **Baseline Establishment**: Set performance benchmarks

## 📚 **References**

- **mBART-50**: [Multilingual Denoising Pre-training](https://arxiv.org/abs/2001.08210)
- **LoRA**: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **Filipino NLP**: [CalamanCy Documentation](https://github.com/ljvmiranda921/calamanCy)

---

**Note**: This baseline system provides a foundation for understanding the impact of advanced preprocessing techniques on Filipino machine translation quality.
