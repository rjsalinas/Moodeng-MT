# 🇵🇭 Filipino-to-English Translation Model Fine-tuning

This directory contains the converted Python script from `test1.ipynb` for fine-tuning an mBART-50 model for Filipino-to-English translation.

## 📁 Files

- **`test1.py`** - Main fine-tuning script (converted from Jupyter notebook)
- **`requirements_translation.txt`** - Required Python dependencies
- **`README_TRANSLATION.md`** - This documentation file

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_translation.txt
```

### 2. Run the Script
```bash
python test1.py
```

## 🔧 What the Script Does

### **Training Pipeline**
1. **Data Loading**: Loads Filipino-English translation pairs from CSV or creates sample data
2. **Model Setup**: Initializes mBART-50 with LoRA (Low-Rank Adaptation) for efficient training
3. **Training Loop**: Runs 5 epochs with mixed precision training
4. **Model Saving**: Saves the fine-tuned model to `fine-tuned-mbart-tl2en/`

### **Inference Testing**
- Tests the trained model on sample Filipino sentences
- Shows translation quality and results

## 📊 Dataset Format

The script expects a CSV file named `test.csv` with two columns:
- **`src`**: Filipino text (source language)
- **`tgt`**: English text (target language)

**Example:**
```csv
src,tgt
"Kamusta ka?","How are you?"
"Salamat sa tulong mo.","Thank you for your help."
"Magandang umaga.","Good morning."
```

## 🎯 Key Features

### **LoRA Configuration**
- **Rank (r)**: 8 (low-rank adaptation)
- **Alpha**: 32 (scaling factor)
- **Target Modules**: q_proj, v_proj (query and value projections)
- **Dropout**: 0.1
- **Memory Efficient**: Only ~1% of parameters are trainable

### **Training Parameters**
- **Learning Rate**: 2e-4
- **Batch Size**: 4
- **Epochs**: 5
- **Max Length**: 128 tokens
- **Mixed Precision**: Enabled for faster training

### **Device Support**
- **CUDA**: Automatic detection and usage
- **CPU**: Fallback support (slower but functional)

## 🔍 Sample Output

```
🚀 Starting Filipino-to-English Translation Model Fine-tuning
============================================================
📱 Using device: cuda
📚 Loading tokenizer...
✅ Tokenizer loaded successfully
📊 Loading dataset...
✅ Dataset loaded: 5 samples
📝 Sample data:
                src                    tgt
0        Kamusta ka?            How are you?
1  Salamat sa tulong mo.  Thank you for your help.
2      Magandang umaga.           Good morning.
3         Paalam na.                Goodbye.
4    Gusto ko ng kape.         I want coffee.

🔥 Starting training...
Epoch 1/5: 100%|██████████| 2/2 [00:05<00:00,  2.50s/it, loss=9.234]
📈 Epoch 1 completed - Average Loss: 9.234
...

🔍 INFERENCE TESTING
============================================================
📘 Translation Results:
----------------------------------------
1. TL: Hindi ko makita ang susi ko.
   EN: I can't find my key.

2. TL: Ang daming trabaho ngayon.
   EN: There's so much work today.
```

## ⚠️ Important Notes

### **Hardware Requirements**
- **GPU**: Recommended for reasonable training speed
- **RAM**: At least 8GB (16GB+ recommended)
- **Storage**: ~2GB for model and dependencies

### **Dataset Requirements**
- **Minimum**: 5-10 sample pairs for testing
- **Recommended**: 100+ pairs for meaningful training
- **Quality**: Clean, parallel Filipino-English text

### **Model Output**
- **Directory**: `fine-tuned-mbart-tl2en/`
- **Contents**: Model weights, configuration, and tokenizer
- **Size**: ~4.5MB (LoRA adapters only)

## 🛠️ Customization

### **Modify Training Parameters**
```python
# In test1.py, adjust these values:
lora_config = LoraConfig(
    r=16,           # Increase rank for more capacity
    lora_alpha=64,  # Adjust scaling
    lora_dropout=0.2,  # Change dropout
    # ...
)

# Training parameters
for epoch in range(10):  # More epochs
    # ...
```

### **Add Custom Dataset**
1. Create `test.csv` with your Filipino-English pairs
2. Ensure proper CSV format (src, tgt columns)
3. Run the script

### **Modify Test Sentences**
```python
test_sentences = [
    "Your Filipino text here",
    "Another sentence",
    # Add more test cases
]
```

## 🚨 Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   - Reduce batch size (change `batch_size=2`)
   - Reduce max length (change `max_len=64`)

2. **Import Errors**
   - Install requirements: `pip install -r requirements_translation.txt`
   - Check PyTorch version compatibility

3. **Poor Translation Quality**
   - Increase training epochs
   - Improve dataset quality
   - Adjust LoRA parameters

4. **Slow Training on CPU**
   - This is expected - consider using GPU
   - Reduce dataset size for testing

## 📚 Technical Details

### **Model Architecture**
- **Base Model**: facebook/mbart-large-50-many-to-many-mmt
- **Parameters**: ~610M total, ~6M trainable (LoRA)
- **Languages**: 50+ languages including Filipino (tl_XX)

### **LoRA Benefits**
- **Memory Efficient**: Only train small adapter layers
- **Fast Training**: Reduced parameter count
- **Easy Deployment**: Small model size
- **Maintains Quality**: Preserves base model capabilities

## 🤝 Contributing

To improve the translation model:
1. **Better Dataset**: Add more Filipino-English pairs
2. **Hyperparameter Tuning**: Experiment with LoRA settings
3. **Data Augmentation**: Create variations of existing pairs
4. **Evaluation Metrics**: Add BLEU, ROUGE, or other metrics

## 📄 License

This script is for research and educational purposes. The mBART model is subject to Facebook's license terms.

---

**🎯 Ready to train your Filipino-to-English translation model!** 🎯
