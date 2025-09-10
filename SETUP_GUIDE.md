# 🚀 Setup Guide for Filipino-to-English Translation Project

This guide will help you set up the complete environment for the Filipino-to-English translation project using mBART-50 and LoRA.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **Git**: For cloning the repository
- **At least 8GB RAM** (16GB recommended)
- **At least 10GB free disk space**

## 🔧 Virtual Environment Setup

### Option 1: Using venv (Recommended)

#### Create Virtual Environment
```bash
# Windows
python -m venv moodeng_env

# Linux/Mac
python3 -m venv moodeng_env
```

#### Activate Virtual Environment
```bash
# Windows (Command Prompt)
moodeng_env\Scripts\activate

# Windows (PowerShell)
.\moodeng_env\Scripts\Activate.ps1

# Linux/Mac
source moodeng_env/bin/activate
```

#### Deactivate Virtual Environment
```bash
deactivate
```

### Option 2: Using conda

#### Create conda environment
```bash
conda create -n moodeng_env python=3.9
```

#### Activate conda environment
```bash
conda activate moodeng_env
```

#### Deactivate conda environment
```bash
conda deactivate
```

## 📦 Package Installation

### 1. Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install All Requirements
```bash
pip install -r requirements.txt
```

### 3. Install spaCy Language Models
```bash
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

### 4. Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🧪 Verify Installation

Run the environment test script:
```bash
python test_environment.py
```

You should see all packages marked with ✅ if installation was successful.

## 🚀 Quick Start

### 1. Activate Environment
```bash
# Windows
.\moodeng_env\Scripts\activate

# Linux/Mac
source moodeng_env/bin/activate
```

### 2. Run Training
```bash
python model_training_enhanced.py
```

### 3. Test Translation
```bash
python translate_with_model.py
```

### 4. Generate Documentation
```bash
python generate_thesis_docs.py
```

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. Package Version Conflicts
```bash
# Clean install
pip uninstall -y torch transformers peft
pip install -r requirements.txt
```

#### 3. Memory Issues
- Reduce batch size in `model_training_enhanced.py`
- Use gradient accumulation
- Enable mixed precision training

#### 4. Virtual Environment Issues

**Windows PowerShell Execution Policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Permission Denied (Linux/Mac):**
```bash
chmod +x moodeng_env/bin/activate
```

## 📁 Project Structure

```
Moodeng-MT/
├── moodeng_env/              # Virtual environment
├── model_training_enhanced.py # Main training script
├── translate_with_model.py   # Translation script
├── generate_thesis_docs.py   # Document generation
├── requirements.txt          # Package dependencies
├── SETUP_GUIDE.md           # This file
├── test_environment.py      # Environment test
├── data/                    # Dataset files
├── models/                  # Trained models
├── logs/                    # Training logs
└── docs/                    # Generated documentation
```

## 🎯 Environment Variables (Optional)

Create a `.env` file for custom configurations:
```bash
# .env
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [CalamanCy Documentation](https://github.com/ljvmiranda921/calamancy)

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your Python and CUDA versions
3. Ensure all packages are installed correctly
4. Check the logs in the `logs/` directory

---

**Happy Training! 🎉**

