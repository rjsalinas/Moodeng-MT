# 🚀 CalamanCy Integration for Filipino Text Preprocessing

This document explains how to integrate CalamanCy (Tagalog NLP library) with your Filipino-to-English translation pipeline to achieve better preprocessing and model performance.

## 📋 Table of Contents

1. [What is CalamanCy?](#what-is-calamancy)
2. [Benefits of Integration](#benefits-of-integration)
3. [Installation Guide](#installation-guide)
4. [File Structure](#file-structure)
5. [Usage Examples](#usage-examples)
6. [Integration with Training](#integration-with-training)
7. [Performance Comparison](#performance-comparison)
8. [Troubleshooting](#troubleshooting)

## 🎯 What is CalamanCy?

**CalamanCy** is a specialized NLP library for Tagalog (Filipino) built on top of spaCy. It provides:

- **Tagalog Language Model**: Pre-trained on Filipino text
- **Part-of-Speech Tagging**: Accurate Filipino grammar analysis
- **Dependency Parsing**: Understanding sentence structure
- **Named Entity Recognition**: Identifying people, places, organizations
- **Morphological Analysis**: Word form variations and patterns

## 🚀 Benefits of Integration

### **Before CalamanCy (Basic Preprocessing):**
- ❌ Simple word counting for complexity
- ❌ Basic regex-based text cleaning
- ❌ Limited data augmentation
- ❌ No linguistic validation
- ❌ Generic tokenization

### **After CalamanCy (Enhanced Preprocessing):**
- ✅ **Linguistic Complexity**: POS-based, dependency-based, morphological scoring
- ✅ **Advanced Augmentation**: Verb form variations, noun pluralization
- ✅ **Quality Validation**: Grammar structure, entity preservation, Filipino indicators
- ✅ **Filipino-Aware Tokenization**: Proper handling of contractions and word boundaries
- ✅ **Smart Filtering**: Remove low-quality translation pairs

## 📦 Installation Guide

### **Step 1: Install CalamanCy Dependencies**

```bash
# Install the enhanced requirements
pip install -r requirements_calamancy.txt

# Download the Tagalog language model
python -m spacy download tl_core_news_sm
```

### **Step 2: Verify Installation**

```bash
# Test if CalamanCy works
python -c "import calamancy; print('✅ CalamanCy installed successfully!')"

# Test if spaCy with Tagalog works
python -c "import spacy; nlp = spacy.load('tl_core_news_sm'); print('✅ Tagalog model loaded!')"
```

### **Step 3: Test Integration**

```bash
# Run the integration test
python test_calamancy_integration.py
```

## 📁 File Structure

```
Moodeng-MT/
├── enhanced_preprocessing.py          # Core CalamanCy integration
├── test_calamancy_integration.py     # Test script for integration
├── model_training_enhanced.py        # Enhanced training script
├── requirements_calamancy.txt         # CalamanCy dependencies
├── README_CALAMANCY_INTEGRATION.md   # This file
└── model_training.py                 # Original training script
```

## 🔧 Usage Examples

### **Basic Usage**

```python
from enhanced_preprocessing import enhance_filipino_dataset

# Load your dataset
df = pd.read_csv("filipino_english_parallel_corpus.csv")
df = df.rename(columns={
    "preprocessed_text": "src",
    "english_translation": "tgt"
})

# Apply CalamanCy enhancements
enhanced_df = enhance_filipino_dataset(df)

print(f"Original: {len(df)} samples")
print(f"Enhanced: {len(enhanced_df)} samples")
```

### **Advanced Usage**

```python
from enhanced_preprocessing import EnhancedFilipinoPreprocessor

# Initialize preprocessor
preprocessor = EnhancedFilipinoPreprocessor()

# Calculate linguistic complexity
complexity = preprocessor.enhanced_complexity_calculation("Magandang umaga sa inyong lahat.")
print(f"Complexity Score: {complexity['total_score']}")
print(f"POS Complexity: {complexity['pos_complexity']}")
print(f"Dependency Complexity: {complexity['dependency_complexity']}")

# Validate translation quality
quality = preprocessor.validate_filipino_quality(
    "Gusto ko ng kape",
    "I want coffee"
)
print(f"Quality Score: {quality['score']}")
print(f"Issues: {quality['issues']}")

# Apply Filipino-aware tokenization
tokenized = preprocessor.filipino_aware_tokenization("Di'ba maganda ang panahon?")
print(f"Tokenized: {tokenized}")
```

## 🎓 Integration with Training

### **Option 1: Use Enhanced Training Script**

```bash
# Run the enhanced training script
python model_training_enhanced.py
```

**Features:**
- ✅ Automatic CalamanCy integration
- ✅ Enhanced complexity calculation
- ✅ Quality-based filtering
- ✅ Advanced data augmentation
- ✅ Fallback to basic preprocessing if CalamanCy fails

### **Option 2: Integrate with Existing Script**

```python
# In your existing model_training.py
try:
    from enhanced_preprocessing import enhance_filipino_dataset
    CALAMANCY_AVAILABLE = True
except ImportError:
    CALAMANCY_AVAILABLE = False

# Use enhanced preprocessing if available
if CALAMANCY_AVAILABLE:
    df = enhance_filipino_dataset(df)
    print("✅ CalamanCy enhancements applied")
else:
    print("⚠️  Using basic preprocessing")
```

## 📊 Performance Comparison

### **Data Quality Metrics**

| Metric | Basic Preprocessing | CalamanCy Enhanced |
|--------|-------------------|-------------------|
| **Complexity Scoring** | Word count + punctuation | Linguistic + structural |
| **Data Augmentation** | Synonym replacement | Morphological variations |
| **Quality Validation** | Length + similarity | Grammar + entities |
| **Tokenization** | Generic splitting | Filipino-aware |
| **Filtering** | Basic thresholds | Linguistic quality |

### **Expected Improvements**

- **Better Complexity Distribution**: More accurate curriculum learning
- **Higher Quality Data**: Remove poor translation pairs
- **More Training Varieties**: Morphological augmentations
- **Improved Tokenization**: Better Filipino word boundaries
- **Linguistic Validation**: Ensure proper Filipino structure

## 🛠️ Troubleshooting

### **Common Issues**

#### **1. CalamanCy Installation Failed**

```bash
# Try alternative installation
pip install git+https://github.com/calamanCy/calamanCy.git

# Or install specific versions
pip install spacy==3.7.2
pip install calamancy==0.1.0
```

#### **2. Tagalog Model Not Found**

```bash
# Download manually
python -m spacy download tl_core_news_sm

# Verify installation
python -c "import spacy; print(spacy.util.get_package('tl_core_news_sm'))"
```

#### **3. Import Errors**

```bash
# Check if packages are installed
pip list | grep -E "(calamancy|spacy)"

# Reinstall if needed
pip uninstall calamancy spacy
pip install -r requirements_calamancy.txt
```

#### **4. Memory Issues**

```bash
# Use smaller batch sizes
# Reduce max_length in tokenization
# Process data in chunks
```

### **Fallback Strategy**

The enhanced preprocessing automatically falls back to basic preprocessing if CalamanCy fails:

```python
try:
    enhanced_df = enhance_filipino_dataset(df)
    print("✅ CalamanCy enhancement successful")
except Exception as e:
    print(f"⚠️  CalamanCy failed: {e}")
    print("🔄 Using basic preprocessing...")
    # Basic preprocessing logic here
```

## 🎯 Best Practices

### **1. Gradual Integration**

- Start with basic preprocessing
- Test CalamanCy on small datasets
- Gradually increase dataset size
- Monitor performance improvements

### **2. Quality Control**

- Validate enhanced data quality
- Compare with original preprocessing
- Monitor training metrics
- Adjust complexity thresholds

### **3. Performance Optimization**

- Cache linguistic analysis results
- Process data in batches
- Use appropriate batch sizes
- Monitor memory usage

## 🚀 Next Steps

### **Immediate Actions**

1. ✅ **Install CalamanCy**: `pip install -r requirements_calamancy.txt`
2. ✅ **Test Integration**: `python test_calamancy_integration.py`
3. ✅ **Run Enhanced Training**: `python model_training_enhanced.py`

### **Future Enhancements**

- **Custom Filipino Grammar Rules**: Add domain-specific patterns
- **Advanced Augmentation**: Implement more sophisticated variations
- **Quality Metrics**: Add more validation criteria
- **Performance Optimization**: Cache and batch processing
- **Integration with Other Tools**: Connect with existing preprocessing pipeline

## 📞 Support

If you encounter issues:

1. **Check the test script**: `python test_calamancy_integration.py`
2. **Verify dependencies**: `pip list | grep -E "(calamancy|spacy)"`
3. **Check error messages**: Look for specific import or loading errors
4. **Fallback to basic**: The system automatically falls back if needed

## 🎉 Conclusion

CalamanCy integration provides a significant upgrade to your Filipino text preprocessing pipeline. It offers:

- **Better Data Quality**: Linguistic validation and filtering
- **Enhanced Complexity**: Accurate scoring for curriculum learning
- **Advanced Augmentation**: Morphological variations for better training
- **Filipino-Aware Processing**: Proper handling of Tagalog-specific features

Start with the installation and testing, then gradually integrate it into your training pipeline for improved model performance! 🚀
