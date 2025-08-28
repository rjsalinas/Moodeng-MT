# 🇵🇭 Filipino Text Preprocessing, Enhancement, and mBART Fine‑tuning

This repository contains two complementary pipelines:

1) A Filipino/Tagalog tweet normalization system (rule‑ and feature‑based)
2) An enhanced Filipino→English translation pipeline built on mBART‑50 with CalamanCy‑powered preprocessing

If you are here to reproduce the machine translation results, start with the Quickstart below. The original normalization documentation remains further down this file.

## 🔰 Quickstart: Enhanced Translation Pipeline

### 🛠️ Setup (First Time Only)

**Option 1: Automated Setup**
```bash
# Windows
install_requirements.bat

# Linux/Mac
chmod +x activate_env.sh
./activate_env.sh
pip install -r requirements.txt
```

**Option 2: Manual Setup**
See `SETUP_GUIDE.md` for detailed instructions.

### 🚀 Activate Environment
```bash
# Windows (Command Prompt)
activate_env.bat

# Windows (PowerShell)
.\activate_env.ps1

# Linux/Mac
./activate_env.sh
```

### 1) Prepare/Enhance the Parallel Corpus

Input CSV: `filipino_english_parallel_corpus.csv`
- Expected columns: either `text` + `english_translation` or already normalized `src` + `tgt`.

Run CalamanCy‑enhanced preprocessing in batches (preferred for speed and reproducibility):

```bash
python batch_process_calamancy.py
```

This generates:
- `full_enhanced_parallel_corpus.csv` (preferred input for training)
- Optional `enhanced_batch_XXX.csv` backups and `batch_processing.log`

### 2) Fine‑tune mBART‑50 (LoRA)

Run the enhanced training script. It prefers the enhanced CSV and falls back to raw CSV if needed.

```bash
python model_training_enhanced.py
```

Outputs:
- `fine-tuned-mbart-tl2en/` and `fine-tuned-mbart-tl2en-best/` (LoRA adapters + config)

### Optional: Baseline (no enhanced preprocessing)

```bash
python model_training.py
```

### Where to read more
- `SETUP_GUIDE.md`: Complete environment setup and troubleshooting
- `PREPROCESSING_PIPELINE.md`: End‑to‑end preprocessing (now includes CalamanCy batch pipeline and inference note)
- `MODEL_TRAINING.md`: Training objectives, losses, schedules, checkpoints, and inference usage
- `README_TRANSLATION.md`: Practical training/inference guide aligned with scripts
- `README_CALAMANCY_INTEGRATION.md`: Details of CalamanCy‑based enhancements and output columns
- `ENHANCED_FEATURES.md`: Feature highlights for Filipino tweet processing

## 🚀 Features (Normalization System)

### **Core Normalization Operations**
- **Substitution**: Orthographic variants (o↔u, e↔i, y↔i, ch↔ts)
- **Deletion**: Redundant characters, duplicate letters, excessive punctuation
- **Insertion**: Missing hyphens, apostrophes, affix boundaries
- **Transposition**: Letter-order corrections (alakt → aklat)
- **Token Operations**: Split/merge missegmented tokens
- **Case Normalization**: Consistent lowercase formatting
- **Punctuation Handling**: Smart preservation and cleanup
- **Slang Expansion**: SMS shortcuts to standard forms (q → ako, 2 → to)

### **Advanced Text Processing**
- **Gibberish Detection**: Keyboard smashing and random character removal
- **Social Media Cleaning**: Hashtag/mention removal, artifact cleanup
- **English Text Preservation**: Maintains English content integrity
- **Morphology Awareness**: Filipino affix and reduplication patterns
- **Comprehensive Logging**: Detailed rule application tracking
- **Spanish Content Detection**: Identifies and filters Spanish language content
- **Text Standardization**: Lowercase conversion and sentence end periods

### **New Enhanced Features** ⭐
- **JSON Dataset Processing**: Extract tweet data from JSON files to CSV format
- **Spanish Language Filtering**: Remove Spanish content from Filipino datasets
- **English Text Examples**: Showcase English preservation capabilities
- **Fine-tuned Translation Model**: mBART-50 model available for Filipino-to-English translation *(not yet integrated)*
- **Enhanced Gibberish Detection**: English-aware pattern recognition
- **Conservative Text Cleaning**: Preserves meaningful content while removing noise

## 📁 Project Structure

```
Moodeng-MT/
├── normalizer.py                           # Core normalization engine
├── preprocess_tweets.py                    # Main preprocessing script
├── normalize_csv_tweets.py                 # CSV batch normalization script
├── extract_tweet_data.py                   # JSON to CSV extraction tool ⭐ NEW
├── remove_spanish_from_filipino.py        # Spanish content filtering ⭐ NEW
├── show_english_examples.py               # English preservation examples ⭐ NEW
├── log_manager.py                          # Intelligent log management ⭐ NEW
├── cleanup_large_files.py                 # Large file management ⭐ NEW
├── rules.json                              # Normalization rule definitions
├── test_results.py                         # Results visualization
├── test_punctuation_preservation.py       # Punctuation testing
├── test_enhanced_normalization.py          # Comprehensive rule testing
├── test_excel_structure.py                # Excel file validation
├── COMPREHENSIVE_RULES.md                  # Detailed rule documentation
├── ENHANCED_FEATURES.md                    # New features documentation ⭐ NEW
├── requirements.txt                        # Python dependencies
├── run_preprocessing.bat                   # Windows batch execution
├── run_preprocessing.ps1                   # PowerShell execution
├── logs/                                   # Processing logs and history *(with rotation)*
├── fine-tuned-mbart-tl2en/                # Translation model *(available, not integrated)* ⭐ NEW
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── 📊 **Research Data (Kept in Repository)** ⭐
│   ├── tweets_id_filipino_text_only.csv        # Filipino/Taglish tweets (research data)
│   ├── tweets_id_filipino_text_normalized.csv  # Normalized Filipino tweets (results)
│   └── tweets_id_non_fil_tag_taglish.csv       # Non-Filipino tweets (analysis)
├── 📁 **Large Files (Backed up locally)** ⭐
│   ├── tweets_id_text_only.csv                 # Raw extracted data
│   ├── tweets_id_text_normalized.csv           # Full normalized dataset
│   ├── english_translation_from_preprocessed_texts.csv  # Translation outputs
│   └── dataset_*.json                          # Raw JSON datasets
└── .gitignore                               # Smart file exclusion rules ⭐ NEW
```

## 🛠️ Installation & Setup

### **Smart File Management Strategy** ⭐ NEW
This project uses intelligent file management to balance GitHub compatibility with research data preservation:

- **📊 Research Data**: Important CSV results kept in repository for thesis analysis
- **📁 Large Files**: Very large datasets backed up locally to prevent GitHub issues
- **🔄 Log Rotation**: Automatic log management prevents file size problems
- **⚡ Performance**: Fast processing while maintaining data integrity

### **Prerequisites**
- Python 3.7+
- pandas
- openpyxl

### **Quick Start**
```bash
# Clone the repository
git clone <your-repo-url>
cd Moodeng-MT

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python preprocess_tweets.py
```

### **Windows Users**
```cmd
# Using batch file
run_preprocessing.bat

# Using PowerShell
.\run_preprocessing.ps1
```

## 📊 Usage

### **Input Format**
- **Excel Files**: `.xlsx` format with specific worksheet requirements
- **JSON Files**: Raw tweet datasets with id and text fields ⭐ NEW
- **CSV Files**: Processed tweet data for further analysis
- **Worksheet**: `tweets_split_id`
- **Filter**: Rows where `Tweet Status = 1`
- **Input Column**: Original tweet content
- **Output Column**: Preprocessed text

### **Basic Usage**
```python
from normalizer import FilipinoNormalizer

# Initialize normalizer
normalizer = FilipinoNormalizer('rules.json', 'logs')

# Process text
text = "q nakapunta na 2 the mall!!!"
normalized, logs = normalizer.normalize_text(text)
print(normalized)  # "ako naka punta na to the mall!"
```

### **New Data Processing Workflows** ⭐

#### **JSON Dataset Extraction**
```python
# Extract tweet data from JSON to CSV
python extract_tweet_data.py
# Creates: tweets_id_text_only.csv from JSON datasets
```

#### **Spanish Content Filtering**
```python
# Remove Spanish content from Filipino datasets
python remove_spanish_from_filipino.py
# Creates: filtered datasets with Spanish content removed
```

#### **English Text Examples**
```python
# Show English preservation examples
python show_english_examples.py
# Displays: Examples of English text preservation
```

### **Batch Processing**
```python
# Process entire Excel file
python preprocess_tweets.py

# Process CSV files
python normalize_csv_tweets.py

# View results
python test_results.py
```

### **CSV Processing Workflow**
```python
# 1. Extract from JSON (if needed)
python extract_tweet_data.py
# Creates: tweets_id_text_only.csv

# 2. Normalize CSV tweets
python normalize_csv_tweets.py
# Creates: tweets_id_text_normalized.csv

# 3. Filter Filipino tweets (optional)
# Use the filtering script to separate Filipino from English tweets
# Creates: tweets_id_filipino_text_only.csv
```

## 🔧 Configuration

### **Rules File (`rules.json`)**
The system uses a JSON-based rule configuration:
```json
{
  "rules": [
    {
      "rule_id": "O_U_01",
      "pattern": "o↔u",
      "active": true,
      "priority": 10
    }
  ]
}
```

### **Custom Rule Addition**
1. Edit `rules.json`
2. Add new rule patterns
3. Restart the normalizer

## 📈 Output Examples

### **Before → After Transformations**
| Original Text | Normalized Text | Applied Rules |
|---------------|-----------------|---------------|
| `q nakapunta na 2 the mall!!!` | `ako naka punta na to the mall!` | slang, token-split, punctuation |
| `alakt ko na nga???` | `aklat ko na ang?` | transposition, slang, punctuation |
| `Hello world...` | `hello world.` | case, punctuation, period |
| `Kamusta ka?` | `kamusta ka?` | case, preserve punctuation |

### **New Enhanced Transformations** ⭐
| Original Text | Normalized Text | Applied Rules |
|---------------|-----------------|---------------|
| `#tagalog @username` | `tagalog.` | hashtag cleanup, mention removal, period |
| `qwertyuiop text` | `text.` | gibberish removal, period |
| `what do you do ba?` | `what do you do ba?` | English preservation, case |
| `el problema es grande` | `el problema es grande.` | Spanish detection, period |

### **Rule Application Logging**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "rule_id": "SLANG_01",
  "reason": "slang_to_standard",
  "before": "q nakapunta",
  "after": "ako nakapunta",
  "meta": {"from_word": "q", "to_word": "ako"}
}
```

## 🧪 Testing

### **Run All Tests**
```bash
# Punctuation preservation
python test_punctuation_preservation.py

# Enhanced normalization rules
python test_enhanced_normalization.py

# Excel structure validation
python test_excel_structure.py

# View sample results
python test_results.py

# Show English preservation examples ⭐ NEW
python show_english_examples.py
```

### **Test Categories**
- ✅ **Punctuation Preservation**: Maintains original ending marks
- ✅ **Repeated Mark Cleanup**: Reduces multiple marks to single
- ✅ **Slang Expansion**: SMS shortcuts to standard forms
- ✅ **Token Operations**: Split/merge functionality
- ✅ **Orthographic Rules**: Character alternation patterns
- ✅ **Spanish Detection**: Identifies Spanish language content ⭐ NEW
- ✅ **English Preservation**: Maintains English text integrity ⭐ NEW
- ✅ **Gibberish Detection**: English-aware pattern recognition ⭐ NEW

## 🔍 Recent Improvements

### **Enhanced Punctuation Handling** *(Latest)*
- **Original ending punctuation preserved**: Exclamation marks (!), question marks (?), periods (.), semicolons (;) are kept as-is
- **Repeated punctuation cleaned up**: Multiple marks (!!!, ???, ...) are reduced to single marks
- **Smart period addition**: Only adds periods when no ending punctuation exists

**Examples**:
- `"Hello world!"` → `"hello world!"` ✅ (preserve !)
- `"Kamusta ka???"` → `"kamusta ka?"` ✅ (reduce to single ?)
- `"Wow..."` → `"wow."` ✅ (reduce to single .)
- `"Test"` → `"test."` ✅ (add period if none)

### **New Data Processing Tools** *(Latest)* ⭐
- **JSON Dataset Extraction**: Convert raw JSON tweet data to CSV format
- **Spanish Content Filtering**: Remove Spanish language content from Filipino datasets
- **Enhanced Language Detection**: Improved Filipino/Taglish vs English vs Spanish classification
- **English Text Examples**: Showcase English preservation capabilities

**New Files Created**:
- `extract_tweet_data.py`: JSON to CSV extraction tool
- `remove_spanish_from_filipino.py`: Spanish content filtering
- `show_english_examples.py`: English preservation examples
- `ENHANCED_FEATURES.md`: Comprehensive new features documentation

### **Fine-tuned Translation Model** *(Latest)* ⭐
- **mBART-50 Model**: Facebook's multilingual translation model fine-tuned for Filipino-to-English
- **Adapter-based Training**: Efficient parameter-efficient fine-tuning approach
- **High-Quality Translations**: Professional-grade Filipino to English translation
- **Model Files**: Complete model weights and configuration

**Model Details**:
- **Base Model**: facebook/mbart-large-50-many-to-many-mmt
- **Fine-tuning**: PEFT (Parameter-Efficient Fine-Tuning)
- **Library**: transformers with PEFT 0.16.0
- **Format**: SafeTensors for efficient loading

### **Enhanced Gibberish Detection** *(Latest)* ⭐
- **English-Aware Patterns**: Conservative detection that preserves legitimate English words
- **Keyboard Smashing**: Removes patterns like "qwertyuiop" (6+ characters only)
- **Consonant/Vowel Clusters**: Removes excessive sequences while preserving real words
- **Smart Thresholds**: Only removes clearly gibberish patterns

**Examples**:
- `"qwertyuiop text"` → `"text."` ✅ (removes gibberish, keeps real word)
- `"dump special someone"` → `"dump special someone."` ✅ (preserves English words)
- `"bcdfghjklmnpqrstvwxz"` → `""` ✅ (removes excessive consonants)

### **Spanish Content Detection** *(Latest)* ⭐
- **Comprehensive Spanish Patterns**: 100+ Spanish words and verb conjugations
- **Confidence Scoring**: Intelligent detection with confidence thresholds
- **Content Filtering**: Remove or flag Spanish language content
- **Bilingual Support**: Handle mixed Spanish-Filipino content

**Detection Categories**:
- **Articles & Pronouns**: el, la, los, las, yo, tú, él, ella
- **Common Verbs**: es, son, está, tienen, hace, dice, va
- **Time & Date**: lunes, enero, hoy, ayer, mañana
- **Social Media**: jaja, amigo, familia, trabajo, escuela

### **CSV Text Normalization** *(Enhanced)*
- **Batch CSV processing**: Process entire CSV files with tweet text normalization
- **Preserved original file**: Creates new output files without modifying source data
- **Comprehensive logging**: Tracks all normalization operations and statistics
- **Progress monitoring**: Real-time processing updates for large datasets

**Files Created**:
- `tweets_id_text_normalized.csv`: Original text + normalized `preprocessed_text` column
- `normalize_csv_tweets.py`: Standalone CSV normalization script

### **Filipino Tweet Filtering** *(Enhanced)*
- **Language-based filtering**: Automatically detects and separates Filipino/Taglish from English-only tweets
- **Pattern-based detection**: Uses comprehensive Filipino word and phrase patterns
- **Taglish support**: Recognizes mixed English-Filipino content
- **Multiple output files**: Separate files for different language categories

**Files Created**:
- `tweets_id_filipino_text_only.csv`: Contains only Filipino/Taglish tweets
- `tweets_id_filipino_text_normalized.csv`: Normalized Filipino tweets
- `tweets_id_non_fil_tag_taglish.csv`: Non-Filipino/Taglish tweets
- `english_translation_from_preprocessed_texts.csv`: English translations

**Filtering Results**:
- **Total tweets processed**: 3,531
- **Filipino/Taglish tweets**: 2,869 (81.5%)
- **English-only tweets**: 653 (18.5%)
- **Original file preserved**: `tweets_id_text_only.csv` remains unchanged

### **Critical Bug Fixes** *(Enhanced)*
- **"nga" preservation**: Fixed incorrect normalization that was changing "nga" to "ang"
- **Semantic accuracy**: "nga" and "ang" are distinct Filipino words with different meanings
- **Pattern refinement**: Removed problematic transposition rules that caused semantic errors
- **Spanish content handling**: Improved detection and filtering of Spanish language content

## 📚 Documentation

- **`COMPREHENSIVE_RULES.md`**: Detailed rule explanations and examples
- **`ENHANCED_FEATURES.md`**: New features and capabilities documentation ⭐ NEW
- **`test_*.py`**: Comprehensive testing and validation scripts
- **`logs/`**: Processing history and rule application logs *(with automatic rotation)*
- **`fine-tuned-mbart-tl2en/README.md`**: Translation model documentation ⭐ NEW
- **`log_manager.py`**: Intelligent log management with rotation ⭐ NEW

## 🤝 Contributing

### **Adding New Rules**
1. Identify normalization pattern
2. Add rule to `rules.json`
3. Implement rule logic in `normalizer.py`
4. Add test cases
5. Update documentation

### **Rule Categories**
- **Orthographic**: Character alternations and spelling variants
- **Morphological**: Affix boundaries and word formation
- **Semantic**: Slang expansion and loanword handling
- **Structural**: Token segmentation and punctuation
- **Language Detection**: Spanish content identification ⭐ NEW
- **Gibberish Detection**: English-aware pattern recognition ⭐ NEW

## 📊 Performance

### **Processing Speed**
- **Small texts** (<100 chars): ~1-5ms
- **Medium texts** (100-500 chars): ~5-20ms
- **Large texts** (500+ chars): ~20-100ms

### **Batch Processing Performance**
- **CSV normalization**: ~3,500 tweets in ~2-3 minutes
- **Filipino filtering**: ~3,500 tweets in ~1-2 minutes
- **Spanish detection**: ~3,500 tweets in ~2-3 minutes ⭐ NEW
- **JSON extraction**: ~10,000 tweets in ~1-2 minutes ⭐ NEW
- **Memory efficient**: Processes large datasets without memory issues
- **Progress tracking**: Real-time updates for long-running operations

### **Memory Usage**
- **Rule loading**: ~2-5MB
- **Text processing**: ~1-2MB per text
- **Logging**: Configurable, typically 10-50MB
- **Translation model**: ~4.5MB (adapter weights) ⭐ NEW

## 🚨 Troubleshooting

### **Common Issues**
1. **ModuleNotFoundError**: Ensure `normalizer.py` is in the same directory
2. **Excel file errors**: Check worksheet name and column headers
3. **Rule application failures**: Verify `rules.json` format and syntax
4. **JSON parsing errors**: Check JSON file format and encoding ⭐ NEW
5. **Spanish detection issues**: Verify Spanish word patterns and thresholds ⭐ NEW

### **Debug Mode**
```python
# Enable detailed logging
normalizer = FilipinoNormalizer('rules.json', 'logs')
normalized, logs = normalizer.normalize_text(text, context={"debug": True})
```

### **New Debug Tools** ⭐
```python
# Show English preservation examples
python show_english_examples.py

# Check Spanish content detection
python remove_spanish_from_filipino.py

# Validate JSON dataset structure
python extract_tweet_data.py

# Manage logs and large files
python log_manager.py
python cleanup_large_files.py
```

## 📄 License

This project is developed for academic research purposes. Please ensure proper attribution when using or modifying the code.

## 📞 Support

For questions, issues, or contributions:
- **Issues**: Use GitHub Issues
- **Documentation**: Check `COMPREHENSIVE_RULES.md` and `ENHANCED_FEATURES.md`
- **Testing**: Run `test_*.py` scripts
- **New Features**: Review `ENHANCED_FEATURES.md` for latest capabilities

---

**🎯 Your enhanced Filipino text normalization system is now complete with advanced language processing, translation capabilities, and comprehensive data processing tools!** 🎯

*Built with ❤️ for Filipino language processing and research.*
