# Automated TweetTaglish Processor

## 🚀 **Complete Automation for 21,000+ Tweets**

This system automatically processes all tweets from your `twitter_links.txt` file and generates the three columns you need:

1. **Original Taglish Tweet** - Raw tweet extracted from Twitter
2. **Preprocessed Taglish Tweet** - Cleaned using your trained AI model
3. **English Translation** - Generated using free translation services

## 🎯 **What This Solves**

- **Manual processing bottleneck**: You manually processed 1,300 tweets, but have 21,000+ more
- **Time efficiency**: Automates the entire pipeline from extraction to translation
- **Consistency**: Uses your trained AI model to ensure consistent preprocessing
- **Scalability**: Handles thousands of tweets automatically with progress tracking

## 🏗️ **System Architecture**

```
twitter_links.txt → Tweet Extractor → AI Preprocessor → Translation Service → CSV Output
       ↓                ↓                ↓                ↓              ↓
   21,000+ links   Extract text    Apply your rules   Translate to EN   Final dataset
```

## 📋 **Prerequisites**

### 1. **Required Files**
- `twitter_links.txt` - Your file with 21,000+ Twitter links
- `Tweets/` directory - Contains your training data for AI model
- Trained AI preprocessing model (will be created if not exists)

### 2. **Dependencies**
```bash
pip install -r requirements_ai_preprocessor.txt
pip install aiohttp
```

### 3. **Optional: Twitter API Access**
- Set `X_BEARER_TOKEN` environment variable for better tweet extraction
- Without it, falls back to web scraping methods

### 4. **Optional: Apify Integration (Recommended for Speed)**
- **Sign up** at [Apify.com](https://apify.com) for faster tweet extraction
- **Get API token** from your Apify account dashboard
- **Set environment variable**: `APIFY_TOKEN=your_token_here`
- **Benefits**: 5-10x faster extraction, better success rate, residential proxies
- **Cost**: Pay-per-use pricing (typically $0.50-2.00 per 1000 tweets)

## 🚀 **Quick Start**

### **Setup Environment Variables (Optional but Recommended)**
```bash
# For faster tweet extraction with Apify
export APIFY_TOKEN=your_apify_token_here

# For Twitter API access (fallback)
export X_BEARER_TOKEN=your_twitter_bearer_token_here
```

**Windows (PowerShell):**
```powershell
$env:APIFY_TOKEN="your_apify_token_here"
$env:X_BEARER_TOKEN="your_twitter_bearer_token_here"
```

### **Option 1: Windows Batch File (Easiest)**
```cmd
run_automation.bat
```

### **Option 2: Direct Python Execution**
```bash
cd scripts
python simple_automated_processor.py
```

### **Option 3: Advanced Configuration**
```bash
cd scripts
python automated_tweet_processor.py
```

## ⚙️ **How It Works**

### **Phase 1: Tweet Extraction**
- **Primary**: **Apify Twitter Scraper** (fastest method, requires API token)
- **Secondary**: Twitter API v2 (if bearer token available)
- **Fallback**: Web scraping with multiple HTML patterns
- **Robust**: Handles various Twitter URL formats

### **Phase 2: AI Preprocessing**
- **Loads your trained model** from 1,300 manual examples
- **Applies learned rules** for emoji removal, elongation normalization, etc.
- **Maintains consistency** with your manual preprocessing style

### **Phase 3: Translation Generation**
- **LibreTranslate**: Free, open-source translation service
- **MyMemory**: Free service with daily limits
- **Caching**: Saves translations to avoid re-processing
- **Fallback**: Marks failed translations for manual review

### **Phase 4: Output Generation**
- **CSV format**: Matches your existing `tweets_split_id.csv` structure
- **Progress tracking**: Saves every 100 tweets to prevent data loss
- **Statistics**: Comprehensive processing reports

## 📊 **Expected Output**

## ⚡ **Performance Comparison**

| Method | Speed | Success Rate | Cost | Setup |
|--------|-------|--------------|------|-------|
| **Apify** | 🚀 **5-10x faster** | 🎯 **95%+** | 💰 **$0.50-2.00/1000** | ⚙️ **API token** |
| **Twitter API** | 🐌 **Standard** | 🎯 **90%+** | 💰 **Free** | ⚙️ **Bearer token** |
| **Web Scraping** | 🐌 **Slowest** | 🎯 **70-80%** | 💰 **Free** | ✅ **No setup** |

**Recommendation**: Use Apify for production runs, Twitter API as backup, web scraping as last resort.

### **CSV Columns Generated**
| Column | Description | Example |
|--------|-------------|---------|
| Original Taglish Tweet | Raw tweet from Twitter | "well sa mga ayaw maniwala pwede naman kayo mag file ng petition...ganyan naman uso ngayon di ba 😂😅" |
| Preprocessed Taglish Tweet | Cleaned by AI model | "well sa mga ayaw maniwala pwede naman kayo mag file ng petition ganyan naman uso ngayon di ba" |
| English Translation | Generated translation | "Well, for those who don't want to believe it, you can file a petition, that's the trend these days, right?" |
| Tweet ID | Twitter status ID | "1463245505031512074" |
| Confidence | AI preprocessing confidence | 0.85 |
| Code Switch Density | Mixing intensity | 0.33 |
| Applied Rules | Preprocessing operations | "remove_emoji_😂; remove_emoji_😅" |
| Processing Status | Success/failure status | "success" |

### **Files Generated**
- `data/automated_processing/final_processed_tweets.csv` - Complete dataset
- `data/automated_processing/processing_statistics.json` - Processing metrics
- `data/cache/translation_cache.json` - Translation cache
- `data/automated_processing/processing.log` - Detailed logs

## ⏱️ **Processing Time Estimates**

### **With Twitter API (Recommended)**
- **Rate**: ~50 tweets/minute
- **Total time**: ~7 hours for 21,000 tweets
- **Success rate**: 90-95%

### **Without Twitter API (Web Scraping)**
- **Rate**: ~20 tweets/minute  
- **Total time**: ~17 hours for 21,000 tweets
- **Success rate**: 70-80%

### **Progress Tracking**
- **Real-time updates** every 10 tweets
- **Intermediate saves** every 100 tweets
- **Statistics updates** every 100 tweets
- **Resume capability** from intermediate files

## 🔧 **Configuration Options**

### **Batch Processing**
```python
# In config.py
BATCH_SIZE = 20                    # Tweets per batch
DELAY_BETWEEN_BATCHES = 2.0       # Seconds between batches
DELAY_BETWEEN_TWEETS = 0.5        # Seconds between tweets
```

### **Rate Limiting**
```python
MAX_REQUESTS_PER_MINUTE = 30      # API rate limiting
MAX_REQUESTS_PER_HOUR = 1000      # Hourly limits
```

### **AI Model Settings**
```python
CONFIDENCE_THRESHOLD = 0.7        # Minimum confidence for rules
SIMILARITY_THRESHOLD = 0.5        # Similarity threshold
```

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **"No trained model found"**
   - Solution: Run `python scripts/tweet_taglish_preprocessor_ai.py` first
   - This trains the AI model on your 1,300 manual examples

2. **"Translation failed"**
   - Solution: Check internet connection
   - Free services have rate limits; wait and retry

3. **"Extraction failed"**
   - Solution: Set `X_BEARER_TOKEN` environment variable
   - Or wait for web scraping fallback

4. **"Memory errors"**
   - Solution: Reduce `BATCH_SIZE` in config.py
   - Process in smaller batches

### **Error Recovery**
- **Automatic retries**: Up to 3 attempts per tweet
- **Graceful degradation**: Falls back to alternative methods
- **Progress preservation**: Intermediate saves prevent data loss
- **Resume capability**: Can restart from last saved point

## 📈 **Monitoring & Progress**

### **Real-time Monitoring**
```bash
# Watch the log file
tail -f data/automated_processing/processing.log

# Check progress
ls -la data/automated_processing/intermediate_*.csv
```

### **Progress Indicators**
- **Batch progress**: "Processing batch 15/1050 (20 tweets)"
- **Overall progress**: "Progress: 1,500/21,152 tweets processed"
- **Success rates**: Updated every 100 tweets
- **Error counts**: Failed extractions, preprocessing, translations

## 🔄 **Resume & Recovery**

### **Automatic Recovery**
- **Intermediate saves**: Every 100 tweets
- **Crash recovery**: Restart from last saved point
- **Partial results**: Keep successful processing

### **Manual Resume**
```bash
# Check last intermediate file
ls -la data/automated_processing/intermediate_*.csv

# Resume processing
python scripts/simple_automated_processor.py
```

## 📊 **Quality Assurance**

### **AI Model Quality**
- **Trained on your data**: Learns from your 1,300 manual examples
- **Consistent rules**: Applies same preprocessing logic
- **Confidence scoring**: Only applies rules when confident

### **Translation Quality**
- **Multiple services**: LibreTranslate + MyMemory fallback
- **Caching**: Avoids re-translating identical text
- **Error marking**: Clearly marks failed translations

### **Data Validation**
- **Format consistency**: Matches your existing CSV structure
- **Content validation**: Checks for empty or malformed data
- **Progress tracking**: Monitors success/failure rates

## 🚀 **Advanced Features**

### **Custom Translation Services**
```python
# Add your own translation API
async def translate_via_custom_service(self, text: str):
    # Implement your translation logic
    pass
```

### **Custom Preprocessing Rules**
```python
# Add domain-specific rules
ai_preprocessor.preprocessing_rules.append(PreprocessingRule(
    pattern=r'custom_pattern',
    replacement='custom_replacement',
    rule_type='custom',
    confidence=0.9
))
```

### **Batch Customization**
```python
# Process specific ranges
processor.process_tweet_range(start_index=1000, end_index=2000)
```

## 📋 **Usage Examples**

### **Process All Tweets**
```bash
python scripts/simple_automated_processor.py
```

### **Process Specific Range**
```python
# Modify the script to process specific indices
links = links[1000:2000]  # Process tweets 1000-2000
```

### **Custom Configuration**
```python
# Modify config.py for your needs
BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 1.0
```

## 🔮 **Future Enhancements**

### **Planned Features**
- **mBART integration**: Direct pipeline to translation model
- **Quality metrics**: Correlation with translation quality
- **Rule evolution**: Automatic rule refinement
- **Multi-language**: Support for other target languages

### **Performance Optimizations**
- **Parallel processing**: Multiple translation services
- **Smart caching**: Intelligent cache management
- **Load balancing**: Distribute across services

## 📞 **Support & Troubleshooting**

### **Getting Help**
1. **Check logs**: `data/automated_processing/processing.log`
2. **Review statistics**: `data/automated_processing/processing_statistics.json`
3. **Check intermediate files**: Look for partial results
4. **Verify configuration**: Check `scripts/config.py`

### **Common Solutions**
- **Slow processing**: Increase delays in config.py
- **High failure rate**: Check internet connection and API limits
- **Memory issues**: Reduce batch size
- **Translation failures**: Wait for rate limits to reset

---

## 🎉 **Ready to Process 21,000+ Tweets?**

This automation system transforms your manual preprocessing expertise into a scalable, automated pipeline that can handle your entire dataset while maintaining the quality and consistency needed for successful mBART training.

**Start with**: `run_automation.bat` (Windows) or `python scripts/simple_automated_processor.py`

**Expected outcome**: Complete dataset with all three columns in the same format as your existing CSV, ready for mBART fine-tuning!
