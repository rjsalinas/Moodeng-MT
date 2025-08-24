# Enhanced Filipino Tweet Preprocessing System

## **English Text Preservation** ✅
**IMPORTANT**: This system now preserves English text while cleaning Filipino/Tagalog content. English words, phrases, and structure are maintained intact.

## **New Text Standardization Features** ✅
**IMPORTANT**: All text is now converted to lowercase and periods are added at sentence endings for consistent formatting.

## New Features Added

### 1. **Text Standardization** (NEW!)
- **Lowercase Conversion**: All text converted to lowercase for consistency
- **Sentence End Periods**: Periods automatically added at sentence endings
- **Uniform Format**: Consistent text formatting across all tweets

### 2. **Gibberish and Keyboard Smashing Detection** (English-Aware)
- **Keyboard Smashing**: Removes patterns like "qwertyuiop", "asdfghjkl", "zxcvbnm" (6+ characters only)
- **Long Consonant Clusters**: Removes sequences like "bcdfghjklmnpqrstvwxz" (7+ consonants only)
- **Long Vowel Clusters**: Removes sequences like "aeiou" (6+ vowels only)
- **Random Character Sequences**: Removes alphanumeric gibberish (10+ characters only)
- **Smart Detection**: Uses English word patterns to avoid removing legitimate English words
- **Conservative Approach**: Only removes clearly gibberish patterns, preserves real words

### 3. **Social Media Artifact Removal** (Content-Preserving)
- **Hashtags**: Converts "#tagalog" → "tagalog" (keeps text, removes #)
- **Mentions**: Removes "@username" completely
- **RT Patterns**: Removes "RT @username", "via @username", "cc @username"
- **Punctuation Cleanup**: Only removes excessive symbols (3+ repeated), keeps basic punctuation
- **Content Preservation**: Keeps meaningful content in parentheses and brackets

### 4. **Enhanced Text Cleaning** (English-Friendly)
- **URL Removal**: Removes all HTTP/HTTPS links
- **Non-printing Characters**: Removes invisible characters and emojis
- **Whitespace Normalization**: Cleans up multiple spaces and line breaks
- **Character Encoding**: Handles Filipino diacritics and special characters
- **English Preservation**: Maintains English word structure and readability

### 5. **Improved Slang Handling** (Context-Aware)
- **Context-Aware**: Only replaces standalone slang terms, not letters within words
- **Examples**: 
  - "q" → "ako" (only when standalone)
  - "u" → "ikaw" (only when standalone)
  - "sya" → "siya" (only when standalone)
- **Preserves**: Words like "dump", "you", "putol", "special", "someone" remain intact

### 6. **Final Formatting Cleanup** (Conservative)
- **Punctuation Spacing**: Ensures proper spacing around punctuation marks
- **Sentence Boundaries**: Adds spaces after sentence endings
- **Empty Elements**: Removes only empty parentheses and brackets
- **Leading/Trailing**: Cleans up punctuation at text boundaries
- **Gentle Approach**: Minimal changes to preserve original text structure

## Example Transformations

### Before (Original Tweet):
```
"what do you do ba when have makulog? we make putol it di ba? #tagalog @username"
```

### After (Processed):
```
"what do you do ba when have makulog? we make putol it di ba? tagalog."
```

### What Was Preserved:
- `what do you do` → **Kept intact** (English words)
- `when have` → **Kept intact** (English words)
- `we make putol` → **Kept intact** (English + Filipino mixed)
- `special someone` → **Kept intact** (English words)

### What Was Cleaned:
- `#tagalog` → `tagalog` (hashtag symbol removed)
- `@username` → Completely removed (mention)
- Excessive punctuation → Normalized (3+ repeated only)
- **Text case** → **Converted to lowercase**
- **Sentence end** → **Period added**

## Rule Priority System

1. **Text Cleaning** (highest priority)
   - URL removal, whitespace cleanup, character encoding

2. **Gibberish Detection** (English-aware)
   - Conservative keyboard smashing, random sequences
   - Preserves English word patterns

3. **Social Media Cleaning** (content-preserving)
   - Hashtags, mentions, social artifacts
   - Keeps meaningful content

4. **Orthographic Normalization**
   - o↔u, e↔i, y↔i, ng rules, etc.
   - Only applies to Filipino content

5. **Final Cleanup** (conservative)
   - Minimal formatting and spacing
   - Preserves original text structure

6. **Text Standardization** (new - lowest priority)
   - Convert to lowercase
   - Add sentence end periods

## Benefits

- **English Text Preserved**: All English words, phrases, and structure maintained
- **Cleaner Filipino Data**: Removes noise from Filipino/Tagalog content
- **Consistent Formatting**: All text in lowercase with proper sentence endings
- **Better NLP**: Improves machine learning model performance
- **Audit Trail**: Complete logging of all changes
- **Bilingual Aware**: Respects both English and Filipino language patterns

## Usage

Run the preprocessing:
```bash
python preprocess_tweets.py
```

Check results:
```bash
python test_results.py
```

View enhanced results (lowercase + periods):
```bash
python test_enhanced_results.py
```

View English preservation examples:
```bash
python show_english_examples.py
```

View logs:
```bash
# Check logs/normalization_log.jsonl for detailed change tracking
```
