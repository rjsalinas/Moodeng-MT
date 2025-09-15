# Large Data Files Management

This directory contains large data files that were moved to prevent GitHub issues.

## Files Backed Up:
- **Large CSV files**: Very large tweet datasets (>10MB)
- **JSON files**: Raw tweet data from extraction
- **Model files**: Fine-tuned translation models
- **Log files**: Processing logs

## Important CSV Files Kept in Repository:
- `tweets_id_filipino_text_only.csv`: Filipino/Taglish tweets (important for research)
- `tweets_id_filipino_text_normalized.csv`: Normalized Filipino tweets (research results)
- `tweets_id_non_fil_tag_taglish.csv`: Non-Filipino tweets (analysis data)

## Large Files Moved to Backup:
- `tweets_id_text_only.csv`: Raw extracted data (very large)
- `tweets_id_text_normalized.csv`: Full normalized dataset (very large)
- `english_translation_from_preprocessed_texts.csv`: Translation outputs (large)
- `dataset_*.json`: Raw JSON datasets (very large)

## To Use These Files:
1. **For Research**: Use the kept CSV files directly
2. **For Processing**: Copy large files back when needed
3. **For Analysis**: The important results are already in the repository

## Recommended Workflow:
1. **Keep in Git**: Important processed results and analysis data
2. **Backup**: Very large raw datasets and intermediate files
3. **Share**: Large files via cloud storage or data repositories
4. **Research**: Use the kept CSV files for your thesis analysis

## File Types to Keep in Git:
- Python scripts (*.py)
- Configuration files (rules.json, requirements.txt)
- Documentation (*.md)
- **Important CSV results** (Filipino tweets, normalized data)
- Batch files (*.bat, *.ps1)

## File Types to Exclude from Git:
- Very large datasets (>10MB)
- Raw extraction files
- Model weights (*.safetensors, *.bin)
- Log files (*.log, *.jsonl)
- Temporary processing outputs
