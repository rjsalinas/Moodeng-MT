try:
    import pandas as pd  # Optional: only needed for CSV->preprocessed_text flow
except Exception:
    pd = None
import os
from datetime import datetime
# Normalizer is archived; make import optional to keep this module importable
try:
    from src.preprocessing.normalizer import FilipinoNormalizer  # archived
except Exception:
    FilipinoNormalizer = None
from collections import Counter, defaultdict
# Pipeline-level normalization (context-aware rules + lexica)
from src.utils.normalize_pipeline import (
    load_json,
    compile_regex_patterns,
    load_and_merge_lexica,
    augment_rules,
    validate_rules,
    normalize_text,
    load_and_merge_rules,
)

def normalize_csv_tweets(input_csv, output_csv, rules_path='config/rules.json', log_dir='logs'):
    """
    Normalize the 'text' column from a CSV file and save with 'preprocessed_text' column
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        rules_path (str): Path to rules.json file
        log_dir (str): Directory for logging
    """
    
    print(f"Starting tweet normalization process...")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    print(f"Rules file: {rules_path}")
    print(f"Log directory: {log_dir}")
    
    if pd is None:
        print("✗ pandas is not installed. Install it to use normalize_csv_tweets(). Raw JSON normalization can run without pandas.")
        return
    # Initialize the normalizer
    try:
        normalizer = FilipinoNormalizer(rules_path, log_dir)
        print("✓ Normalizer initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing normalizer: {e}")
        return
    
    # Read the input CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df)} rows from {input_csv}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        return
    
    # Check if 'text' column exists
    if 'text' not in df.columns:
        print(f"✗ 'text' column not found. Available columns: {list(df.columns)}")
        return
    
    # Check if 'id' column exists
    if 'id' not in df.columns:
        print(f"✗ 'id' column not found. Available columns: {list(df.columns)}")
        return
    
    # Add preprocessed_text column
    df['preprocessed_text'] = ''
    
    # Process each tweet
    print("\nStarting normalization...")
    processed_count = 0
    error_count = 0
    
    for index, row in df.iterrows():
        try:
            tweet_id = row['id']
            original_text = row['text']
            
            # Skip empty or NaN text
            if pd.isna(original_text) or original_text == '':
                df.at[index, 'preprocessed_text'] = ''
                continue
            
            # Normalize the text
            context = {"tweet_id": str(tweet_id)}
            normalized_text, applied_logs = normalizer.normalize_text(original_text, context)
            
            # Store the normalized text
            df.at[index, 'preprocessed_text'] = normalized_text
            
            processed_count += 1
            
            # Progress indicator
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} tweets...")
                
        except Exception as e:
            print(f"  ✗ Error processing tweet {index + 1} (ID: {row.get('id', 'unknown')}): {e}")
            error_count += 1
            # Set empty string for failed tweets
            df.at[index, 'preprocessed_text'] = ''
    
    print(f"\n✓ Completed processing {processed_count} tweets")
    if error_count > 0:
        print(f"⚠ {error_count} tweets had errors during processing")
    
    # Save the results
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✓ Saved normalized data to: {output_csv}")
        
        # Show some statistics
        total_tweets = len(df)
        empty_original = df['text'].isna().sum() + (df['text'] == '').sum()
        empty_processed = df['preprocessed_text'].isna().sum() + (df['preprocessed_text'] == '').sum()
        
        print(f"\nSummary:")
        print(f"  Total tweets: {total_tweets}")
        print(f"  Empty original text: {empty_original}")
        print(f"  Empty processed text: {empty_processed}")
        print(f"  Successfully processed: {total_tweets - empty_processed}")
        
    except Exception as e:
        print(f"✗ Error saving output file: {e}")
        return
    
    print(f"\n🎉 Normalization complete! Check the output file: {output_csv}")

if __name__ == "__main__":
    # File paths
    input_file = "tweets_id_filipino_text_only.csv"
    output_file = "tweets_id_filipino_text_normalized.csv"
    
    # Run the normalization
    normalize_csv_tweets(input_file, output_file)

# ---------------------
# New: corpus runner used by top-level shim
# ---------------------
def run_parallel_corpus_normalization(
    in_path: str = 'data/corpus/filipino_english_parallel_corpus.csv',
    out_path: str = 'data/processed/filipino_english_parallel_corpus_normalized_v3.csv',
    tmp_path: str = 'data/processed/filipino_english_parallel_corpus_normalized.tmp.csv',
    rules_path: str = 'config/rules.json',
    regex_path: str = 'config/regex_patterns.json',
    lexica_dir: str = 'config/lexica/lexica',
    pipeline_cfg_path: str = 'config/pipeline_config.json',
):
    print('Loading configuration...')
    # Load rules from rules.json only (single source of truth)
    from pathlib import Path
    primary = Path('config/rules.json')
    rules = load_and_merge_rules(primary, None)
    regexes = compile_regex_patterns(regex_path)
    lex = load_and_merge_lexica(lexica_dir)
    cfg = load_json(pipeline_cfg_path)

    print('Augmenting and validating rules...')
    rules, _ = augment_rules(rules, cfg)
    errs = validate_rules(rules)
    if errs:
        print(f'Rule validation errors: {errs}')

    print('Processing corpus...')
    rule_counts = Counter()
    examples = defaultdict(list)

    import csv
    with open(in_path, 'r', encoding='utf-8') as f, \
         open(tmp_path, 'w', encoding='utf-8', newline='') as w:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['normalized_text'] if 'normalized_text' not in reader.fieldnames else reader.fieldnames
        writer = csv.DictWriter(w, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0
        for row in reader:
            text = row.get('preprocessed_text') or row.get('text') or ''
            normalized, metadata = normalize_text(text, rules, lex, regexes, cfg)
            row['normalized_text'] = normalized
            writer.writerow(row)

            for applied_rule in metadata['applied_rules']:
                rid = applied_rule.get('rule')
                rule_counts[rid] += 1
                if len(examples[rid]) < 3:
                    examples[rid].append({'orig': text[:160], 'norm': normalized[:160]})

            processed += 1
            if processed % 100 == 0:
                print(f'Processed {processed} rows...')

    try:
        os.remove(out_path)
    except FileNotFoundError:
        pass
    os.replace(tmp_path, out_path)

    print('\n' + '='*50)
    print('NORMALIZATION COMPLETE!')
    print('='*50)
    print(f'Total rows processed: {processed}')
    print(f'Total rules applied: {sum(rule_counts.values())}')
    print(f'Output saved to: {out_path}')

    print('\nTop 15 most applied rules:')
    for rule, count in rule_counts.most_common(15):
        print(f'  {rule}: {count} times')

    print('\nSample transformations:')
    for rule, count in rule_counts.most_common(5):
        if examples[rule]:
            print(f'\n{rule} (applied {count} times):')
            for i, ex in enumerate(examples[rule][:2], 1):
                print(f'  Example {i}:')
                print(f"    Original: {ex['orig']}")
                print(f"    Normalized: {ex['norm']}")


# ---------------------
# New: normalize raw tweet JSON dumps (dataset_01.json..dataset_07.json)
# ---------------------
def run_normalize_raw_tweets(
    raw_dir: str = 'data/raw/tweets',
    out_path: str = 'data/processed/raw_tweets_normalized_v1.csv',
    tmp_path: str = 'data/processed/raw_tweets_normalized.tmp.csv',
    rules_path: str = 'config/rules.json',
    regex_path: str = 'config/regex_patterns.json',
    lexica_dir: str = 'config/lexica/lexica',
    pipeline_cfg_path: str = 'config/pipeline_config.json',
    include_glob: str = 'dataset_*.json',
    exclude_glob: str = '',
    dedup_by_normalized: bool = False,
):
    print('Loading configuration...')
    # Load rules from rules.json only (single source of truth)
    from pathlib import Path
    primary = Path('config/rules.json')
    rules = load_and_merge_rules(primary, None)
    regexes = compile_regex_patterns(regex_path)
    lex = load_and_merge_lexica(lexica_dir)
    cfg = load_json(pipeline_cfg_path)

    print('Augmenting and validating rules...')
    rules, _ = augment_rules(rules, cfg)
    errs = validate_rules(rules)
    if errs:
        print(f'Rule validation errors: {errs}')

    # Collect JSON files
    import glob, json, csv
    files = sorted(glob.glob(os.path.join(raw_dir, include_glob)))
    if exclude_glob:
        excl = set(glob.glob(os.path.join(raw_dir, exclude_glob)))
        files = [f for f in files if f not in excl]
    print(f'Found {len(files)} raw datasets in {raw_dir}')

    with open(tmp_path, 'w', encoding='utf-8', newline='') as w:
        writer = csv.DictWriter(w, fieldnames=['id', 'text', 'normalized_text'])
        writer.writeheader()

        total = 0
        seen_ids = set()
        seen_norm = set()
        import re
        url_re = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        mention_re = re.compile(r'@\w+')
        hashtag_re = re.compile(r'#\w+')

        for jf in files:
            print(f'Reading {jf} ...')
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f'  Skipping {jf} due to read error: {e}')
                continue

            for item in data:
                tweet_id = item.get('id')
                text = (item.get('text') or '').replace('\n', ' ').strip()

                # Filtering similar to prior preprocessing
                if not text:
                    continue
                # drop retweets
                if text.startswith('RT '):
                    continue
                # remove urls/mentions/hashtags and check emptiness
                stripped = hashtag_re.sub('', mention_re.sub('', url_re.sub('', text))).strip()
                if not stripped:
                    continue
                # de-dup by id
                if tweet_id in seen_ids:
                    continue

                normalized, _ = normalize_text(text, rules, lex, regexes, cfg)

                # de-dup by normalized text (optional)
                if dedup_by_normalized:
                    norm_key = normalized.lower()
                    if norm_key in seen_norm:
                        continue

                writer.writerow({'id': tweet_id, 'text': text, 'normalized_text': normalized})
                seen_ids.add(tweet_id)
                if dedup_by_normalized:
                    seen_norm.add(norm_key)
                total += 1
                if total % 100 == 0:
                    print(f'  Processed {total} tweets...')

    try:
        os.remove(out_path)
    except FileNotFoundError:
        pass
    os.replace(tmp_path, out_path)
    print(f'Done. Wrote {out_path}')
