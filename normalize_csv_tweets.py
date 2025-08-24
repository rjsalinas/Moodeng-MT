import pandas as pd
import os
from datetime import datetime
from normalizer import FilipinoNormalizer

def normalize_csv_tweets(input_csv, output_csv, rules_path='rules.json', log_dir='logs'):
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
    input_file = "tweets_id_text_only.csv"
    output_file = "tweets_id_text_normalized.csv"
    
    # Run the normalization
    normalize_csv_tweets(input_file, output_file)
