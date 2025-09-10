import json
import pandas as pd
import re
import os
from datetime import datetime
from normalizer import FilipinoNormalizer

class TweetPreprocessor:
    def __init__(self, rules_path, log_dir):
        """Initialize the preprocessor with rules and logging"""
        self.normalizer = FilipinoNormalizer(rules_path, log_dir)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def preprocess_tweet(self, original_text, tweet_id=None):
        """Preprocess a single tweet using the enhanced normalizer"""
        if pd.isna(original_text) or original_text == '':
            return '', []
            
        # Apply comprehensive normalization using the enhanced normalizer
        context = {"tweet_id": tweet_id} if tweet_id else None
        normalized_text, applied_logs = self.normalizer.normalize_text(original_text, context)
        
        return normalized_text, applied_logs
    
    def process_excel(self, input_file, output_file, worksheet_name='tweets_split_id'):
        """Process the Excel file, filtering by Tweet Status = 1"""
        print(f"Loading Excel file: {input_file}")
        print(f"Worksheet: {worksheet_name}")
        
        # Read the Excel file
        try:
            df = pd.read_excel(input_file, sheet_name=worksheet_name)
            print(f"Loaded {len(df)} total rows from worksheet '{worksheet_name}'")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return
        
        # Check if required columns exist
        required_cols = ['Tweet Status', 'Original Taglish Tweet', 'Preprocessed Taglish Tweet']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Filter rows where Tweet Status = 1 (valid tweets)
        valid_tweets = df[df['Tweet Status'] == 1].copy()
        print(f"Found {len(valid_tweets)} rows with Tweet Status = 1 (valid tweets)")
        
        if len(valid_tweets) == 0:
            print("No valid tweets found to process!")
            return
        
        # Process each valid tweet
        print("Starting preprocessing of valid tweets...")
        processed_count = 0
        error_count = 0
        
        for index, row in valid_tweets.iterrows():
            try:
                original_text = row['Original Taglish Tweet']
                tweet_id = row['id'] if 'id' in row else index
                
                # Preprocess the tweet
                normalized_text, logs = self.preprocess_tweet(original_text, tweet_id)
                
                # Update the preprocessed column in the original dataframe
                df.at[index, 'Preprocessed Taglish Tweet'] = normalized_text
                
                processed_count += 1
                
                # Progress indicator
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} valid tweets...")
                    
            except Exception as e:
                print(f"Error processing tweet {index + 1}: {e}")
                error_count += 1
                # Set empty string for failed tweets
                df.at[index, 'Preprocessed Taglish Tweet'] = ''
        
        # Save the processed data back to Excel
        print(f"Saving processed data to: {output_file}")
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Write the main worksheet with processed data
                df.to_excel(writer, sheet_name=worksheet_name, index=False)
                
                # Also save as CSV for compatibility
                csv_output = output_file.replace('.xlsx', '.csv')
                df.to_csv(csv_output, index=False, encoding='utf-8')
                print(f"Also saved as CSV: {csv_output}")
                
            print(f"Successfully saved {output_file}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return
        
        # Print summary
        print(f"\nPreprocessing completed!")
        print(f"Total rows in worksheet: {len(df)}")
        print(f"Valid tweets (Status = 1): {len(valid_tweets)}")
        print(f"Tweets processed: {processed_count}")
        print(f"Errors encountered: {error_count}")
        print(f"Output file: {output_file}")
        
        # Show sample of processed tweets
        print(f"\nSample of processed tweets:")
        sample_processed = df[df['Tweet Status'] == 1][['Original Taglish Tweet', 'Preprocessed Taglish Tweet']].head(3)
        for i, (_, row) in enumerate(sample_processed.iterrows()):
            print(f"Tweet {i+1}:")
            print(f"  Original: {row['Original Taglish Tweet'][:100]}...")
            print(f"  Processed: {row['Preprocessed Taglish Tweet'][:100]}...")
            print()

def main():
    """Main execution function"""
    # Configuration
    rules_path = 'rules.json'
    log_dir = 'logs'
    input_file = 'tweets_split_id.xlsx'
    output_file = 'tweets_split_id_processed.xlsx'
    worksheet_name = 'tweets_split_id'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    # Check if rules file exists
    if not os.path.exists(rules_path):
        print(f"Rules file not found: {rules_path}")
        return
    
    # Initialize preprocessor
    print("Initializing Tweet Preprocessor...")
    preprocessor = TweetPreprocessor(rules_path, log_dir)
    
    # Process the Excel file
    preprocessor.process_excel(input_file, output_file, worksheet_name)

if __name__ == "__main__":
    main()
