import pandas as pd

# Load the processed Excel data
df = pd.read_excel('tweets_split_id_processed.xlsx', sheet_name='tweets_split_id')

# Filter only valid tweets (Tweet Status = 1)
valid_tweets = df[df['Tweet Status'] == 1]

print("Enhanced Preprocessing Results (Excel File - Valid Tweets Only):")
print("=" * 80)
print(f"Total rows in worksheet: {len(df)}")
print(f"Valid tweets (Status = 1): {len(valid_tweets)}")
print("=" * 80)

# Show first 5 examples of valid tweets
for i, row in valid_tweets.head(5).iterrows():
    print(f"Tweet {i+1} (ID: {row.get('id', 'N/A')}):")
    print(f"Original: {row['Original Taglish Tweet']}")
    print(f"Processed: {row['Preprocessed Taglish Tweet']}")
    print("-" * 80)

print(f"\nNew Features Applied:")
print("✓ All text converted to lowercase")
print("✓ Periods added at sentence endings")
print("✓ English text preserved")
print("✓ Filipino normalization applied")
print("✓ Social media artifacts cleaned")
print("✓ Gibberish removed")
print("✓ Excel file processing")
print("✓ Tweet Status filtering (only Status = 1)")

print("\nCheck the 'tweets_split_id_processed.xlsx' file for complete results.")
print("Check the 'logs/' directory for detailed normalization logs.")
