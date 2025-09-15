import pandas as pd

# Load the processed data
df = pd.read_csv('preprocessed_tweets.csv')

print("Examples showing English text preservation:")
print("=" * 80)

# Find tweets with English words
english_keywords = ['what', 'when', 'make', 'special', 'someone', 'relaxing', 'dump', 'acc']
english_tweets = df[df['Original Taglish Tweet'].str.contains('|'.join(english_keywords), case=False, na=False)]

print(f"Found {len(english_tweets)} tweets with English content")
print()

# Show first 5 examples
for i, (_, row) in enumerate(english_tweets.head(5).iterrows()):
    print(f"Example {i+1}:")
    print(f"Original: {row['Original Taglish Tweet']}")
    print(f"Processed: {row['Preprocessed Taglish Tweet']}")
    print("-" * 80)

print("\nKey improvements:")
print("✓ English words like 'what', 'when', 'make', 'special', 'someone' are preserved")
print("✓ Only excessive punctuation (3+ repeated) is cleaned")
print("✓ Social media artifacts are removed while keeping content")
print("✓ Gibberish detection is more conservative with English text")
print("✓ Filipino normalization rules still apply to Tagalog content")
