import pandas as pd
from normalizer import FilipinoNormalizer

def test_enhanced_normalization():
    """Test all enhanced normalization rules"""
    
    # Initialize the enhanced normalizer
    normalizer = FilipinoNormalizer('rules.json', 'logs')
    
    # Test cases for each rule category
    test_cases = [
        # 1. Substitution Rules
        ("kumusta ka?", "o↔u substitution"),
        ("babae ako", "e↔i substitution"),
        ("kolehiala", "y↔i substitution"),
        
        # 2. Deletion Rules
        ("mahhalaga", "redundant h removal"),
        ("helllo", "duplicate character removal"),
        
        # 3. Insertion Rules
        ("mag aral", "hyphen insertion for affixes"),
        ("sama sama", "hyphen insertion for reduplication"),
        
        # 4. Transposition Rules (NEW)
        ("alakt", "letter-order swap (alakt → aklat)"),
        ("klat", "missing initial vowel (klat → aklat)"),
        ("ngay", "remove ng prefix (ngay → ay)"),
        
        # 5. Token Split Rules (NEW)
        ("nakapunta", "token separation (nakapunta → naka punta)"),
        ("nagprint", "affix separation (nagprint → nag print)"),
        
        # 6. Token Merge Rules (NEW)
        ("na ka punta", "token combination (na ka punta → nakapunta)"),
        ("nag print", "affix combination (nag print → nagprint)"),
        
        # 7. Enhanced Punctuation Rules (NEW)
        ("di ba", "contraction fix (di ba → di'ba)"),
        ("na nga", "apostrophe insertion (na nga → na'nga)"),
        
        # 8. Slang-to-Standard Rules (ENHANCED)
        ("q", "Filipino slang (q → ako)"),
        ("u", "Filipino slang (u → ikaw)"),
        ("2", "English slang (2 → to)"),
        ("gr8", "English slang (gr8 → great)"),
        ("omg", "Internet slang (omg → oh my god)"),
        
        # 9. Mixed Content Examples
        ("q nakapunta na 2 the mall", "mixed Filipino-English with slang"),
        ("alakt ko na nga", "transposition + slang + contraction"),
        ("nagprint ako ng araw araw", "affix split + reduplication"),
    ]
    
    print("🧪 Testing Enhanced Filipino Normalization Rules")
    print("=" * 80)
    
    for i, (test_text, description) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {description}")
        print(f"Original: {test_text}")
        
        # Apply normalization
        normalized_text, logs = normalizer.normalize_text(test_text)
        print(f"Normalized: {normalized_text}")
        
        # Show applied rules
        if logs:
            print(f"Rules applied: {len(logs)}")
            for log in logs[:3]:  # Show first 3 rules
                rule_type = log.get('meta', {}).get('type', log.get('reason', 'unknown'))
                print(f"  - {rule_type}")
            if len(logs) > 3:
                print(f"  ... and {len(logs) - 3} more rules")
        else:
            print("No rules applied")
        
        print("-" * 60)
    
    print(f"\n✅ Enhanced normalization testing completed!")
    print(f"Check the logs/ directory for detailed rule application logs.")

def test_excel_processing_with_enhanced_rules():
    """Test the enhanced normalizer on Excel data"""
    
    print("\n🔍 Testing Enhanced Normalizer on Excel Data")
    print("=" * 80)
    
    try:
        # Load sample data from the processed Excel file
        df = pd.read_excel('tweets_split_id_processed.xlsx', sheet_name='tweets_split_id')
        valid_tweets = df[df['Tweet Status'] == 1].head(3)
        
        print(f"Testing on {len(valid_tweets)} sample tweets from Excel file:")
        
        for i, (_, row) in enumerate(valid_tweets.iterrows(), 1):
            original = row['Original Taglish Tweet']
            processed = row['Preprocessed Taglish Tweet']
            
            print(f"\nTweet {i}:")
            print(f"  Original: {original[:100]}...")
            print(f"  Processed: {processed[:100]}...")
            
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Make sure to run preprocessing first: python preprocess_tweets.py")

if __name__ == "__main__":
    # Test enhanced normalization rules
    test_enhanced_normalization()
    
    # Test on Excel data if available
    test_excel_processing_with_enhanced_rules()
    
    print(f"\n🚀 Enhanced normalization system ready!")
    print(f"Run preprocessing: python preprocess_tweets.py")
    print(f"View results: python test_results.py")
