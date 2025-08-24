from normalizer import FilipinoNormalizer

def test_punctuation_preservation():
    """Test punctuation preservation and repeated mark removal"""
    
    # Initialize the normalizer
    normalizer = FilipinoNormalizer('rules.json', 'logs')
    
    # Test cases for punctuation handling
    test_cases = [
        # Preserve original ending punctuation
        ("Hello world!", "Exclamation mark preserved"),
        ("Kamusta ka?", "Question mark preserved"),
        ("Wow.", "Period preserved"),
        ("Test;", "Semicolon preserved"),
        
        # Remove repeated punctuation marks
        ("Hello world!!!", "Multiple exclamation marks → single"),
        ("Kamusta ka???", "Multiple question marks → single"),
        ("Wow...", "Multiple periods → single"),
        ("Test,,,", "Multiple commas → single"),
        
        # Mixed repeated punctuation
        ("Hello world!!!...", "Mixed repeated → single !"),
        ("Test???...", "Mixed repeated → single ?"),
        ("Wow...!!!", "Mixed repeated → single !"),
        
        # No ending punctuation (should add period)
        ("Hello world", "No punctuation → add period"),
        ("Kamusta ka", "No punctuation → add period"),
        
        # Complex cases
        ("q nakapunta na 2 the mall!!!", "Mixed content with repeated punctuation"),
        ("alakt ko na nga???", "Mixed content with repeated punctuation"),
        ("So relaxing... Pero mas relaxing", "Multiple sentences with ellipsis"),
    ]
    
    print("🔒 Testing Punctuation Preservation & Cleanup")
    print("=" * 70)
    
    for i, (test_text, description) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {description}")
        print(f"Original: '{test_text}'")
        
        # Apply normalization
        normalized_text, logs = normalizer.normalize_text(test_text)
        print(f"Normalized: '{normalized_text}'")
        
        # Analyze the ending
        if normalized_text:
            ending = normalized_text[-1]
            print(f"Ending punctuation: '{ending}'")
            
            # Check if it's a valid ending
            if ending in '.,!?;:':
                print("✅ Valid ending punctuation")
            else:
                print("❌ No ending punctuation")
        
        # Show applied rules
        if logs:
            print(f"Rules applied: {len(logs)}")
            for log in logs:
                rule_type = log.get('reason', 'unknown')
                print(f"  - {rule_type}")
        
        print("-" * 60)
    
    print(f"\n✅ Punctuation preservation testing completed!")
    print(f"Original ending punctuation should be preserved when present.")
    print(f"Repeated punctuation marks should be cleaned up.")

if __name__ == "__main__":
    test_punctuation_preservation()
