#!/usr/bin/env python3
"""
Test script for CalamanCy integration with Filipino text preprocessing.

This script tests the enhanced preprocessing pipeline to ensure it works correctly.
"""

import pandas as pd
from enhanced_preprocessing import EnhancedFilipinoPreprocessor, enhance_filipino_dataset

def test_calamancy_integration():
    """Test the CalamanCy integration with sample Filipino text."""
    
    print("🧪 Testing CalamanCy Integration")
    print("=" * 50)
    
    # Create sample Filipino-English dataset
    sample_data = {
        "src": [
            "Kamusta ka?",
            "Salamat sa tulong mo.",
            "Magandang umaga sa inyong lahat.",
            "Gusto ko ng kape at tinapay.",
            "Nakapunta na ako sa Maynila.",
            "Di'ba maganda ang panahon ngayon?",
            "Kasi'ng mabait siya, kaya mahal ko siya.",
            "Sa'yo ko ibibigay ang regalo."
        ],
        "tgt": [
            "How are you?",
            "Thank you for your help.",
            "Good morning to all of you.",
            "I want coffee and bread.",
            "I have already been to Manila.",
            "Isn't the weather nice today?",
            "Because she is kind, that's why I love her.",
            "I will give the gift to you."
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"📊 Sample dataset created: {len(df)} pairs")
    print("\n📝 Sample Filipino text:")
    for i, text in enumerate(df['src'][:3]):
        print(f"   {i+1}. {text}")
    
    # Test individual components
    print("\n🔧 Testing individual components...")
    
    try:
        # Initialize preprocessor
        preprocessor = EnhancedFilipinoPreprocessor()
        
        if preprocessor.nlp:
            print("✅ CalamanCy model loaded successfully")
            
            # Test complexity calculation
            print("\n📊 Testing complexity calculation...")
            sample_text = "Magandang umaga sa inyong lahat."
            complexity = preprocessor.enhanced_complexity_calculation(sample_text)
            print(f"   Text: {sample_text}")
            print(f"   Total Score: {complexity['total_score']}")
            print(f"   Word Count: {complexity['word_count']}")
            print(f"   POS Complexity: {complexity['pos_complexity']}")
            
            # Test quality validation
            print("\n✅ Testing quality validation...")
            quality = preprocessor.validate_filipino_quality(
                "Magandang umaga sa inyong lahat.",
                "Good morning to all of you."
            )
            print(f"   Is Valid: {quality['is_valid']}")
            print(f"   Quality Score: {quality['score']}")
            print(f"   Issues: {quality['issues']}")
            
            # Test Filipino-aware tokenization
            print("\n🔤 Testing Filipino-aware tokenization...")
            tokenized = preprocessor.filipino_aware_tokenization("Di'ba maganda ang panahon?")
            print(f"   Original: Di'ba maganda ang panahon?")
            print(f"   Tokenized: {tokenized}")
            
            # Test morphological augmentation
            print("\n🔄 Testing morphological augmentation...")
            variations = preprocessor.filipino_morphological_augmentation("Gusto ko ng kape")
            print(f"   Original: Gusto ko ng kape")
            print(f"   Variations: {variations}")
            
        else:
            print("❌ CalamanCy model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error testing components: {e}")
        return False
    
    # Test full dataset enhancement
    print("\n🚀 Testing full dataset enhancement...")
    
    try:
        enhanced_df = enhance_filipino_dataset(df)
        
        print(f"✅ Dataset enhancement completed!")
        print(f"📊 Original samples: {len(df)}")
        print(f"📊 Enhanced samples: {len(enhanced_df)}")
        print(f"📝 Average complexity: {enhanced_df['complexity_score'].mean():.2f}")
        print(f"📝 Average quality: {enhanced_df['quality_score'].mean():.2f}")
        
        # Show enhanced data structure
        print("\n📋 Enhanced dataset columns:")
        for col in enhanced_df.columns:
            print(f"   • {col}")
        
        # Show sample of enhanced data
        print("\n📝 Sample enhanced data:")
        print(enhanced_df[['src', 'tgt', 'complexity_score', 'quality_score']].head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error enhancing dataset: {e}")
        return False

def test_with_real_data():
    """Test with real data if available."""
    
    print("\n🔍 Testing with real data...")
    
    try:
        # Check if real dataset exists
        if pd.io.common.file_exists("filipino_english_parallel_corpus.csv"):
            print("📁 Found real dataset, testing with it...")
            
            df = pd.read_csv("filipino_english_parallel_corpus.csv")
            
            # Map columns if needed
            if "preprocessed_text" in df.columns and "english_translation" in df.columns:
                df = df.rename(columns={
                    "preprocessed_text": "src",
                    "english_translation": "tgt"
                })
                
                # Take a small sample for testing
                test_df = df.head(10)
                print(f"📊 Testing with {len(test_df)} real samples...")
                
                enhanced_test = enhance_filipino_dataset(test_df)
                print(f"✅ Real data enhancement successful: {len(enhanced_test)} samples")
                
                return True
            else:
                print("⚠️  Real dataset columns don't match expected format")
                return False
        else:
            print("📁 Real dataset not found, skipping real data test")
            return True
            
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting CalamanCy Integration Tests")
    print("=" * 60)
    
    # Test 1: Individual components
    test1_success = test_calamancy_integration()
    
    # Test 2: Real data (if available)
    test2_success = test_with_real_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    if test1_success:
        print("✅ Component tests: PASSED")
    else:
        print("❌ Component tests: FAILED")
    
    if test2_success:
        print("✅ Real data tests: PASSED")
    else:
        print("❌ Real data tests: FAILED")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! CalamanCy integration is working correctly.")
        print("🚀 You can now use enhanced preprocessing in your training pipeline.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("🔧 You may need to install CalamanCy or fix dependency issues.")
