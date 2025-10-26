#!/usr/bin/env python3
"""
Test script for the baseline training script
This script tests the dataset loading and basic functionality without running full training
"""

import pandas as pd
import os
import sys

def test_dataset_loading():
    """Test if the baseline dataset can be loaded correctly"""
    print("🧪 Testing baseline dataset loading...")
    
    # Check if the baseline corpus exists
    baseline_corpus_path = "filipino_english_parallel_corpus.csv"
    if not os.path.exists(baseline_corpus_path):
        print(f"❌ Baseline corpus not found: {baseline_corpus_path}")
        return False
    
    try:
        # Load the dataset
        df = pd.read_csv(baseline_corpus_path)
        print(f"✅ Loaded {len(df)} samples from baseline corpus")
        
        # Check required columns
        required_columns = ['id', 'text', 'preprocessed_text', 'english_translation']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print(f"✅ All required columns present: {list(df.columns)}")
        
        # Check data quality
        print(f"📊 Dataset statistics:")
        print(f"   Total rows: {len(df)}")
        print(f"   Rows with preprocessed_text: {df['preprocessed_text'].notna().sum()}")
        print(f"   Rows with english_translation: {df['english_translation'].notna().sum()}")
        
        # Sample some data
        print(f"\n📋 Sample data:")
        sample_data = df[['preprocessed_text', 'english_translation']].head(3)
        for i, (_, row) in enumerate(sample_data.iterrows(), 1):
            print(f"   {i}. SRC: {row['preprocessed_text'][:50]}...")
            print(f"      TGT: {row['english_translation'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading baseline dataset: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT (Parameter-Efficient Fine-Tuning)',
        'pandas': 'Pandas',
        'tqdm': 'TQDM',
        'nltk': 'NLTK'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} import failed")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install torch transformers peft pandas tqdm nltk")
        return False
    
    print("✅ All required packages imported successfully")
    return True

def test_cuda_availability():
    """Test CUDA availability"""
    print("\n🧪 Testing CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   Device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA is not available, will use CPU")
        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Baseline Training Script Components")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed")
        return
    
    # Test CUDA
    if not test_cuda_availability():
        print("\n❌ CUDA test failed")
        return
    
    # Test dataset loading
    if not test_dataset_loading():
        print("\n❌ Dataset loading test failed")
        return
    
    print("\n🎉 All tests passed!")
    print("✅ The baseline training script should work correctly")
    print("\n💡 To run training:")
    print("   python model_training_baseline.py")

if __name__ == "__main__":
    main()
