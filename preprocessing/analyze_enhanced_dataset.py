#!/usr/bin/env python3
"""
Quick Analysis of Enhanced Filipino-English Parallel Corpus

This script analyzes the enhanced dataset to show:
- Dataset statistics
- Complexity and quality distributions
- Sample data preview
- Training readiness assessment
"""

import pandas as pd
import numpy as np

def analyze_enhanced_dataset():
    """Analyze the enhanced dataset."""
    
    print("🔍 ANALYZING ENHANCED DATASET")
    print("=" * 60)
    
    try:
        # Load the enhanced dataset
        df = pd.read_csv('full_enhanced_parallel_corpus.csv')
        print(f"✅ Dataset loaded successfully!")
        
        # Basic info
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns)}")
        
        # Sample data preview
        print(f"\n📝 SAMPLE DATA (first 3 rows):")
        display_cols = ['src', 'tgt', 'tagalog_complexity', 'quality_score']
        for i, row in df.head(3).iterrows():
            print(f"\nRow {i+1}:")
            for col in display_cols:
                if col in df.columns:
                    value = row[col]
                    if col in ['src', 'tgt'] and isinstance(value, str):
                        # Truncate long text
                        if len(value) > 80:
                            value = value[:80] + "..."
                    print(f"  {col}: {value}")
        
        # Complexity analysis
        if 'tagalog_complexity' in df.columns:
            complexity = df['tagalog_complexity']
            print(f"\n🎯 TAGALOG COMPLEXITY ANALYSIS:")
            print(f"   Range: {complexity.min():.1f} - {complexity.max():.1f}")
            print(f"   Mean: {complexity.mean():.2f}")
            print(f"   Std: {complexity.std():.2f}")
            print(f"   Median: {complexity.median():.2f}")
            
            # Complexity distribution
            print(f"\n   Distribution:")
            print(f"     Simple (0-2): {len(complexity[complexity <= 2])} samples")
            print(f"     Medium (2-5): {len(complexity[(complexity > 2) & (complexity <= 5)])} samples")
            print(f"     Complex (5+): {len(complexity[complexity > 5])} samples")
        
        # Quality analysis
        if 'quality_score' in df.columns:
            quality = df['quality_score']
            print(f"\n⭐ QUALITY SCORE ANALYSIS:")
            print(f"   Range: {quality.min():.2f} - {quality.max():.2f}")
            print(f"   Mean: {quality.mean():.2f}")
            print(f"   Std: {quality.std():.2f}")
            
            # Quality distribution
            print(f"\n   Distribution:")
            print(f"     Low (0.6-0.7): {len(quality[(quality >= 0.6) & (quality < 0.7)])} samples")
            print(f"     Medium (0.7-0.8): {len(quality[(quality >= 0.7) & (quality < 0.8)])} samples")
            print(f"     High (0.8+): {len(quality[quality >= 0.8])} samples")
        
        # Augmentation info
        if 'is_augmented' in df.columns:
            augmented_count = df['is_augmented'].sum()
            total_count = len(df)
            print(f"\n🔄 DATA AUGMENTATION:")
            print(f"   Original samples: {total_count - augmented_count}")
            print(f"   Augmented samples: {augmented_count}")
            print(f"   Augmentation rate: {augmented_count/total_count*100:.1f}%")
        
        # CalamanCy usage
        if 'uses_calamancy' in df.columns:
            calamancy_count = df['uses_calamancy'].sum()
            total_count = len(df)
            print(f"\n🎯 CALAMANCY USAGE:")
            print(f"   CalamanCy processed: {calamancy_count}/{total_count}")
            print(f"   Usage rate: {calamancy_count/total_count*100:.1f}%")
        
        # Training readiness assessment
        print(f"\n🎯 TRAINING READINESS ASSESSMENT:")
        print(f"   ✅ Parallel data: {len(df)} Filipino-English pairs")
        print(f"   ✅ Enhanced features: {len([col for col in df.columns if col not in ['src', 'tgt']])} additional columns")
        
        if 'tagalog_complexity' in df.columns:
            print(f"   ✅ Curriculum learning: Complexity scores available for progressive training")
        
        if 'quality_score' in df.columns:
            high_quality = len(df[df['quality_score'] >= 0.8])
            print(f"   ✅ Quality filtering: {high_quality} high-quality samples for training")
        
        print(f"\n🚀 RECOMMENDATION: Dataset is ready for enhanced training!")
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: 'full_enhanced_parallel_corpus.csv' not found!")
        print("💡 Make sure you've run the batch processing first")
        return None
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None

if __name__ == "__main__":
    df = analyze_enhanced_dataset()
    if df is not None:
        print(f"\n✅ Analysis completed! Your enhanced dataset has {len(df)} samples ready for training.")
