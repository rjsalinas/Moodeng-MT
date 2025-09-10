#!/usr/bin/env python3
"""
Test CalamanCy Enhanced Preprocessing with Real Filipino-English Dataset

This script tests the enhanced preprocessing pipeline with the actual dataset
to see how it performs on real Filipino text data.
"""

import pandas as pd
import logging
from enhanced_preprocessing import EnhancedFilipinoPreprocessor
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_dataset():
    """Test the enhanced preprocessing with the real dataset."""
    
    logger.info("🚀 Starting Real Dataset CalamanCy Test")
    logger.info("=" * 60)
    
    # Load the real dataset
    try:
        logger.info("📁 Loading real dataset...")
        df = pd.read_csv('tweets_id_filipino_text_normalized.csv')
        logger.info(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Display dataset info
        logger.info(f"📊 Dataset shape: {df.shape}")
        logger.info(f"📋 Columns: {list(df.columns)}")
        
        # Show sample data
        logger.info("\n📝 Sample data (first 3 rows):")
        for i, row in df.head(3).iterrows():
            logger.info(f"Row {i+1}:")
            for col in df.columns:
                if col in ['src', 'tgt'] and pd.notna(row[col]):
                    # Truncate long text for display
                    text = str(row[col])
                    if len(text) > 100:
                        text = text[:100] + "..."
                    logger.info(f"  {col}: {text}")
                else:
                    logger.info(f"  {col}: {row[col]}")
            logger.info("")
        
    except FileNotFoundError:
        logger.error("❌ Dataset file 'tweets_id_filipino_text_normalized.csv' not found!")
        logger.info("💡 Make sure the file is in the current directory")
        return
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return
    
    # Check if we have the required columns and adapt if needed
    if 'src' in df.columns and 'tgt' in df.columns:
        # Dataset already has src/tgt columns
        logger.info("✅ Dataset has required src/tgt columns")
    elif 'text' in df.columns and 'preprocessed_text' in df.columns:
        # Adapt the dataset to work with our preprocessing
        logger.info("🔄 Adapting dataset columns for preprocessing...")
        df = df.rename(columns={
            'text': 'src',
            'preprocessed_text': 'tgt'
        })
        logger.info("✅ Columns renamed: text → src, preprocessed_text → tgt")
    else:
        logger.error("❌ Cannot find suitable columns for preprocessing")
        logger.info(f"💡 Available columns: {list(df.columns)}")
        logger.info("💡 Need either (src, tgt) or (text, preprocessed_text)")
        return
    
    # Clean the data
    logger.info("🧹 Cleaning dataset...")
    initial_count = len(df)
    
    # Remove rows with missing source or target
    df_clean = df.dropna(subset=['src', 'tgt'])
    
    # Remove empty strings
    df_clean = df_clean[(df_clean['src'].str.strip() != '') & (df_clean['tgt'].str.strip() != '')]
    
    # Remove very short texts (likely noise)
    df_clean = df_clean[(df_clean['src'].str.len() > 3) & (df_clean['tgt'].str.len() > 3)]
    
    # Check for Filipino text indicators
    filipino_indicators = ['ng', 'sa', 'ang', 'ay', 'na', 'pa', 'di', 'ba', 'ko', 'mo', 'ni', 'nila', 'kasi', 'gusto', 'kailangan']
    has_filipino = df_clean['src'].str.lower().str.contains('|'.join(filipino_indicators), na=False)
    df_clean = df_clean[has_filipino]
    
    cleaned_count = len(df_clean)
    logger.info(f"✅ Data cleaning: {initial_count} → {cleaned_count} samples")
    logger.info(f"🎯 Filipino text filtering: {len(df_clean[has_filipino])} samples contain Filipino indicators")
    
    if cleaned_count == 0:
        logger.error("❌ No valid samples after cleaning!")
        return
    
    # Take a subset for testing (to avoid long processing time)
    test_size = min(50, cleaned_count)  # Test with up to 50 samples
    df_test = df_clean.head(test_size).copy()
    logger.info(f"🧪 Testing with {test_size} samples")
    
    # Initialize the enhanced preprocessor
    logger.info("🔧 Initializing CalamanCy Enhanced Preprocessor...")
    start_time = time.time()
    
    try:
        preprocessor = EnhancedFilipinoPreprocessor(use_calamancy=True)
        init_time = time.time() - start_time
        logger.info(f"✅ Preprocessor initialized in {init_time:.2f}s")
        logger.info(f"🔧 Using CalamanCy: {preprocessor.use_calamancy}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize preprocessor: {e}")
        return
    
    # Test the enhanced preprocessing
    logger.info("\n🚀 Starting Enhanced Preprocessing...")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        enhanced_df = preprocessor.enhance_dataset(df_test)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Enhanced preprocessing completed in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Enhanced preprocessing failed: {e}")
        logger.info("💡 Check the error details above")
        return
    
    # Analyze the results
    logger.info("\n📊 ENHANCED DATASET ANALYSIS")
    logger.info("=" * 60)
    
    logger.info(f"📈 Sample count: {len(enhanced_df)}")
    logger.info(f"📋 Columns: {list(enhanced_df.columns)}")
    
    # Complexity analysis
    if 'tagalog_complexity' in enhanced_df.columns:
        complexity_stats = enhanced_df['tagalog_complexity'].describe()
        logger.info(f"🎯 Tagalog Complexity Statistics:")
        logger.info(f"  Min: {complexity_stats['min']:.2f}")
        logger.info(f"  Max: {complexity_stats['max']:.2f}")
        logger.info(f"  Mean: {complexity_stats['mean']:.2f}")
        logger.info(f"  Std: {complexity_stats['std']:.2f}")
    
    # Quality analysis
    if 'quality_score' in enhanced_df.columns:
        quality_stats = enhanced_df['quality_score'].describe()
        logger.info(f"⭐ Quality Score Statistics:")
        logger.info(f"  Min: {quality_stats['min']:.2f}")
        logger.info(f"  Max: {quality_stats['max']:.2f}")
        logger.info(f"  Mean: {quality_stats['mean']:.2f}")
        logger.info(f"  Std: {quality_stats['std']:.2f}")
    
    # CalamanCy usage
    if 'uses_calamancy' in enhanced_df.columns:
        calamancy_usage = enhanced_df['uses_calamancy'].sum()
        total_samples = len(enhanced_df)
        logger.info(f"🎯 CalamanCy Usage: {calamancy_usage}/{total_samples} samples ({calamancy_usage/total_samples*100:.1f}%)")
    
    # Show sample enhanced data
    logger.info("\n📝 Sample Enhanced Data (first 5 rows):")
    display_columns = ['src', 'tgt', 'tagalog_complexity', 'quality_score', 'is_augmented']
    available_columns = [col for col in display_columns if col in enhanced_df.columns]
    
    for i, row in enhanced_df.head(5).iterrows():
        logger.info(f"Row {i+1}:")
        for col in available_columns:
            if col in ['src', 'tgt'] and pd.notna(row[col]):
                # Truncate long text for display
                text = str(row[col])
                if len(text) > 80:
                    text = text[:80] + "..."
                logger.info(f"  {col}: {text}")
            else:
                logger.info(f"  {col}: {row[col]}")
        logger.info("")
    
    # Save the enhanced dataset
    output_file = 'enhanced_filipino_dataset.csv'
    try:
        enhanced_df.to_csv(output_file, index=False)
        logger.info(f"💾 Enhanced dataset saved to: {output_file}")
    except Exception as e:
        logger.warning(f"⚠️  Could not save enhanced dataset: {e}")
    
    # Performance summary
    logger.info("\n📈 PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"⏱️  Initialization time: {init_time:.2f}s")
    logger.info(f"⚡ Processing time: {processing_time:.2f}s")
    logger.info(f"📊 Processing speed: {len(enhanced_df)/processing_time:.1f} samples/second")
    
    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS")
    logger.info("=" * 60)
    
    if preprocessor.use_calamancy:
        logger.info("✅ CalamanCy is working correctly!")
        logger.info("🎯 You can now use this enhanced preprocessing in your training pipeline")
        logger.info("📚 The enhanced dataset includes Tagalog-specific linguistic features")
    else:
        logger.warning("⚠️  CalamanCy is not available, using spaCy fallback")
        logger.info("💡 Consider installing CalamanCy for better Tagalog analysis")
    
    if 'tagalog_complexity' in enhanced_df.columns and enhanced_df['tagalog_complexity'].max() > 0:
        logger.info("🎯 Tagalog complexity analysis is working")
        logger.info("📊 You can use complexity scores for curriculum learning")
    
    if 'quality_score' in enhanced_df.columns:
        avg_quality = enhanced_df['quality_score'].mean()
        if avg_quality > 0.7:
            logger.info(f"⭐ Good quality scores (avg: {avg_quality:.2f})")
        else:
            logger.warning(f"⚠️  Low quality scores (avg: {avg_quality:.2f}) - may need data cleaning")
    
    logger.info("\n🎉 Real dataset test completed successfully!")
    return enhanced_df

if __name__ == "__main__":
    try:
        enhanced_data = test_real_dataset()
        if enhanced_data is not None:
            print(f"\n✅ Test completed! Enhanced dataset has {len(enhanced_data)} samples")
        else:
            print("\n❌ Test failed! Check the logs above for details")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
