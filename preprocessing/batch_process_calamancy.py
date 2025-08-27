#!/usr/bin/env python3
"""
Batch Process Filipino-English Parallel Corpus with CalamanCy Enhanced Preprocessing

This script processes the entire dataset in manageable batches to:
- Monitor progress in real-time
- Handle errors gracefully
- Save intermediate results
- Provide detailed progress reporting
- Optimize memory usage

Usage:
    python batch_process_calamancy.py
"""

import pandas as pd
import logging
import time
import os
from pathlib import Path
from enhanced_preprocessing import EnhancedFilipinoPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, batch_size=500, save_batches=True, cleanup_batches=True):
        """
        Initialize the batch processor.
        
        Args:
            batch_size (int): Number of samples to process per batch
            save_batches (bool): Whether to save individual batches as backup
            cleanup_batches (bool): Whether to clean up batch files after completion
        """
        self.batch_size = batch_size
        self.save_batches = save_batches
        self.cleanup_batches = cleanup_batches
        self.preprocessor = None
        self.start_time = None
        
    def initialize_preprocessor(self):
        """Initialize the CalamanCy enhanced preprocessor."""
        try:
            logger.info("🔧 Initializing CalamanCy Enhanced Preprocessor...")
            self.preprocessor = EnhancedFilipinoPreprocessor(use_calamancy=True)
            logger.info(f"✅ Preprocessor initialized successfully")
            logger.info(f"🔧 Using CalamanCy: {self.preprocessor.use_calamancy}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize preprocessor: {e}")
            return False
    
    def load_dataset(self, file_path='filipino_english_parallel_corpus.csv'):
        """Load and prepare the dataset."""
        try:
            logger.info(f"📁 Loading dataset: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"📋 Columns: {list(df.columns)}")
            
            # Check required columns
            if 'text' in df.columns and 'english_translation' in df.columns:
                # Rename columns to match preprocessing pipeline
                df = df.rename(columns={
                    'text': 'src',           # Original Filipino text
                    'english_translation': 'tgt'  # English translation
                })
                logger.info("✅ Columns renamed: text → src, english_translation → tgt")
            elif 'src' in df.columns and 'tgt' in df.columns:
                logger.info("✅ Dataset already has src/tgt columns")
            else:
                logger.error("❌ Cannot find suitable columns for preprocessing")
                logger.info(f"💡 Need either (text, english_translation) or (src, tgt)")
                return None
            
            # Show sample data
            logger.info("\n📝 Sample data (first 3 rows):")
            for i, row in df.head(3).iterrows():
                logger.info(f"Row {i+1}:")
                for col in ['src', 'tgt']:
                    if col in df.columns and pd.notna(row[col]):
                        text = str(row[col])
                        if len(text) > 100:
                            text = text[:100] + "..."
                        logger.info(f"  {col}: {text}")
                logger.info("")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"❌ Dataset file '{file_path}' not found!")
            logger.info("💡 Make sure the file is in the current directory")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return None
    
    def process_batch(self, batch_df, batch_num, total_batches):
        """Process a single batch of data."""
        try:
            logger.info(f"🚀 Processing Batch {batch_num + 1}/{total_batches}")
            logger.info(f"📊 Samples: {len(batch_df)} samples")
            
            # Process the batch
            batch_start = time.time()
            enhanced_batch = self.preprocessor.enhance_dataset(batch_df)
            batch_time = time.time() - batch_start
            
            # Save batch if requested
            if self.save_batches:
                batch_file = f'enhanced_batch_{batch_num + 1:03d}.csv'
                enhanced_batch.to_csv(batch_file, index=False)
                logger.info(f"💾 Batch saved: {batch_file}")
            
            logger.info(f"✅ Batch completed in {batch_time:.1f}s")
            return enhanced_batch, batch_time
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_num + 1} failed: {e}")
            return None, 0
    
    def process_dataset(self, df):
        """Process the entire dataset in batches."""
        if df is None or len(df) == 0:
            logger.error("❌ No data to process")
            return None
        
        # Calculate batch information
        total_samples = len(df)
        total_batches = (total_samples - 1) // self.batch_size + 1
        
        logger.info(f"🎯 Processing {total_samples} samples in {total_batches} batches")
        logger.info(f"📦 Batch size: {self.batch_size} samples")
        logger.info("=" * 60)
        
        # Initialize tracking variables
        self.start_time = time.time()
        all_enhanced_data = []
        successful_batches = 0
        failed_batches = 0
        
        # Process each batch
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            
            logger.info(f"\n🔄 Batch {batch_num + 1}/{total_batches}")
            logger.info(f"📍 Processing samples {start_idx + 1}-{end_idx}")
            
            # Get current batch
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            # Process the batch
            enhanced_batch, batch_time = self.process_batch(batch_df, batch_num, total_batches)
            
            if enhanced_batch is not None:
                all_enhanced_data.append(enhanced_batch)
                successful_batches += 1
                
                # Progress update
                processed_samples = len(all_enhanced_data) * self.batch_size
                elapsed_time = time.time() - self.start_time
                avg_speed = processed_samples / elapsed_time if elapsed_time > 0 else 0
                
                logger.info(f"📈 Progress: {processed_samples}/{total_samples} samples ({processed_samples/total_samples*100:.1f}%)")
                logger.info(f"⚡ Average speed: {avg_speed:.1f} samples/second")
                logger.info(f"⏱️  Total elapsed time: {elapsed_time/60:.1f} minutes")
                
                # Estimate remaining time
                if avg_speed > 0:
                    remaining_samples = total_samples - processed_samples
                    estimated_remaining = remaining_samples / avg_speed
                    logger.info(f"⏳ Estimated time remaining: {estimated_remaining/60:.1f} minutes")
            else:
                failed_batches += 1
                logger.warning(f"⚠️  Batch {batch_num + 1} failed, continuing...")
        
        # Final summary
        self._print_final_summary(total_samples, successful_batches, failed_batches, all_enhanced_data)
        
        return all_enhanced_data
    
    def _print_final_summary(self, total_samples, successful_batches, total_batches, all_enhanced_data):
        """Print final processing summary."""
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 BATCH PROCESSING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"📊 Processing Summary:")
        logger.info(f"  ✅ Total samples: {total_samples}")
        logger.info(f"  ✅ Successful batches: {successful_batches}/{total_batches}")
        logger.info(f"  ⏱️  Total processing time: {total_time/60:.1f} minutes")
        
        if all_enhanced_data:
            final_count = sum(len(batch) for batch in all_enhanced_data)
            overall_speed = final_count / total_time if total_time > 0 else 0
            
            logger.info(f"  📈 Final sample count: {final_count}")
            logger.info(f"  ⚡ Overall speed: {overall_speed:.1f} samples/second")
            
            # Show new columns from enhanced data
            if all_enhanced_data:
                sample_batch = all_enhanced_data[0]
                logger.info(f"  🆕 New columns: {list(sample_batch.columns)}")
        
        logger.info("=" * 60)
    
    def combine_and_save(self, all_enhanced_data, output_file='full_enhanced_parallel_corpus.csv'):
        """Combine all batches and save the final dataset."""
        if not all_enhanced_data:
            logger.error("❌ No enhanced data to combine")
            return False
        
        try:
            logger.info("🔗 Combining all batches...")
            final_enhanced_df = pd.concat(all_enhanced_data, ignore_index=True)
            
            # Save final combined dataset
            final_enhanced_df.to_csv(output_file, index=False)
            logger.info(f"💾 Final dataset saved: {output_file}")
            logger.info(f"📊 Final dataset shape: {final_enhanced_df.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error combining batches: {e}")
            return False
    
    def cleanup_batch_files(self, total_batches):
        """Clean up individual batch files."""
        if not self.cleanup_batches:
            return
        
        logger.info("🧹 Cleaning up batch files...")
        cleaned_count = 0
        
        for i in range(total_batches):
            batch_file = f'enhanced_batch_{i + 1:03d}.csv'
            if os.path.exists(batch_file):
                try:
                    os.remove(batch_file)
                    cleaned_count += 1
                    logger.info(f"  🗑️  Cleaned up: {batch_file}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not remove {batch_file}: {e}")
        
        logger.info(f"✅ Cleaned up {cleaned_count} batch files")

def main():
    """Main function to run the batch processing."""
    logger.info("🚀 Starting CalamanCy Batch Processing")
    logger.info("=" * 60)
    
    # Configuration
    BATCH_SIZE = 500
    INPUT_FILE = 'filipino_english_parallel_corpus.csv'
    OUTPUT_FILE = 'full_enhanced_parallel_corpus.csv'
    
    try:
        # Initialize processor
        processor = BatchProcessor(
            batch_size=BATCH_SIZE,
            save_batches=True,
            cleanup_batches=True
        )
        
        # Initialize preprocessor
        if not processor.initialize_preprocessor():
            logger.error("❌ Failed to initialize preprocessor. Exiting.")
            return
        
        # Load dataset
        df = processor.load_dataset(INPUT_FILE)
        if df is None:
            logger.error("❌ Failed to load dataset. Exiting.")
            return
        
        # Process dataset in batches
        all_enhanced_data = processor.process_dataset(df)
        
        if all_enhanced_data:
            # Combine and save final dataset
            if processor.combine_and_save(all_enhanced_data, OUTPUT_FILE):
                logger.info("✅ Batch processing completed successfully!")
                
                # Clean up batch files
                total_batches = (len(df) - 1) // BATCH_SIZE + 1
                processor.cleanup_batch_files(total_batches)
                
                logger.info(f"\n🎯 Final output: {OUTPUT_FILE}")
                logger.info("🎉 You can now use this enhanced dataset for training!")
            else:
                logger.error("❌ Failed to save final dataset")
        else:
            logger.error("❌ No data was processed successfully")
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main()
