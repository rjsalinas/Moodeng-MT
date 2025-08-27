import pandas as pd
import os

def test_excel_structure():
    """Test the Excel file structure before preprocessing"""
    excel_file = 'tweets_split_id.xlsx'
    worksheet_name = 'tweets_split_id'
    
    print("🔍 Testing Excel File Structure")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"❌ File not found: {excel_file}")
        return False
    
    try:
        # Read the Excel file
        print(f"📁 Reading Excel file: {excel_file}")
        xl = pd.ExcelFile(excel_file)
        print(f"📚 Worksheets found: {xl.sheet_names}")
        
        # Check if required worksheet exists
        if worksheet_name not in xl.sheet_names:
            print(f"❌ Required worksheet '{worksheet_name}' not found!")
            return False
        
        # Read the specific worksheet
        print(f"\n📖 Reading worksheet: {worksheet_name}")
        df = pd.read_excel(excel_file, sheet_name=worksheet_name)
        print(f"📊 Total rows: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['Tweet Status', 'Original Taglish Tweet', 'Preprocessed Taglish Tweet']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print(f"✅ All required columns found!")
        
        # Analyze Tweet Status values
        print(f"\n📊 Tweet Status Analysis:")
        status_counts = df['Tweet Status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} rows")
        
        # Check valid tweets (Status = 1)
        valid_tweets = df[df['Tweet Status'] == 1]
        print(f"\n✅ Valid tweets (Status = 1): {len(valid_tweets)} rows")
        
        if len(valid_tweets) > 0:
            print(f"📝 Sample valid tweet:")
            sample = valid_tweets.iloc[0]
            print(f"  Original: {sample['Original Taglish Tweet'][:100]}...")
        
        # Check for NaN values in required columns
        print(f"\n🔍 Data Quality Check:")
        for col in ['Tweet Status', 'Original Taglish Tweet']:
            nan_count = df[col].isna().sum()
            print(f"  {col}: {nan_count} NaN values")
        
        print(f"\n✅ Excel file structure is valid and ready for preprocessing!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return False

if __name__ == "__main__":
    success = test_excel_structure()
    if success:
        print(f"\n🚀 Ready to run preprocessing!")
        print(f"Run: python preprocess_tweets.py")
    else:
        print(f"\n⚠️  Please fix the issues before running preprocessing.")
