#!/usr/bin/env python3
"""
Script to clean up large files and prepare repository for Git.
This script helps manage large files that cause GitHub issues.
"""

import os
import shutil
from pathlib import Path

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def find_large_files(directory=".", threshold_mb=10):
    """Find files larger than threshold."""
    large_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip .git directory
        if '.git' in root:
            continue
            
        for file in files:
            file_path = Path(root) / file
            try:
                size_mb = get_file_size_mb(file_path)
                if size_mb > threshold_mb:
                    large_files.append((file_path, size_mb))
            except (OSError, FileNotFoundError):
                continue
    
    return sorted(large_files, key=lambda x: x[1], reverse=True)

def create_backup_directory():
    """Create backup directory for large files."""
    backup_dir = Path("large_files_backup")
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def backup_large_files(large_files, backup_dir):
    """Move large files to backup directory."""
    backed_up = []
    
    for file_path, size_mb in large_files:
        try:
            # Create subdirectories in backup if needed
            relative_path = file_path.relative_to(Path("."))
            backup_path = backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file to backup
            shutil.move(str(file_path), str(backup_path))
            backed_up.append((file_path, backup_path, size_mb))
            print(f"Backed up: {file_path} -> {backup_path} ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"Failed to backup {file_path}: {e}")
    
    return backed_up

def create_data_instructions():
    """Create instructions for handling large data files."""
    instructions = """# Large Data Files Management

This directory contains large data files that were moved to prevent GitHub issues.

## Files Backed Up:
- **Large CSV files**: Very large tweet datasets (>10MB)
- **JSON files**: Raw tweet data from extraction
- **Model files**: Fine-tuned translation models
- **Log files**: Processing logs

## Important CSV Files Kept in Repository:
- `tweets_id_filipino_text_only.csv`: Filipino/Taglish tweets (important for research)
- `tweets_id_filipino_text_normalized.csv`: Normalized Filipino tweets (research results)
- `tweets_id_non_fil_tag_taglish.csv`: Non-Filipino tweets (analysis data)

## Large Files Moved to Backup:
- `tweets_id_text_only.csv`: Raw extracted data (very large)
- `tweets_id_text_normalized.csv`: Full normalized dataset (very large)
- `english_translation_from_preprocessed_texts.csv`: Translation outputs (large)
- `dataset_*.json`: Raw JSON datasets (very large)

## To Use These Files:
1. **For Research**: Use the kept CSV files directly
2. **For Processing**: Copy large files back when needed
3. **For Analysis**: The important results are already in the repository

## Recommended Workflow:
1. **Keep in Git**: Important processed results and analysis data
2. **Backup**: Very large raw datasets and intermediate files
3. **Share**: Large files via cloud storage or data repositories
4. **Research**: Use the kept CSV files for your thesis analysis

## File Types to Keep in Git:
- Python scripts (*.py)
- Configuration files (rules.json, requirements.txt)
- Documentation (*.md)
- **Important CSV results** (Filipino tweets, normalized data)
- Batch files (*.bat, *.ps1)

## File Types to Exclude from Git:
- Very large datasets (>10MB)
- Raw extraction files
- Model weights (*.safetensors, *.bin)
- Log files (*.log, *.jsonl)
- Temporary processing outputs
"""
    
    with open("large_files_backup/README.md", "w", encoding="utf-8") as f:
        f.write(instructions)

def main():
    """Main cleanup function."""
    print("🔍 Scanning for large files...")
    
    # Find large files (>10MB)
    large_files = find_large_files(threshold_mb=10)
    
    if not large_files:
        print("✅ No large files found!")
        return
    
    print(f"\n📁 Found {len(large_files)} large files:")
    for file_path, size_mb in large_files:
        print(f"  {file_path} ({size_mb:.1f} MB)")
    
    # Ask for confirmation
    total_size = sum(size for _, size in large_files)
    print(f"\n📊 Total size: {total_size:.1f} MB")
    
    print("\n💡 Strategy:")
    print("  - Keep important CSV results in repository")
    print("  - Move very large files to backup")
    print("  - Preserve research data for your thesis")
    
    response = input("\n❓ Do you want to backup these large files? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    # Create backup directory
    backup_dir = create_backup_directory()
    
    # Backup large files
    print("\n🔄 Backing up large files...")
    backed_up = backup_large_files(large_files, backup_dir)
    
    # Create instructions
    create_data_instructions()
    
    print(f"\n✅ Cleanup complete!")
    print(f"📁 Large files backed up to: {backup_dir}")
    print(f"📝 Instructions created: {backup_dir}/README.md")
    print(f"🔢 Files backed up: {len(backed_up)}")
    
    # Show what to do next
    print("\n📋 Next steps:")
    print("1. Commit the .gitignore changes")
    print("2. Add and commit your Python scripts")
    print("3. Large files are now in large_files_backup/")
    print("4. Important CSV results remain in repository")
    print("5. Use log_manager.py for future log rotation")
    
    print("\n🎯 Research Data Preserved:")
    print("  ✅ tweets_id_filipino_text_only.csv")
    print("  ✅ tweets_id_filipino_text_normalized.csv") 
    print("  ✅ tweets_id_non_fil_tag_taglish.csv")

if __name__ == "__main__":
    main()
