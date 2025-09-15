import os
import json
import logging
from datetime import datetime
from pathlib import Path

class LogManager:
    """
    Manages log files with rotation to prevent them from growing too large.
    """
    
    def __init__(self, log_dir="logs", max_file_size_mb=10, max_files=5):
        """
        Initialize log manager.
        
        Args:
            log_dir (str): Directory to store log files
            max_file_size_mb (int): Maximum size of each log file in MB
            max_files (int): Maximum number of log files to keep
        """
        self.log_dir = Path(log_dir)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)
        
        # Current log file
        self.current_log_file = self.log_dir / "normalization_log.jsonl"
        
    def get_log_file_path(self):
        """Get the current log file path."""
        return str(self.current_log_file)
    
    def should_rotate(self):
        """Check if log rotation is needed."""
        if not self.current_log_file.exists():
            return False
        
        return self.current_log_file.stat().st_size > self.max_file_size_bytes
    
    def rotate_logs(self):
        """Rotate log files when they get too large."""
        if not self.should_rotate():
            return
        
        # Create timestamp for the old log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_log_name = f"normalization_log_{timestamp}.jsonl"
        old_log_path = self.log_dir / old_log_name
        
        # Rename current log
        if self.current_log_file.exists():
            self.current_log_file.rename(old_log_path)
        
        # Clean up old logs if we have too many
        self._cleanup_old_logs()
        
        # Create new empty log file
        self.current_log_file.touch()
        
        print(f"Log rotated: {old_log_name}")
    
    def _cleanup_old_logs(self):
        """Remove old log files, keeping only the most recent ones."""
        log_files = list(self.log_dir.glob("normalization_log_*.jsonl"))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove excess files
        for old_log in log_files[self.max_files:]:
            try:
                old_log.unlink()
                print(f"Removed old log: {old_log.name}")
            except Exception as e:
                print(f"Failed to remove old log {old_log.name}: {e}")
    
    def write_log_entry(self, log_entry):
        """
        Write a log entry to the current log file.
        
        Args:
            log_entry (dict): Log entry to write
        """
        # Check if rotation is needed
        self.rotate_logs()
        
        # Write the log entry
        try:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Failed to write log entry: {e}")
    
    def get_log_stats(self):
        """Get statistics about log files."""
        stats = {
            "current_log_size_mb": 0,
            "total_log_files": 0,
            "total_size_mb": 0
        }
        
        if self.current_log_file.exists():
            stats["current_log_size_mb"] = round(
                self.current_log_file.stat().st_size / (1024 * 1024), 2
            )
        
        # Count all log files
        all_logs = list(self.log_dir.glob("*.jsonl"))
        stats["total_log_files"] = len(all_logs)
        
        # Calculate total size
        total_size = sum(log_file.stat().st_size for log_file in all_logs)
        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats
    
    def cleanup_all_logs(self):
        """Remove all log files (use with caution)."""
        log_files = list(self.log_dir.glob("*.jsonl"))
        removed_count = 0
        
        for log_file in log_files:
            try:
                log_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {log_file.name}: {e}")
        
        print(f"Removed {removed_count} log files")
        return removed_count

# Example usage
if __name__ == "__main__":
    # Initialize log manager
    log_manager = LogManager(max_file_size_mb=5, max_files=3)
    
    # Show current stats
    stats = log_manager.get_log_stats()
    print("Current log statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example log entry
    sample_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": "Log manager initialized",
        "file_size_limit_mb": 5
    }
    
    log_manager.write_log_entry(sample_entry)
    print(f"\nLog entry written to: {log_manager.get_log_file_path()}")
