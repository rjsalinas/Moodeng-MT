#!/usr/bin/env python3
"""
Configuration file for TweetTaglish processing tools
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Input files
TWITTER_LINKS_FILE = BASE_DIR / "twitter_links.txt"

# Output directories
OUTPUT_DIR = DATA_DIR / "output"
EXTRACTED_TWEETS_DIR = DATA_DIR / "extracted_tweets"
PROCESSED_TWEETS_DIR = DATA_DIR / "processed_tweets"
AI_PREPROCESSED_DIR = DATA_DIR / "ai_preprocessed"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, EXTRACTED_TWEETS_DIR, PROCESSED_TWEETS_DIR, AI_PREPROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "").strip()
TWITTER_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()

# Apify Configuration
APIFY_ACTOR_ID = "VsTreSuczsXhhRIqa"  # Twitter scraper actor
APIFY_BASE_URL = "https://api.apify.com/v2"

# Twitter API Configuration
TWITTER_API_BASE_URL = "https://api.twitter.com/2"
TWITTER_API_VERSION = "v2"

# Extraction Configuration
EXTRACTION_BATCH_SIZE = int(os.getenv("EXTRACTION_BATCH_SIZE", "50"))
EXTRACTION_MAX_RETRIES = int(os.getenv("EXTRACTION_MAX_RETRIES", "3"))
EXTRACTION_RETRY_DELAY = float(os.getenv("EXTRACTION_RETRY_DELAY", "5.0"))
EXTRACTION_RATE_LIMIT_DELAY = float(os.getenv("EXTRACTION_RATE_LIMIT_DELAY", "1.0"))

# Processing Configuration
PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", "20"))
PROCESSING_DELAY = float(os.getenv("PROCESSING_DELAY", "2.0"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "twitter_extraction.log"

# File patterns
CSV_PATTERN = "*.csv"
JSON_PATTERN = "*.json"
LOG_PATTERN = "*.log"

# Validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if twitter_links.txt exists
    if not TWITTER_LINKS_FILE.exists():
        errors.append(f"Twitter links file not found: {TWITTER_LINKS_FILE}")
    
    # Check if at least one API token is provided
    if not APIFY_TOKEN and not TWITTER_BEARER_TOKEN:
        errors.append("No API tokens provided. Set either APIFY_TOKEN or X_BEARER_TOKEN")
    
    # Check if directories are writable
    for directory in [OUTPUT_DIR, EXTRACTED_TWEETS_DIR, PROCESSED_TWEETS_DIR]:
        if not os.access(directory, os.W_OK):
            errors.append(f"Directory not writable: {directory}")
    
    return errors

def print_config():
    """Print current configuration"""
    print("TweetTaglish Processing Configuration")
    print("=" * 50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Scripts Directory: {SCRIPTS_DIR}")
    print(f"Twitter Links File: {TWITTER_LINKS_FILE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Extracted Tweets Directory: {EXTRACTED_TWEETS_DIR}")
    print(f"Processed Tweets Directory: {PROCESSED_TWEETS_DIR}")
    print(f"AI Preprocessed Directory: {AI_PREPROCESSED_DIR}")
    print()
    print("API Configuration:")
    print(f"  Apify Token: {'✓ Set' if APIFY_TOKEN else '✗ Not set'}")
    print(f"  Twitter Bearer Token: {'✓ Set' if TWITTER_BEARER_TOKEN else '✗ Not set'}")
    print(f"  Apify Actor ID: {APIFY_ACTOR_ID}")
    print()
    print("Extraction Configuration:")
    print(f"  Batch Size: {EXTRACTION_BATCH_SIZE}")
    print(f"  Max Retries: {EXTRACTION_MAX_RETRIES}")
    print(f"  Retry Delay: {EXTRACTION_RETRY_DELAY}s")
    print(f"  Rate Limit Delay: {EXTRACTION_RATE_LIMIT_DELAY}s")
    print()
    print("Processing Configuration:")
    print(f"  Batch Size: {PROCESSING_BATCH_SIZE}")
    print(f"  Delay: {PROCESSING_DELAY}s")
    print()
    print("Logging Configuration:")
    print(f"  Log Level: {LOG_LEVEL}")
    print(f"  Log File: {LOG_FILE}")

if __name__ == "__main__":
    print_config()
    print()
    
    errors = validate_config()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("✓ Configuration is valid")
