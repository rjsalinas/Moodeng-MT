#!/usr/bin/env python3
"""
Test script to verify the TweetTaglish extraction setup
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import aiohttp
        print("✓ aiohttp imported successfully")
    except ImportError as e:
        print(f"✗ aiohttp import failed: {e}")
        return False
    
    try:
        import pandas
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import apify_client
        print("✓ apify-client imported successfully")
    except ImportError as e:
        print(f"✗ apify-client import failed: {e}")
        print("  Install with: pip install apify-client")
        return False
    
    try:
        import numpy
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    
    try:
        from config import validate_config, print_config
        print("✓ config.py imported successfully")
        
        # Print configuration
        print_config()
        
        # Validate configuration
        errors = validate_config()
        if errors:
            print("\nConfiguration Errors:")
            for error in errors:
                print(f"  ✗ {error}")
            return False
        else:
            print("\n✓ Configuration is valid")
            return True
            
    except Exception as e:
        print(f"✗ config.py test failed: {e}")
        return False

def test_file_structure():
    """Test if required files and directories exist"""
    print("\nTesting file structure...")
    
    base_dir = Path(__file__).parent.parent
    
    # Check required files
    required_files = [
        "twitter_links.txt",
        "requirements.txt",
        "README.md"
    ]
    
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name} found")
        else:
            print(f"✗ {file_name} not found")
            return False
    
    # Check required directories
    required_dirs = [
        "scripts",
        "data"
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ {dir_name}/ directory found")
        else:
            print(f"✗ {dir_name}/ directory not found")
            return False
    
    return True

def test_environment():
    """Test environment variables"""
    print("\nTesting environment variables...")
    
    apify_token = os.getenv("APIFY_TOKEN", "").strip()
    twitter_token = os.getenv("X_BEARER_TOKEN", "").strip()
    
    if apify_token:
        print("✓ APIFY_TOKEN is set")
    else:
        print("⚠ APIFY_TOKEN is not set (required for Apify extraction)")
    
    if twitter_token:
        print("✓ X_BEARER_TOKEN is set (optional, for Twitter API fallback)")
    else:
        print("⚠ X_BEARER_TOKEN is not set (optional)")
    
    # At least one token should be set
    if not apify_token and not twitter_token:
        print("✗ No API tokens found. Set at least APIFY_TOKEN")
        return False
    
    return True

def test_twitter_links():
    """Test twitter_links.txt file"""
    print("\nTesting twitter_links.txt...")
    
    base_dir = Path(__file__).parent.parent
    links_file = base_dir / "twitter_links.txt"
    
    if not links_file.exists():
        print("✗ twitter_links.txt not found")
        return False
    
    try:
        with open(links_file, 'r', encoding='utf-8') as f:
            links = [line.strip() for line in f if line.strip()]
        
        print(f"✓ Found {len(links)} Twitter links")
        
        # Check first few links
        if links:
            print("Sample links:")
            for i, link in enumerate(links[:3]):
                print(f"  {i+1}. {link}")
            
            if len(links) > 3:
                print(f"  ... and {len(links) - 3} more")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading twitter_links.txt: {e}")
        return False

def main():
    """Main test function"""
    print("TweetTaglish Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Configuration", test_config),
        ("Environment Variables", test_environment),
        ("Twitter Links File", test_twitter_links)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Set your APIFY_TOKEN environment variable")
        print("2. Run: python scripts/twitter_extractor.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Set environment variables: export APIFY_TOKEN='your_token'")
        print("3. Ensure twitter_links.txt exists in the project root")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
