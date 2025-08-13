# TweetTaglish - Twitter Data Extraction & Processing Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A powerful, automated tool for extracting and processing Twitter data using Apify's Twitter scraper with intelligent fallback methods. Perfect for researchers, data scientists, and developers working with Twitter datasets.

## �� Features

- **Fast Extraction**: Uses Apify's Twitter scraper for high-speed data collection
- **Intelligent Fallback**: Twitter API v2 fallback when Apify is unavailable
- **Batch Processing**: Efficiently handles thousands of tweets with progress tracking
- **Multiple Outputs**: CSV, JSON, and detailed statistics
- **Cross-Platform**: Windows batch files, PowerShell scripts, and direct Python execution
- **Smart Caching**: Translation and processing result caching
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 📋 Prerequisites

- **Python 3.8+** with pip
- **Apify Account** (free tier available)
- **Git** (for cloning the repository)

## ��️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rjsalinas/Moodeng-MT.git
cd Moodeng-MT
```

### 2. Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# OR
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Required: Your Apify API token
APIFY_TOKEN=your_apify_api_token_here

# Optional: Twitter API bearer token (for fallback)
X_BEARER_TOKEN=your_twitter_bearer_token_here

# Optional: Customize settings
EXTRACTION_BATCH_SIZE=50
LOG_LEVEL=INFO
```

## 🚀 Quick Start

### Option 1: Windows Batch File (Easiest)
```cmd
run_extractor.bat
```

### Option 2: PowerShell Script
```powershell
.\run_extractor.ps1
```

### Option 3: Direct Python Execution
```bash
python scripts/twitter_extractor.py
```

## 📁 Project Structure
