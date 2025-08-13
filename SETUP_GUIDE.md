# TweetTaglish Setup Guide

## Problems Identified and Solutions

### ❌ Problem 1: Missing Data Directory
**Issue**: The system expects a `data/` directory structure that doesn't exist.

**Solution**: ✅ **FIXED** - The required directories have been created:
- `data/`
- `data/output/`
- `data/extracted_tweets/`
- `data/processed_tweets/`
- `data/ai_preprocessed/`

### ❌ Problem 2: Missing API Tokens
**Issue**: No `APIFY_TOKEN` or `X_BEARER_TOKEN` environment variables are set.

**Solution**: You need to set these environment variables. Two options:

#### Option A: Use the Setup Scripts (Recommended)
1. **For PowerShell users**: Run `.\setup_env.ps1`
2. **For Command Prompt users**: Run `setup_env.bat`
3. Edit the script to replace placeholder tokens with your actual API tokens
4. Run the script again to activate the tokens

#### Option B: Set Environment Variables Manually
```powershell
# PowerShell
$env:APIFY_TOKEN = "your_actual_apify_token"
$env:X_BEARER_TOKEN = "your_actual_twitter_token"

# Command Prompt
set APIFY_TOKEN=your_actual_apify_token
set X_BEARER_TOKEN=your_actual_twitter_token
```

### ❌ Problem 3: Configuration Validation Fails
**Issue**: The system can't proceed without proper API authentication.

**Solution**: Set the API tokens as described above, then run the test again.

## How to Get API Tokens

### 1. Apify Token (Required)
- Go to [Apify Console](https://console.apify.com/account/integrations)
- Sign up/Login to your account
- Navigate to Integrations → API tokens
- Copy your API token

### 2. Twitter Bearer Token (Optional)
- Go to [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
- Apply for a developer account if you don't have one
- Create a new app
- Get your Bearer Token from the app settings

## Complete Setup Steps

1. ✅ **Virtual Environment**: Already created and activated
2. ✅ **Dependencies**: Already installed via `pip install -r requirements.txt`
3. ✅ **Data Directories**: Already created
4. ⚠️ **API Tokens**: **YOU NEED TO SET THESE**
5. ✅ **File Structure**: All required files are present

## Testing Your Setup

After setting the API tokens:

```bash
# Activate virtual environment (if not already active)
.venv\Scripts\activate

# Run the setup test
python scripts\test_setup.py
```

## Expected Results

When properly configured, you should see:
- ✅ All tests passing
- ✅ Configuration is valid
- ✅ API tokens are set
- ✅ File structure is correct

## Troubleshooting

- **"No API tokens found"**: Set the `APIFY_TOKEN` environment variable
- **"Directory not found"**: The data directories should now exist
- **"Package import errors"**: Make sure you're in the virtual environment and dependencies are installed

## Next Steps

Once the setup test passes:
1. You can run the main extraction script: `python scripts\twitter_extractor.py`
2. Or use the automation scripts: `.\run_extractor.ps1` or `run_extractor.bat`

## Support

If you continue to have issues:
1. Check that your virtual environment is activated
2. Verify API tokens are correctly set
3. Ensure all dependencies are installed
4. Check the log files in the project root for detailed error messages
