@echo off
echo Installing requirements for Filipino-to-English Translation Project...
echo.

echo Step 1: Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Step 2: Installing all other requirements...
pip install -r requirements.txt

echo.
echo Step 3: Installing spaCy language models...
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm

echo.
echo Step 4: Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo.
echo Installation completed!
echo You can now run: python test_environment.py
pause
