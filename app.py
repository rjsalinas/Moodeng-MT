# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import re
# # import emoji  # Commented out due to installation issues  # Commented out due to installation issues

# ======================
# Text Cleaning Function
# ======================
def preprocess_text(text: str) -> str:
    """
    Applies the same cleaning steps used during model training.
    - Removes URLs, mentions, emojis.
    - Replaces hashtags with the tag text.
    - Normalizes whitespace.
    """
    # Ensure text is a string and strip leading/trailing whitespace
    s = str(text).strip()

    # Define regex patterns
    url_pattern = r'(https?://\S+|www\.\S+)'
    mention_pattern = r'@[A-Za-z0-9_]+'
    hashtag_pattern = r'#([A-Za-z0-9_]+)'

    # Apply cleaning steps
    s = re.sub(url_pattern, ' ', s)          # Remove URLs
    s = re.sub(mention_pattern, ' ', s)      # Remove mentions
    s = re.sub(hashtag_pattern, r'\1', s)   # Keep hashtag text, remove '#'
    # Remove emojis using regex (alternative to emoji library)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    s = emoji_pattern.sub(' ', s)  # Remove emojis

    # Replace multiple whitespace chars with a single space and strip again
    s = re.sub(r'\s+', ' ', s).strip()

    return s

# ======================
# Load Fine-tuned Model
# ======================
@st.cache_resource
def load_model():
    model_path = "./LoRA-fine-tuned-mbart-tl2en-baseline-best"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load LoRA config + model
    config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_path)

    # Set evaluation mode
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ======================
# Translation Function
# ======================
def translate_text(text: str, tokenizer, model, src_lang="tl_XX", tgt_lang="en_XX"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# ======================
# Streamlit App UI
# ======================
st.set_page_config(page_title="Tagalog → English Translator", page_icon="🌐")

st.title("🌐 Tagalog → English Translator (Fine-tuned mBART50 + LoRA)")

user_input = st.text_area("Enter Tagalog text:", placeholder="Halimbawa: Kumusta ka na? #hello @world 😊", height=100)

if st.button("Translate"):
    if user_input.strip():
        # 1. Preprocess the user input
        cleaned_input = preprocess_text(user_input)

        # 2. Check if the text is empty *after* cleaning
        if not cleaned_input:
            st.warning("The input text is empty after cleaning artifacts. Please enter valid text.")
        else:
            with st.spinner("Translating..."):
                # 3. Pass the cleaned text to the translation function
                translation = translate_text(cleaned_input, tokenizer, model)
            st.success("✅ Translation complete!")
            st.text_area("English Translation:", value=translation, height=100)
    else:
        st.warning("Please enter some text to translate.")