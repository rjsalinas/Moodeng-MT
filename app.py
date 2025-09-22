# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

# ======================
# Load Fine-tuned Model
# ======================
@st.cache_resource
def load_model():
    model_path = "./python-script/mbart-lora-finetuned"
    
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

st.set_page_config(page_title="Tagalog → English Translator", page_icon="🌐")

st.title("🌐 Tagalog → English Translator (Fine-tuned mBART50 + LoRA)")

user_input = st.text_area("Enter Tagalog text:", placeholder="Halimbawa: Kumusta ka na?", height=100)

if st.button("Translate"):
    if user_input.strip():
        with st.spinner("Translating..."):
            translation = translate_text(user_input, tokenizer, model)
        st.success("✅ Translation complete!")
        st.text_area("English Translation:", value=translation, height=100)
    else:
        st.warning("Please enter some text to translate.")