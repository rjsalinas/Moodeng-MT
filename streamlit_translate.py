#!/usr/bin/env python3
"""
Streamlit app: Compare vanilla mBART50 vs fine-tuned baseline (LoRA) for TL→EN.

- Left: input Filipino/Taglish text
- Right: translations from both models
- Shows a simple confidence score (avg token probability) per output

Run:
  streamlit run streamlit_translate.py
"""

import os
import torch
import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel


BASE_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
FINETUNED_DIR = "fine-tuned-mbart-tl2en-baseline-best"  # folder in repo


@st.cache_resource(show_spinner=False)
def load_vanilla():
    model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    tok = MBart50Tokenizer.from_pretrained(BASE_MODEL_NAME)
    tok.src_lang = "tl_XX"
    tok.tgt_lang = "en_XX"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tok, device


@st.cache_resource(show_spinner=False)
def load_finetuned():
    base = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    try:
        model = PeftModel.from_pretrained(base, FINETUNED_DIR)
    except Exception:
        model = base
    tok = MBart50Tokenizer.from_pretrained(BASE_MODEL_NAME)
    tok.src_lang = "tl_XX"
    tok.tgt_lang = "en_XX"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tok, device


def translate_with_confidence(model, tok, device, text: str):
    enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    bos = tok.lang_code_to_id.get("en_XX", tok.eos_token_id)

    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=bos,
            max_length=128,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            repetition_penalty=1.2,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    seq = out.sequences[0]
    text_out = tok.decode(seq, skip_special_tokens=True).strip()

    # Compute average token probability as a simple confidence
    # Use transition scores util from HF to align scores with generated tokens
    try:
        transition_scores = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]  # (seq_len-1,)
        # Use sigmoid over log-prob? Scores are log softmax; convert to probs
        token_probs = transition_scores.exp()
        # Confidence as average prob over generated tokens (ignore first token)
        conf = float(token_probs.mean().item())
    except Exception:
        conf = 0.0

    return text_out, conf


def main():
    st.set_page_config(page_title="TL→EN Translation Comparison", page_icon="🌐", layout="wide")
    st.title("TL→EN Translation Comparison")
    st.caption("Vanilla mBART50 vs Fine-tuned baseline (LoRA)")

    text = st.text_area("Enter Filipino/Taglish text", height=140, placeholder="Hal.: Good evening! Maganda sales namin these days …")

    col1, col2 = st.columns(2)
    with st.sidebar:
        st.subheader("Decoding")
        beams = st.slider("num_beams", 1, 8, 4)
        ngram = st.slider("no_repeat_ngram_size", 0, 6, 3)
        len_pen = st.slider("length_penalty", 0.5, 1.5, 0.8, 0.1)
        rep_pen = st.slider("repetition_penalty", 1.0, 2.0, 1.2, 0.1)
        max_len = st.slider("max_length", 64, 256, 128, 8)
        fast_mode = st.checkbox("Fast mode (beams=2, lighter confidence)", value=False)
        st.caption("Note: sliders affect both models equally for fair comparison.")

    if "_decode_params" not in st.session_state:
        st.session_state._decode_params = {}
    st.session_state._decode_params.update(
        dict(beams=(2 if fast_mode else beams), ngram=ngram, len_pen=len_pen, rep_pen=rep_pen, max_len=max_len)
    )

    if st.button("Translate", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
            return

        # Load models
        v_model, v_tok, device = load_vanilla()
        f_model, f_tok, _ = load_finetuned()

        # Override decoding per sidebar (patch via closure)
        def _english_fraction(s: str) -> float:
            if not s:
                return 0.0
            # Count basic English letters, digits, space and common punctuation as "English"
            eng_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.?!'\"-:;()[]{}\n\t")
            eng = sum((ch in eng_chars) for ch in s)
            return min(1.0, max(0.0, eng / max(1, len(s))))

        def run(model, tok):
            enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            bos = tok.lang_code_to_id.get("en_XX", tok.eos_token_id)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    forced_bos_token_id=bos,
                    max_length=st.session_state._decode_params["max_len"],
                    num_beams=st.session_state._decode_params["beams"],
                    do_sample=False,
                    no_repeat_ngram_size=st.session_state._decode_params["ngram"],
                    length_penalty=st.session_state._decode_params["len_pen"],
                    repetition_penalty=st.session_state._decode_params["rep_pen"],
                    early_stopping=True,
                )
            seq = out_ids[0]
            text_out = tok.decode(seq, skip_special_tokens=True).strip()

            # Confidence: entropy-based certainty under teacher forcing (lower entropy → higher confidence)
            try:
                with torch.no_grad():
                    labels = seq.unsqueeze(0)
                    labels[labels == tok.pad_token_id] = -100
                    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
                    logits = outputs.logits  # (1, T, V)
                    # Align with labels (ignore -100)
                    mask = (labels != -100)
                    T = int(mask.sum().item())
                    if T == 0:
                        raise RuntimeError("no valid tokens for confidence")
                    sel_logits = logits[mask]  # (T, V)
                    probs = torch.softmax(sel_logits, dim=-1)
                    # token-level entropy H = -sum p log p
                    ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # (T,)
                    # Normalize by log(V) to get 0..1
                    V = sel_logits.size(-1)
                    ent_norm = ent / float(torch.log(torch.tensor(V, dtype=ent.dtype, device=ent.device)))
                    entropy_mean = float(ent_norm.mean().item())  # 0..1
                    prob_core = max(0.0, min(1.0, 1.0 - entropy_mean))
                # English content ratio to penalize wrong-script outputs
                lang_factor = _english_fraction(text_out)
                # Very short outputs are unreliable → downweight
                length_factor = min(1.0, max(0.2, len(text_out) / 40.0))
                conf = max(0.0, min(1.0, prob_core * lang_factor * length_factor))
            except Exception:
                conf = 0.0
            return text_out, conf

        t_vanilla, c_vanilla = run(v_model, v_tok)
        t_finetuned, c_finetuned = run(f_model, f_tok)

        with col1:
            st.subheader("mBART50 (vanilla)")
            st.write(t_vanilla)
            st.progress(min(max(c_vanilla, 0.0), 1.0))
            st.caption(f"Confidence: {c_vanilla*100:.0f}%")

        with col2:
            st.subheader("Fine-tuned baseline (LoRA)")
            st.write(t_finetuned)
            st.progress(min(max(c_finetuned, 0.0), 1.0))
            st.caption(f"Confidence: {c_finetuned*100:.0f}%")


if __name__ == "__main__":
    main()


