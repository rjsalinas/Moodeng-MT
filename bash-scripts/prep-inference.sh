#!/bin/bash
set -e

# ====== CONFIG ======
CSV_FILE="../test-corpus.csv"
SRC_COL="preprocessed_text"
TGT_COL="english_translation"

SRC_LANG_CODE="tl_XX"
TGT_LANG_CODE="en_XX"

SPM_MODEL="../mbart/mbart.cc25.v2/sentence.bpe.model"
DICTIONARY="../mbart/mbart.cc25.v2/dict.txt"

OUT_DIR="data_bin_test"
TMP_DIR="tmp_test"
RESULTSPATH="results"

echo "[1/4] Extracting parallel text from CSV..."
mkdir -p "$TMP_DIR"
python3 - <<EOF
import pandas as pd
df = pd.read_csv("$CSV_FILE")
df["$SRC_COL"].to_csv(f"$TMP_DIR/test.src", index=False, header=False)
df["$TGT_COL"].to_csv(f"$TMP_DIR/test.tgt", index=False, header=False)
EOF

echo "[2/4] Encoding with SentencePiece..."
spm_encode --model="$SPM_MODEL" < "$TMP_DIR/test.src" > "$TMP_DIR/test.$SRC_LANG_CODE"
spm_encode --model="$SPM_MODEL" < "$TMP_DIR/test.tgt" > "$TMP_DIR/test.$TGT_LANG_CODE"

echo "[3/4] Binarizing test set..."
fairseq-preprocess \
  --source-lang $SRC_LANG_CODE --target-lang $TGT_LANG_CODE \
  --testpref "$TMP_DIR/test" \
  --destdir "$OUT_DIR" \
  --srcdict "$DICTIONARY" \
  --tgtdict "$DICTIONARY" \
  --workers 4

echo "[4/4] Running inference..."
mkdir -p "$RESULTSPATH"
CUDA_VISIBLE_DEVICES=0 fairseq-generate "$OUT_DIR" \
  --source-lang $SRC_LANG_CODE --target-lang $TGT_LANG_CODE \
  --path "$MODEL_PATH" \
  --results-path "$RESULTS_PATH" \
  --bpe sentencepiece --sentencepiece-model "$SPM_MODEL" \
  --task translation_from_pretrained_bart \
  --langs $LANGS \
  --nbest $NBEST \
  --beam $BEAM \
  --remove-bpe=sentencepiece \
  --scoring sacrebleu

echo "✅ Done! Check $RESULTS_PATH for translations and BLEU score."