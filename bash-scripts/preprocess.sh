#!/bin/bash

SPM_MODEL="../mbart/mbart.cc25.v2/sentence.bpe.model"
DATA="../corpus-parallel-txt"
DATA_BPE="../mbart/bpe"
SRC="tl"
TGT="en"

# Apply SentencePiece BPE
spm_encode --model=$SPM_MODEL < "$DATA/corpus.$SRC" > "$DATA_BPE/corpus.$SRC"
spm_encode --model=$SPM_MODEL < "$DATA/corpus.$TGT" > "$DATA_BPE/corpus.$TGT"