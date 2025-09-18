#!/bin/bash

DATA="../mbart/bpe"
DEST_DIR="../mbart/bin"

SRC_LANG="tl_XX"
TGT_LANG="en_XX"

BPE_TYPE="sentencepiece"
DICTIONARY="../mbart/mbart.cc25.v2/dict.txt"

CUDA_VISIBLE_DEVICES=0,1 fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref $DATA/train --validpref $DATA/valid \
    --destdir $DEST_DIR \
    --bpe $BPE_TYPE \
    --srcdict $DICTIONARY --tgtdict $DICTIONARY \
    --workers 70 \