#!/bin/bash

DATA='../mbart/bin'
SRC_LANG='en_XX'
TGT_LANG='gl_XX'
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

RESULTS_PATH='../mbart/results'

MODEL_PATH='../mbart/checkpoints/checkpoint_best.pt'

TOKENIZER_TYPE='moses'

BPE_TYPE='sentencepiece'
SPM_MODEL='../mbart/mbart.cc25.v2/sentence.bpe.model'

CUDA_VISIBLE_DEVICES=0,1 fairseq-generate "$DATA" --source-lang $SRC_LANG --target-lang $TGT_LANG \
  --path $MODEL_PATH \
  --results-path $RESULTS_PATH \
  --bpe $BPE_TYPE --sentencepiece-model $SPM_MODEL \
  --task translation_from_pretrained_bart \
  --langs $langs \
  --nbest $N_HYPOTHESIS \
  --beam 5