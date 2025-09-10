#!/bin/bash

DATA='../mbart/bin' 
CHECKPOINTS_DIR='../mbart/checkpoints'

SRC_LANG='en_XX'
TGT_LANG='tl_XX'
mBART_MODEL='../mbart/mbart.cc25.v2/model.pt'
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
TASK='translation_from_pretrained_bart'

SCHEDULER='polynomial_decay' 
ARCHITECTURE='mbart_large'
LEARNING_RATE=3e-4

BPE_TYPE='sentencepiece'
SPM_MODEL='../mbart/mbart.cc25.v2/sentence.bpe.model'

TOKENIZER_TYPE='moses'

EPOCH=10
MAX_TOKENS=2048
OPTIMIZADOR='adam'
LOSS='label_smoothed_cross_entropy'

CUDA_VISIBLE_DEVICES=0,1 fairseq-train $DATA \
   --save-dir $CHECKPOINTS_DIR \
   --source-lang $SRC_LANG --target-lang $TGT_LANG \
   --bpe $BPE_TYPE --sentencepiece-model $SPM_MODEL \
   --max-epoch $EPOCH \
   --max-tokens $MAX_TOKENS \
   --optimizer $OPTIMIZADOR --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
   --lr-scheduler $SCHEDULER --lr $LEARNING_RATE --warmup-updates 2500 --total-num-update 40000 \
   --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
   --arch $ARCHITECTURE --layernorm-embedding \
   --encoder-normalize-before --decoder-normalize-before \
   --share-decoder-input-output-embed \
   --criterion $LOSS --label-smoothing 0.2 \
   --encoder-learned-pos \
   --update-freq 8 \
   --validate-interval-updates 5000 \
   --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
   --restore-file $mBART_MODEL \
   --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
   --seed 222 \
   --task $TASK \
   --langs $langs