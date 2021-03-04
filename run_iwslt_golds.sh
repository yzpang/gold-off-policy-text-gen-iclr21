TOTAL_NUM_UPDATES=150000
WARMUP_UPDATES=1000
UPDATE_FREQ=1

MLE_PATH=[TODO] 

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en --save-dir checkpoints_iwslt_golds \
    --restore-file $MLE_PATH \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 3e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --p40 1 --use-is-obj 1 --save-interval-updates 10000 --keep-interval-updates 3 --policy-update-per-k-epoch 3000 --q-baseline -10.0 --iw-min 0.20 --reset-optimizer --trunc-min 0.1 --reward-type sump
