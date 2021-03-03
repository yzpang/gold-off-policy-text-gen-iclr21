fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints_iwslt/checkpoint.backup.pt.0 \
    --batch-size 128 --beam 5 --remove-bpe --fix-fairseq-bug-summarization ## > ./iwslt/mle_tmp.inf.debug
