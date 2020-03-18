CHECKPOINT=checkpoints/wmt17_enzh
mkdir -p $CHECKPOINT
CUDA_VISIBLE_DEVICES=2 fairseq-train \
    data-bin/wmt \
    --source-lang en --target-lang zh \
    --arch transformer --share-decoder-input-output-embed \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --num-workers 32 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --min-lr '1e-09' --warmup-init-lr '1e-07' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --fixed-validation-seed 7 \
    --save-dir $CHECKPOINT \
    --log-format 'simple' --log-interval 100
    --max-update 200000
    # --eval-bleu \
    # --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    # --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe \
    # --eval-bleu-print-samples \
    # --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    # --fp16

