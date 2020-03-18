TEXT=vatex_zhen
OUTPUT=tmp/vatex/test20200318
CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/vatex \
    --path checkpoints/$TEXT/checkpoint_best.pt \
    --beam 5 --remove-bpe \
    --batch-size 200 \
    --source-lang zh --target-lang en | tee $OUTPUT.tmp

# TODO: decode subword BPE
cat $OUTPUT.tmp | sed -r 's/(@@ )|(@@ ?$)//g' > $OUTPUT.out

grep ^H $OUTPUT.out | cut -f3- > $OUTPUT.out.sys
grep ^T $OUTPUT.out | cut -f2- > $OUTPUT.out.ref
fairseq-score --sys $OUTPUT.out.sys --ref $OUTPUT.out.ref