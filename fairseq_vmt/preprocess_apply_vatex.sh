#!/bin/sh
TEXT=dataset/vatex
DATADIR=data-bin/vatex
PRETRAIN=dataset
DICT=data-bin/combine
# build subword vocab
SUBWORD_NMT=../subword-nmt
NUM_OPS=32000
# # Use codes and vocab of wmt17
CODES=codes.${NUM_OPS}.bpe
VOCAB=vocab.${NUM_OPS}.bpe

echo "Applying vocab to training"
$SUBWORD_NMT/apply_bpe.py -c $PRETRAIN/${CODES}.en --vocabulary $PRETRAIN/${VOCAB}.en < $TEXT/train.clean.en > $TEXT/train.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $PRETRAIN/${CODES}.zh --vocabulary $PRETRAIN/${VOCAB}.zh < $TEXT/train.clean.zh > $TEXT/train.${NUM_OPS}.bpe.zh

# encode validation
echo "Applying vocab to valid"
$SUBWORD_NMT/apply_bpe.py -c $PRETRAIN/${CODES}.en --vocabulary $PRETRAIN/${VOCAB}.en < $TEXT/valid.clean.en > $TEXT/valid.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $PRETRAIN/${CODES}.zh --vocabulary $PRETRAIN/${VOCAB}.zh < $TEXT/valid.clean.zh > $TEXT/valid.${NUM_OPS}.bpe.zh

# encode test
echo "Applying vocab to test"
$SUBWORD_NMT/apply_bpe.py -c $PRETRAIN/${CODES}.en --vocabulary $PRETRAIN/${VOCAB}.en < $TEXT/test.clean.en > $TEXT/test.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $PRETRAIN/${CODES}.zh --vocabulary $PRETRAIN/${VOCAB}.zh < $TEXT/test.clean.zh > $TEXT/test.${NUM_OPS}.bpe.zh

# Preprocess dataset with combined dictionary
echo "Preprocessing datasets..."
rm -rf $DATADIR
mkdir -p $DATADIR
fairseq-preprocess --source-lang en --target-lang zh \
    --trainpref $TEXT/train.${NUM_OPS}.bpe --validpref $TEXT/valid.${NUM_OPS}.bpe --testpref $TEXT/test.${NUM_OPS}.bpe \
    --srcdict $DICT/dict.en.txt --tgtdict $DICT/dict.zh.txt \
    --thresholdsrc 0 --thresholdtgt 0 --destdir $DATADIR --workers 20 

# TEXT=examples/translation/iwslt14.tokenized.de-en
# fairseq-preprocess --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/iwslt14.tokenized.de-en \
#     --workers 20

