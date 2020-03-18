#!/bin/sh

WMT=dataset/wmt17
VATEX=dataset/vatex
TEXT=dataset
DATADIR=data-bin/combine
# clean and tokenize dataset. 
python ./preprocess/preprocess_wmt.py
python ./preprocess/preprocess_vatex.py

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
MOSESDECODER=../mosesdecoder
$MOSESDECODER/scripts/training/clean-corpus-n.perl $WMT/train en zh $WMT/train.clean 3 70
$MOSESDECODER/scripts/training/clean-corpus-n.perl $WMT/valid en zh $WMT/valid.clean 3 70
$MOSESDECODER/scripts/training/clean-corpus-n.perl $WMT/test en zh $WMT/test.clean 3 70

$MOSESDECODER/scripts/training/clean-corpus-n.perl $VATEX/train en zh $VATEX/train.clean 3 70
$MOSESDECODER/scripts/training/clean-corpus-n.perl $VATEX/valid en zh $VATEX/valid.clean 3 70
$MOSESDECODER/scripts/training/clean-corpus-n.perl $VATEX/test en zh $VATEX/test.clean 3 70

# Concatenate training data from wmt17 and vatex
cat $WMT/train.clean.en $VATEX/train.clean.en > $TEXT/train.clean.en
cat $WMT/train.clean.zh $VATEX/train.clean.zh > $TEXT/train.clean.zh

# build subword vocab
SUBWORD_NMT=../subword-nmt
NUM_OPS=32000

# learn codes and encode separately
CODES=codes.${NUM_OPS}.bpe
echo "Encoding subword with BPE using ops=${NUM_OPS}"
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.en > $TEXT/${CODES}.en
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.zh > $TEXT/${CODES}.zh

echo "Applying vocab to training"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en < $TEXT/train.clean.en > $TEXT/train.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh < $TEXT/train.clean.zh > $TEXT/train.${NUM_OPS}.bpe.zh

VOCAB=vocab.${NUM_OPS}.bpe
echo "Generating vocab: ${VOCAB}.en"
cat $TEXT/train.${NUM_OPS}.bpe.en | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.en

echo "Generating vocab: ${VOCAB}.zh"
cat $TEXT/train.${NUM_OPS}.bpe.zh | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.zh

echo "Preprocessing combined datasets..."
rm -rf $DATADIR
mkdir -p $DATADIR
fairseq-preprocess --source-lang en --target-lang zh \
    --trainpref $TEXT/train.${NUM_OPS}.bpe \
    --thresholdsrc 0 --thresholdtgt 0 --destdir $DATADIR --workers 20