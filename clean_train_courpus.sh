#!/usr/local/bin bash
if [ ! -d "mosesdecoder" ]; then
    git clone https://github.com/moses-smt/mosesdecoder.git
fi

data_root='./data/'
script="mosesdecoder/scripts/training/clean-corpus-n.perl"
dataset='wmt17_en-cs'
l1='en'
l2='cs'
train_splits="train"
# other_splits="validation test"
ratio=5
min=5
max=1024

for split in $train_splits; do
    perl $script -ratio $ratio $data_root/$dataset/$split/raw $l1 $l2 $data_root/$dataset/$split/clean $min $max
done