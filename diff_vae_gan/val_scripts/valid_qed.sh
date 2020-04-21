#!/bin/bash

DIR=$1
NUM=$2

for ((i=0; i<NUM; i++)); do
    f=$DIR/model.iter-$i
    if [ -e $f ]; then
        echo $f
        python ~/iclr19-graph2graph/diff_vae_gan/decode.py --test ~/iclr19-graph2graph/data/qed/valid_toy.txt --vocab ~/iclr19-graph2graph/data/qed/vocab.txt --model $f --hidden_size 10 --use_molatt | python ~/iclr19-graph2graph/scripts/qed_score.py > $DIR/results.$i
        python ~/iclr19-graph2graph/scripts/qed_analyze.py < $DIR/results.$i
    fi
done
