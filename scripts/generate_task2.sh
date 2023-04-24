#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_3.pt \
        --mps \
        --words 200 \
        --temperature 0.95 \
        --log-interval 50 \
        --outf $samples/sample_lowest_ppl.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_6.pt \
        --mps \
        --words 200 \
        --temperature 0.95 \
        --log-interval 50 \
        --outf $samples/sample_highest_ppl.txt
)