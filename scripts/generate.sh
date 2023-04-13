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
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --log-interval 50 \
        --outf $samples/sample_1-1.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --temperature 0.5 \
        --log-interval 50 \
        --outf $samples/sample_1-2.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --temperature 0.75 \
        --log-interval 50 \
        --outf $samples/sample_1-3.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --temperature 0.9 \
        --log-interval 50 \
        --outf $samples/sample_1-4.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --temperature 0.95 \
        --log-interval 50 \
        --outf $samples/sample_1-5.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --temperature 1.25 \
        --log-interval 50 \
        --outf $samples/sample_1-6.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --temperature 2 \
        --log-interval 50 \
        --outf $samples/sample_1-7.txt
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/tarantino \
        --checkpoint $models/model_1.pt \
        --mps \
        --words 200 \
        --temperature 3 \
        --log-interval 50 \
        --outf $samples/sample_1-8.txt
)