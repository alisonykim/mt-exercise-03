#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
logs=$base/logs
ppls=$base/ppls

mkdir -p $models
mkdir -p $logs
mkdir -p $ppls

num_threads=4
device=""

echo $'----------------------------------------------------------'
echo $'Training with 0 dropout'
echo $'----------------------------------------------------------'

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py \
        --data $data/tarantino \
        --epochs 50 \
        --batch_size 20 \
        --mps \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0 --tied \
        --save_model $models/model_2.pt \
        --save_ppl $ppls/ppl_0_drop \
)

echo "time taken:"
echo "$SECONDS seconds"


echo $'----------------------------------------------------------'
echo $'Training with 0.2 dropout'
echo $'----------------------------------------------------------'

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py \
        --data $data/tarantino \
        --epochs 50 \
        --batch_size 20 \
        --mps \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.2 --tied \
        --save_model $models/model_3.pt \
        --save_ppl $ppls/ppl_20_drop \
)

echo "time taken:"
echo "$SECONDS seconds"


echo $'----------------------------------------------------------'
echo $'Training with 0.5 dropout'
echo $'----------------------------------------------------------'

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py \
        --data $data/tarantino \
        --epochs 50 \
        --batch_size 20 \
        --mps \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.5 --tied \
        --save_model $models/model_4.pt \
        --save_ppl $ppls/ppl_50_drop \
)

echo "time taken:"
echo "$SECONDS seconds"


echo $'----------------------------------------------------------'
echo $'Training with 0.7 dropout'
echo $'----------------------------------------------------------'

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py \
        --data $data/tarantino \
        --epochs 50 \
        --batch_size 20 \
        --mps \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.7 --tied \
        --save_model $models/model_5.pt \
        --save_ppl $ppls/ppl_70_drop \
)

echo "time taken:"
echo "$SECONDS seconds"


echo $'----------------------------------------------------------'
echo $'Training with 0.9 dropout'
echo $'----------------------------------------------------------'

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py \
        --data $data/tarantino \
        --epochs 50 \
        --batch_size 20 \
        --mps \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.9 --tied \
        --save_model $models/model_6.pt \
        --save_ppl $ppls/ppl_90_drop \
)

echo "time taken:"
echo "$SECONDS seconds"