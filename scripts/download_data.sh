#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

echo 'Downloading raw screenplays'
mkdir -p $data/tarantino
mkdir -p $data/tarantino/raw
curl -o $data/tarantino/raw/django.txt https://imsdb.com/scripts/Django-Unchained.html
curl -o $data/tarantino/raw/inglorious_basterds.txt https://imsdb.com/scripts/Inglourious-Basterds.html
curl -o $data/tarantino/raw/jackie_brown.txt https://www.dailyscript.com/scripts/jackiebrown.html
curl -o $data/tarantino/raw/kill_bill.txt https://imsdb.com/scripts/Kill-Bill-Volume-1-%2526-2.html
curl -o $data/tarantino/raw/pulp_fiction.txt https://www.dailyscript.com/scripts/pulp_fiction.html
curl -o $data/tarantino/raw/reservoir_dogs.txt https://www.dailyscript.com/scripts/Reservoir+Dogs.txt

# remove beginning credits from both screenplays and combine into single screenplay file

echo 'Concatenate screenplays into single file'
{ cat $data/tarantino/raw/pulp_fiction.txt ; \
tail -n +330 $data/tarantino/raw/django.txt ; \
tail -n +222 $data/tarantino/raw/inglorious_basterds.txt ; \
tail -n +49 $data/tarantino/raw/jackie_brown.txt ; \
tail -n +219 $data/tarantino/raw/kill_bill.txt ; \
tail -n +21 $data/tarantino/raw/reservoir_dogs.txt ; } > \
$data/tarantino/raw/tarantino.txt

# preprocess slightly

echo 'Preprocess'
cat $data/tarantino/raw/tarantino.txt | python $base/scripts/preprocess_raw.py > $data/tarantino/raw/tarantino.cleaned.txt

# tokenize, fix vocabulary upper bound

echo 'Tokenize, fix vocabulary upper bound'
cat $data/tarantino/raw/tarantino.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang 'en' --sent-tokenize --language 'english' > \
    $data/tarantino/raw/tarantino.preprocessed.txt


# split into train, valid and test

echo 'Shuffle lines'
shuf data/tarantino/raw/tarantino.preprocessed.txt >> data/tarantino/raw/tarantino.shuffled.txt

echo 'Split train/valid/test'
head -n 8000 $data/tarantino/raw/tarantino.shuffled.txt > $data/tarantino/train.txt
tail -n +8001 $data/tarantino/raw/tarantino.shuffled.txt | head -n 1000 > $data/tarantino/valid.txt
tail -n +9001 $data/tarantino/raw/tarantino.shuffled.txt > $data/tarantino/test.txt