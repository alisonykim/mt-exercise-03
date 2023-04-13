# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/moritz-steiner/mt-exercise-03
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

# Submission

## 1. Training a recurrent neural network language model
### Step 1: Choosing a dataset
Our dataset comprises a selection of screenplays by Quentin Tarantino. It is freely accessible <a href="" target="_blank">via the Gutenberg Project</a>. We ensured that the preprocessed dataset meets the length requirement set by the assignment, namely that it contains between 5,000-10,000 segments (i.e. sentences in our case).<br>

To download and preprocess the dataset, we made simple modifications to existing scripts:
* ```scripts/download_data.sh```: add URLs to screenplays for download, concatenate screenplays into a single file, shuffle lines before performing train/valid/test split (all other sections remain the same other than filenames)
* ```scripts/preprocess_raw.py```: remove unwanted characters that are specific to the screenplays, write lines to file only if they contain 6 or more tokens
Of course, the above modifications are specific to the screenplays that we downloaded. Many parameters (e.g. how many lines to skip when concatenating each screenplay) were determined manually by examining the .txt file. We note that the preprocessing was not perfect and that this will certainly affect training.

### Step 2: Model training
We trained the model on a MacOS GPU machine (M1 Pro). The availability of the GPU was verified in a Python environment using ```torch.backends.mps.is_available()``` and specified in the training bash script as a flag (```--mps```). However, to help with potential debugging, a small print statement was added to the aforementioned script to inform the user of the device being used. Additionally, the ```ArgumentParser``` description were changed to represent our dataset of choice. Finally, the flag ```save``` was renamed ```save_model``` to distinguish it from the flag ```save_ppl``` that we added in Part 2.<br>

As instructed, training was carried out with the original parameters specified in the training script. (Our interpretation of the instructions was to change settings only if training took longer than 2 hours.) The statements printed to console can be found in ```logs/training.log```.

### Step 3: Text generation
To compare different generation parameters, we extended ```generate.sh``` with additional generation runs and different values of ```--temperature```. The generated results were saved to separate files by specifying different output filenames with the ```--outf``` flag.<br>

As in ```main.py```, minor modifications were made to the ```ArgumentParser``` description and default value of ```--data``` in ```generate.py```.

### Step 4: Discussion
See ```akim_nbleiker_mt_ex03.pdf```

## 2. Parameter tuning: Experimenting with dropout
For training with modifications to the dropout value, we created a copy of the training bash script titled ```train_dropout.sh```. The dropout values were modified in increments of 0.25, i.e. {0, 0.25, 0.50, 0.75, 1}, and each model was saved to a separate file {models/model_2-{1, 2, 3, 4, 5}}, respectively. The embedding size and number of hidden units were both set to 225 with the ```--emsize``` and ```--nhid``` flags, respectively.<br>

To save the three perplexity values, the following modifications were made to ```main.py```:
* Create flag ```--save_ppl``` as an optional CL argument
* Modify ```train()``` so that it outputs the perplexity after each training epoch
* Use the ```csv``` library to create one file per data subset (```ppls/ppl_train.csv```, ```ppls/ppl_valid.csv```, ```ppls/ppl_test.csv```)
* Create the script ```word_language_model/plot_ppls.py``` to plot perplexities
    * Use the ```pandas``` library to create a DataFrames out of the CSV logs
    * Use the ```matplotlib``` library to create line plots from the DataFrame values