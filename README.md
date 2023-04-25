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
Our dataset comprises a selection of screenplays by Quentin Tarantino. They were downloaded from <a href="https://imsdb.com">The Internet Movie Script Database (IMSDb)</a> and <a href="https://www.dailyscript.com" target="_blank">Daily Script</a>. We ensured that the preprocessed dataset meets the length requirement set by the assignment, namely that it contains between 5,000-10,000 segments (sentences in our case).<br>

**Content warning:** Unsurprisingly, Tarantino films contain profanity, including pejoratives based on race, gender, religion/creed, national origin, and other facets of identity. We ask the user to be aware of this when examining the datasets and generated outputs. The views, language, and behaviors of the characters in the screenplays do not necessarily reflect our own.<br>

To download and preprocess the dataset, we made simple modifications to existing scripts:
* ```scripts/download_data.sh```: add URLs to screenplays for download, concatenate screenplays into a single file, shuffle lines before performing train/valid/test split (all other sections remain the same other than filenames)
* ```scripts/preprocess_raw.py```: remove unwanted characters that are specific to the screenplays, write lines to file only if they contain 6 or more tokens

Of course, the above modifications are specific to the screenplays that we downloaded. Many parameters (e.g. how many lines to skip when concatenating each screenplay) were determined by manually examining each text file. We acknowledge that the preprocessing was not perfect and that this will certainly affect training.

### Step 2: Model training
We trained the model on a MacOS GPU machine (M1 Pro). The availability of the GPU was verified in a Python environment using ```torch.backends.mps.is_available()``` and specified in the training bash script as a flag (```--mps```).<br>

An additional flag ```--log-print-statements``` allows the user to specify whether output is printed to the console or to a log file, accessible in ```logs/```. Whether to the console or log file, the speciied training parameters are printed to help with potential debugging. Other changes to ```tools/pytorch-examples/word_language_model/main.py``` include:
* changing the ```ArgumentParser``` description to represent our dataset of choice
* renaming the ```---save``` flag as ```---save_model``` to distinguish it from the flag ```save_ppl``` that we added in Part 2<br>

As instructed, training was carried out with the original parameters specified in the training script. (Our interpretation of the instructions was to change settings only if training took longer than 2 hours.) The loss and perplexity updates printed to console can be found in ```logs/log_1.log```.

### Step 3: Text generation
To compare different generation parameters, we extended ```scripts/generate.sh``` with additional generation runs and different values of ```--temperature```. The generated results were saved to separate files by specifying different output filenames with the ```--outf``` flag.<br>

### Step 4: Discussion/Reflection
The dataset comprises several screenplays directed and/or written by Quentin Tarantino. Speech is not necessarily grammatical and sometimes not fluent, so we don't expect the generated text to reflect these qualities, either. For time reasons, we decided to include stage directions (e.g. "The sequence ends with the Bride arriving at Bill's home."), which we expect to impact the model.<br>

Additionally, screenplays have specific structures, and there was variation between those in our dataset due to different formatting (e.g. HTML to TXT conversion). It would have been time-consuming to adapt ```scripts/preprocess{_raw}.py``` to every screenplay, so we decided to catch as many of these differences as possible within a general preprocessing paradigm. Still, the paradigm is not comprehensive, which we also expect to impact the model.<br>

As expected, our generated text was quite nonsensical. After running generation with the standard parameters, we wondered what the text would look like if we decreased the temperature. Our intuition was that the text at least would be more grammatical, even if at the expense of output diversity. This was more or less the case, albeit largely due to the increased presence of \<eos\> and \<unk\> tokens.<br>

Making the preprocessing paradigm more precise might improve the quality of generated text.

## 2. Parameter tuning: Experimenting with dropout
### Step 1: Training 5 models with varying dropout settings
We trained 5 models with the following dropout settings: 0, 0.2, 0.5, 0.7 and 0.9. As not to overwrite the script from task 1, we saved the training bash script for task 2 separately as ```scripts/train_task2.sh```. We did not change the embedding size and number of hidden units (both 200). We also added the ```--save_ppl``` flag to ```tools/pytorch-examples/word_language_model/main.py``` in order to save the training, validation and test perplexities for each dropout setting in csv files which are in the ```ppls/``` directory. We additionally imported pandas in the ```main.py``` script to save the perplexities in a tabular format. The resulting models are in the ```models/``` directory (models 2-6). We commited a copy of ```main.py``` that includes the changes to ```scripts/main_lm.py```, but to be able to run the programms it should be moved and renamed to ```tools/pytorch-examples/word_language_model/main.py```.

### Step 2: Creating tables and line charts
With the script ```scripts/plot_perplexities.py``` we first created a table for each of the three different perplexities which contains the values of all models / dropout settings (taking the files from ```ppls/``` as input). The tables were stored in the ```ppl_tables/``` directory. We then also created line charts for the training and validation perplexities which were saved in the ```ppl_plot/``` directory.

### Step 3: Generating sample text with models that obtained highest and lowest test perplexity
To generate sample texts using the models that obtained the highest and lowest test perplexity, we adapted ```scripts/generate.sh``` and saved the changes in ```scripts/generate_task2.sh```. The script uses model 3 (lowest test ppl) and model 6 (highest test ppl) to generate sample texts.

### Step 4: Findings / Discussion
#### Connection between training, validation and test perplexity
There doesn't seem to be a clear connection between training, validation and test perplexity but there are some similarities. The lowest test perplexity and thus presumably the best performance was achieved with 0.2 dropout, followed by 0.5 and 0 dropout. Similarly, the lowest training perplexity was achieved with 0 dropout, followed by 0.2 and 0.5 dropout. By contrast, 0.5 dropout led to the lowest validation perplexity, followed by 0.2 and 0.7 dropout. As expected, the 0.9 dropout setting led to the highest training, validation and test perplexity by far. For the training and validation data, the perplexities decrease most significantly within the first 10 epochs, then stabilizing over the following 40 epochs. While for the training and test data, a model with a lower dropout setting performs better, this isn't as much the case with the validation data. However, overall it still seems likely that a dropout setting between 0.1 and 0.5 should work best for our data.

#### Comparison of generated sample texts
As in part 1, the generated sample texts are still mostly nonsensical (see ```samples/``` directory). However, the sample text generated using the model with the lowest test perplexity is slightly more coherent and at least somewhat grammatical compared to the sample text obtained with the worst performing model (i.e. the highest test perplexity).