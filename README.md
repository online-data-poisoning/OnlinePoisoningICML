# Experiment Scripts and Codes for Online Data Poisoning

This repository contains the experiment scripts, codes and necessary data for ICML 2020 submission of paper "An investigation of Data Poisoning Defenses for Online Learning".

## Running Environment

The code uses pytorch, gensim, numpy, scipy and sklearn.
The running environment can be loaded by anaconda using the online-poisoning.yml config file.

## Data and Results
The preprocessed data for experiment is located in the "data" folder.
Experiment results will go under the "results" folder.
Please keep the folder name as it is, as the plotting functions use the same name when loading results.

## Descriptions of Individual Notebooks

### Semi-online attacks



The main folder contains five jupyter notebooks. The "plotting.ipynb" notebook contains the scripts for plotting the figures shown in the paper from result files. The other four notebooks contain scripts that run experiments for different data set. The name of the notebook suggests the data set, i.e. "IMDB.ipynb" for the IMDB sentiment analysis. In order to reproduce the experiment result, one can simply run through all cells in each notebook.

The attacker.py and utils.py contain attacker class and other helper functions used in the scripts.

The data folder contains preprocessed data for the experiments. 


