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

### Semi-online

The notebook "imdb.ipynb", "fmnist.ipynb", "mnist.ipynb", "breast-cancer.ipynb" contain scripts for semi-online experiment.

### Fully-online

The notebook "imdb-fully-online.ipynb", "fmnist-fully-online.ipynb", "mnist-fully-online.ipynb", "breastcancer-fully-online.ipynb" contain scripts for fully-online experiment.

### Plotting

The notebook "plotting.ipynb" generates the plot for semi-online experiment
The notebook "plotting-full.ipynb" generates the plot for fully-online experiment.

The semi-online and fully-online experiment scripts will output results to "results" folder.
The plotting scripts then loads the results from the folder for plotting.
