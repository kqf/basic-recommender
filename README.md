# Basic Recommender ![tests](https://github.com/kqf/basic-recommender/workflows/tests/badge.svg)


This is a toy example of a recommender that produces some suggestions given a set of entities. All the models here are collaborative. They make predictions based on observed co-occurrences.

## Install
```bash
# git clone this repo

pip install -r requirements.txt
pip install .
```

## Run

To run the experiments do
```bash

# Download the dataset *.csv into ./data folder 
# the default model is "semantic", it also supports "pop" and "coo" models
make develop model=pop
```
To train the model on the full dataset:
```bash

make model=pop
```

Prepare the final submission
```bash
make target=submission.csv
```
