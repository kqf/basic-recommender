import random
import pandas as pd
import numpy as np


def read_dataset(path, target_col="icd"):
    df = pd.read_csv(path)

    # Fill everything that has no codes
    cleaned = df.dropna()
    codes = cleaned[target_col].str.split(",")
    return codes


def _split(x):
    shuffled = random.sample(x, len(x))
    return shuffled[:-1], shuffled[-1:]


def to_inference(dataset):
    df = dataset[dataset.str.len() > 1]
    splitted = df.apply(_split)
    return splitted.str[0], splitted.str[1]


def pad(seq, k=5, padding_label="<pad>"):
    return np.stack([(s + [padding_label] * k)[:k] for s in seq])
