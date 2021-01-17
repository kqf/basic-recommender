import click
import numpy as np

from click import Path as cpth
from sklearn.model_selection import KFold

from model.data import read_dataset
# from model.pop import PopRecommender
# from model.coo import CooRecommender
# from model.logistic import build_model
from model.semantic import build_model
from model.metrics import recall_score


@click.command()
@click.option("--name", type=str, default="")
@click.option("--dataset", type=cpth(exists=True), default="data/train.csv")
def develop(name, dataset):
    data = read_dataset(dataset)

    scores_tr, scores_te = [], []
    for idx_tr, idx_te in KFold(5).split(data):
        # Split the data
        train, test = data.iloc[idx_tr], data.iloc[idx_te]

        # Fit the model
        model = build_model()
        model.fit(train)

        # Approximate the recall on the train set
        scores_tr.append(recall_score(model, train))

        # The same for test set
        scores_te.append(recall_score(model, test))

    message = "Recall@5 for {} model at {} set: {:.4g} +/- {:.4g}"
    print(message.format(name, "train", np.mean(scores_tr), np.std(scores_tr)))
    print(message.format(name, "valid", np.mean(scores_te), np.std(scores_te)))


@click.command()
@click.option("--name", type=str, default="")
@click.option("--submission", type=cpth(exists=False))
@click.option("--train", type=cpth(exists=True), default="data/train.csv")
@click.option("--test", type=cpth(exists=True), default="data/test.csv")
def train(name, train, test):
    train = read_dataset(train)
    model = build_model()
    model.fit(train)

    test = read_dataset(test)
    model.predict(test)
