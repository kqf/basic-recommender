import numpy as np
from sklearn.model_selection import KFold

from model.data import read_dataset, to_inference, pad
from model.pop import PopRecommender
from model.coo import CooRecommender
from model.metrics import recall


def recall_score(model, data):
    # Approximate the recall on the data set
    observed, gold = to_inference(data)
    preds = model.predict(observed)
    # Pad gold, so it's possible to use matrix operations
    return recall(pad(gold), preds).mean()


def main(name="popularity"):
    data = read_dataset("data/train.csv")

    scores_tr, scores_te = [], []
    for idx_tr, idx_te in KFold(5).split(data):
        # Split the data
        train, test = data.iloc[idx_tr], data.iloc[idx_te]

        # Fit the model
        model = PopRecommender()
        model.fit(train)

        # Approximate the recall on the train set
        scores_tr.append(recall_score(model, train))

        # The same for test set
        scores_te.append(recall_score(model, test))

    message = "Recall@5 for {} model at {} set: {:.4g} +/- {:.4g}"
    print(message.format(name, "train", np.mean(scores_tr), np.std(scores_tr)))
    print(message.format(name, "valid", np.mean(scores_te), np.std(scores_te)))


if __name__ == '__main__':
    main()
