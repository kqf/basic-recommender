import pandas as pd


def split(seq):
    examples = []
    for i, token in enumerate(seq):
        examples.append([seq[:i] + seq[i + 1:], token])
    return examples


class SemiSupervisedRecommender:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X):
        splitted = X.apply(split)
        flat = splitted.explode()

        data = pd.DataFrame({
            "observed": flat.str[0],
            "gold": flat.str[1],
        })

        self.model.fit(data, data["gold"])
        self

    def predict(self, X):
        data = pd.DataFrame({
            "observed": X,
        })
        return self.model.predict(data)
