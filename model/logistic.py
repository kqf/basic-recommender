from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from model.semisupervised import SemiSupervisedRecommender


def tokenize(x):
    return x


class PandasSelector:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]


def build_model():
    pipeline = make_pipeline(
        PandasSelector("observed"),
        CountVectorizer(lowercase=False, tokenizer=tokenize),
        LogisticRegression(),
    )

    return SemiSupervisedRecommender(pipeline)
