import numpy as np
from collections import Counter


class PopRecommender:
    def __init__(self, k=5):
        self.k = k
        self.candidates_ = None
        self.counts_ = None

    def fit(self, X):
        common = Counter(X.explode()).most_common(self.k)
        self.candidates_, self.counts_ = zip(*common)
        return self

    def predict(self, X):
        # Convert to array of [1, self.k]
        candiates = np.array(self.candidates_)[None, :]

        # Repeat the same answer alog 0th axis -> [len(X), self.k]
        return np.repeat(candiates, X.shape[0], axis=0)
