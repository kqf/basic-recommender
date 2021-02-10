import pytest
import numpy as np

from model.metrics import recall


@pytest.fixture
def gold(n_users=10_000, n_items=5):
    return np.ones((n_users, n_items), dtype=np.int16)


@pytest.fixture
def preds(gold, positive_answers):
    predictions = np.zeros_like(gold)
    predictions[:, :positive_answers] = 1
    return predictions


@pytest.mark.parametrize("positive_answers", [
    0, 1, 2, 3, 4, 5
])
def test_recall(positive_answers, preds, gold):
    desired = positive_answers / gold.shape[-1]
    mean_recall = recall(gold, preds, k=5, pad_token=-1).mean()
    np.testing.assert_almost_equal(mean_recall, desired)
