import pytest

from model.pop import PopRecommender
from model.coo import CooRecommender
# from model.logistic import build_model
from model.semantic import build_model
from model.data import to_inference, pad
from model.metrics import recall


@pytest.mark.parametrize("build", [
    # PopRecommender,
    # CooRecommender,
    build_model,
])
def test_models(build, data):
    test, gold = to_inference(data)
    model = build()
    model.fit(test)
    preds = model.predict(test)
    recall(pad(gold), preds).mean()
