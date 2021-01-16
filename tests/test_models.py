import pytest

from model.pop import PopRecommender
from model.data import to_inference, pad
from model.metrics import recall


@pytest.mark.parametrize("build_model", [
    PopRecommender,
])
def test_models(build_model, data):
    test, gold = to_inference(data)
    model = build_model()
    model.fit(test)
    preds = model.predict(test)
    recall(pad(gold), preds).mean()
