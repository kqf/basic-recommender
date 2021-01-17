import pytest

from model.pop import build_model as build_pop
from model.coo import build_model as build_coo
from model.logistic import build_model as build_logistic
from model.semantic import build_model as build_semantic

from model.data import to_inference, pad
from model.metrics import recall


@pytest.mark.parametrize("build", [
    build_pop,
    build_coo,
    build_logistic,
    build_semantic,
])
def test_models(build, data):
    test, gold = to_inference(data)
    model = build()
    model.fit(test)
    preds = model.predict(test)
    recall(pad(gold), preds).mean()
