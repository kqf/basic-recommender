import pytest

from model.pop import build_model as build_pop
from model.coo import build_model as build_coo
from model.logistic import build_model as build_logistic
from model.semantic import build_model as build_semantic

from model.data import to_inference
from model.metrics import recall_score


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
    model.predict(test)
    print(recall_score(model, data))
