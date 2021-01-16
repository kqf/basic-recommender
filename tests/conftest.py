import pytest
import tempfile

import numpy as np
import pandas as pd

from model.data import read_dataset


def _fake_data(n_users, n_items):
    codes = []

    ids = np.arange(n_items)
    for _ in range(n_users):
        np.random.shuffle(ids)
        codes.append(",".join(ids[:10].astype(str)))

    data = {
        "id": range(n_users),
        "icd": codes
    }

    return pd.DataFrame(data)


@pytest.fixture
def fake_data_path(n_users=10_000, n_items=5000):
    with tempfile.NamedTemporaryFile() as filename:
        df = _fake_data(n_users, n_items)
        df.to_csv(filename.name, index=False)
        yield filename.name


@pytest.fixture
def data(fake_data_path):
    return read_dataset(fake_data_path)
