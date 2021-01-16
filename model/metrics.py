import numpy as np


def recall(y_true, y_pred=None, k=None, padding_label="<pad>"):
    y_true, y_pred = np.atleast_2d(y_true, y_pred)
    tp = (y_pred[:, :, None] == y_true[:, None]).any(axis=-1)
    true = y_true != padding_label
    return tp.sum(-1) / true.sum(-1)
