import pandas as pd
import numpy as np
from model.data import to_inference, pad


def recall(y_true, y_pred=None, k=None, padding_label="<pad>"):
    y_true, y_pred = np.atleast_2d(y_true, y_pred)
    tp = (y_pred[:, :, None] == y_true[:, None]).any(axis=-1)
    true = y_true != padding_label
    return tp.sum(-1) / true.sum(-1)


def recall_score(model, data):
    # Approximate the recall on the data set
    observed, gold = to_inference(data)
    preds = model.predict(observed)

    metrics = pd.DataFrame({"preds": preds.tolist(), "gold": gold})
    metrics["output"] = recall(pad(gold), preds)
    print(metrics)
    return recall(pad(gold), preds).mean()
