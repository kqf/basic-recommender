from model.data import to_inference, pad
from irmetrics.topk import recall as irecall


def recall(y_true, y_pred=None, k=None, pad_token="<pad>"):
    return irecall(y_true, y_pred, k=k, pad_token=pad_token)


def recall_score(model, data):
    # Approximate the recall on the data set
    observed, gold = to_inference(data)
    preds = model.predict(observed)
    return recall(pad(gold), preds).mean()
