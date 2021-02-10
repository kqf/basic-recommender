from model.data import to_inference, pad
from irmetrics import recall as irecall


def recall(y_true, y_pred=None, k=None, padding_label="<pad>"):
    return irecall(y_true, y_pred, k=k, padding_token=padding_label)


def recall_score(model, data):
    # Approximate the recall on the data set
    observed, gold = to_inference(data)
    preds = model.predict(observed)
    return recall(pad(gold), preds).mean()
