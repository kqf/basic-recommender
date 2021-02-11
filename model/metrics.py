from model.data import to_inference, pad
from irmetrics.topk import recall


def recall_score(model, data):
    # Approximate the recall on the data set
    observed, gold = to_inference(data)
    preds = model.predict(observed)
    return recall(pad(gold), preds, pad_token="<unk>").mean()
