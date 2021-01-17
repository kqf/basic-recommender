import torch
import skorch
import random
import numpy as np

from sklearn.pipeline import make_pipeline
from functools import partial

from model.preprocessing import SequenceIterator, build_preprocessor
from model.preprocessing import train_split
from model.semisupervised import SemiSupervisedRecommender
from model.metrics import recall


SEED = 137
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class DynamicParametersSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["observed"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(module__padding_idx=vocab["<pad>"])
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')
        print(f'There number of unique items is {len(vocab)}')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SemanticNet(skorch.NeuralNet):
    def predict(self, X):
        # Now predict_proba returns top k indexes
        indexes = self.predict_proba(X)
        return np.take(X.fields["observed"].vocab.itos, indexes)

    def fit(self, X, y):
        # Ignore y, as preprocessing step has already taken it into account
        return super().fit(X, y=None)


class SemanticModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=100, padding_idx=0):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx)
        self._out = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len] -> embs[batch_size, seq_len, hidden_size]
        embs = self._emb(x)

        # context[batch_size, hidden_size]
        context = embs.mean(dim=1)

        # context[batch_size, hidden_size]  -> logits [batch_size, vocab_size]
        return self._out(context)


def scoring(model, X, y, k):
    probas = model.predict_proba(X)
    return recall(y[:, None], probas, k=k, padding_label=-1).mean()


def build_model(X_val=None, max_epochs=8, k=5):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SemanticNet(
        module=SemanticModel,
        module__vocab_size=3,  # DynamicParametersSetter sets the right value
        module__hidden_size=100,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=max_epochs,
        batch_size=1024,
        iterator_train=SequenceIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SequenceIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=partial(train_split, prep=preprocessor, X_val=X_val),
        device=device,
        predict_nonlinearity=partial(inference, k=k, device=device),
        callbacks=[
            DynamicParametersSetter(),
            skorch.callbacks.BatchScoring(
                partial(scoring, k=k),
                name="recall@5",
                on_train=False,
                lower_is_better=False,
            ),
            skorch.callbacks.ProgressBar(),
        ]
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return SemiSupervisedRecommender(full)


def inference(logits, k, device):
    probas = torch.softmax(logits.to(device), dim=-1)
    return torch.topk(probas, k=k, dim=-1)[-1].clone().detach()
