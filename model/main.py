
from model.data import read_dataset
from model.semantic import build_model
from model.metrics import recall_score


def main(name="popularity"):
    train = read_dataset("data/train.csv")

    model = build_model()
    model.fit(train)

    # print(recall_score(model, train))


if __name__ == '__main__':
    main()
