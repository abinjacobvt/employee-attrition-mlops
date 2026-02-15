import numpy as np
from sklearn.linear_model import LogisticRegression

from attrition.models.train import train_model


def test_train_model_returns_logistic_regression():
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)


def test_train_model_fits_and_predicts():
    X = np.random.rand(30, 4)
    y = np.random.randint(0, 2, 30)

    model = train_model(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_train_model_score():
    X = np.random.rand(40, 3)
    y = np.random.randint(0, 2, 40)

    model = train_model(X, y)
    score = model.score(X, y)

    assert 0.0 <= score <= 1.0
