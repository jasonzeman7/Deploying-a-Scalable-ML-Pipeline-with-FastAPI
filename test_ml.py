import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, load_model

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("data/census.csv")
    return df


@pytest.fixture(scope="module")
def trained_model_and_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(X_train, y_train)
    return model, X_test, y_test


def test_train_model_returns_random_forest(trained_model_and_data):
    """Test that train_model returns a RandomForestClassifier."""
    model, _, _ = trained_model_and_data
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_numpy_array(trained_model_and_data):
    """Test that inference returns a numpy array of predictions."""
    model, X_test, _ = trained_model_and_data
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)


def test_compute_model_metrics_range(trained_model_and_data):
    """Test that metrics are all between 0 and 1."""
    model, X_test, y_test = trained_model_and_data
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0