import pandas as pd
import numpy as np
import pytest
from pathlib import Path
 
from attrition.models import train
 
 
def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        train.load_data(Path("fake.csv"))
 
 
def test_preprocess_function():
    df = pd.DataFrame({
        "Attrition": ["Yes", "No", "Yes", "No"],
        "Age": [25, 30, 35, 40],
        "Salary": [50000, 60000, 70000, 80000],
    })
 
    X_train, X_test, y_train, y_test = train.preprocess(df)
 
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert set(y_train.unique()).issubset({0, 1})
 
 
def test_train_model():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
 
    model = train.train_model(X, y)
 
    assert model is not None