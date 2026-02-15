"""
The baseline model for Employee Attrition Prediction.

This script:
1. Loads the dataset
2. Preprocesses the data
3. Trains a Logistic Regression model
4. Prints accuracy
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DATA_PATH = Path("data/raw/employee_attrition.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Encode target
    label_encoder = LabelEncoder()
    df["Attrition"] = label_encoder.fit_transform(df["Attrition"])

    # Drop non-numeric columns except target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X = X.select_dtypes(include=["number"])

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
