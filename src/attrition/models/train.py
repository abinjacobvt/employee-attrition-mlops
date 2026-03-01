"""
The baseline model for Employee Attrition Prediction.

This script:
1. Loads the dataset
2. Cleans unnecessary columns
3. Builds preprocessing + model pipeline
4. Trains a Logistic Regression model
5. Prints accuracy
"""

from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = Path("data/raw/employee_attrition.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def build_pipeline(df: pd.DataFrame):
    df = df.copy()

    # Encode target
    df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})

    # Drop useless / constant columns
    columns_to_drop = [
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]

    X = df.drop(columns=columns_to_drop)
    y = df["Attrition"]

    # Identify column types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return pipeline, X, y


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Building pipeline...")
    pipeline, X, y = build_pipeline(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = train_model(pipeline, X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()