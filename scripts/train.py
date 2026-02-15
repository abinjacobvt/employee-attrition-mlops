import os

from attrition.models.train import evaluate_model, train_model
from attrition.utils.preprocess import preprocess_data


def main():
    # Path to dataset
    data_path = os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data_path)

    print("Training baseline model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    accuracy, report = evaluate_model(model, X_test, y_test)

    print("\nModel Performance")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    main()
