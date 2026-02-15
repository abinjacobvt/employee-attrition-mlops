import os
from src.preprocess import preprocess_data


from src.models.train_model import train_model, evaluate_model


def main():
    # ✔ Path to dataset (place IBM HR dataset CSV inside data/raw/)
    data_path = os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data_path)

    print("Training baseline model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    accuracy, report = evaluate_model(model, X_test, y_test)

    print("\n Model Performance")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    # ✔ Entry point for training pipeline
    main()
