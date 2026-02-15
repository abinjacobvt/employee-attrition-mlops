from pathlib import Path

import mlflow
import mlflow.sklearn

from attrition.models.train import load_data, preprocess, train_model

DATA_PATH = Path("data/raw/employee_attrition.csv")


def main():
    mlflow.set_experiment("Attrition_Experiments")

    with mlflow.start_run(run_name="LR_baseline_v1"):
        print("Loading data...")
        df = load_data(DATA_PATH)

        print("Preprocessing...")
        X_train, X_test, y_train, y_test = preprocess(df)

        print("Training model...")
        model = train_model(X_train, y_train)

        print("Evaluating...")
        accuracy = model.score(X_test, y_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
