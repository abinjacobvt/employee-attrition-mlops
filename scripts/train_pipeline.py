from pathlib import Path
import joblib

import mlflow
import mlflow.sklearn

from attrition.models.train import load_data, build_pipeline, train_model
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/employee_attrition.csv")
MODEL_OUTPUT_PATH = Path("model.joblib")


def main():
    mlflow.set_experiment("Attrition_Experiments")

    with mlflow.start_run(run_name="LR_pipeline_v3"):

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

        print("Evaluating...")
        accuracy = model.score(X_test, y_test)

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log FULL pipeline to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model locally for deployment
        joblib.dump(model, MODEL_OUTPUT_PATH)
        print(f"Model saved locally at {MODEL_OUTPUT_PATH}")

        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()