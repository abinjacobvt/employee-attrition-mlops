from fastapi import FastAPI, HTTPException
import pandas as pd
import logging

from attrition.api.model_loader import load_model

app = FastAPI(title="Employee Attrition API")

logging.basicConfig(level=logging.INFO)

model = load_model()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: dict):
    try:
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

        return {
            "prediction": int(prediction)
        }

    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")