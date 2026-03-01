import logging

import pandas as pd
from fastapi import FastAPI, HTTPException

# Import model loading utility
from attrition.api.model_loader import load_model

# Initialize FastAPI application
app = FastAPI(title="Employee Attrition API")

# Configure basic logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)

# Load trained ML model at startup
model = load_model()


@app.get("/health")
def health():
    # Simple health check endpoint to verify API is running
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: dict):
    """
    Predict employee attrition.
    Expects input features as JSON dictionary.
    """
    try:
        # Convert incoming JSON to pandas DataFrame
        input_df = pd.DataFrame([data])

        # Perform prediction using trained model
        prediction = model.predict(input_df)[0]

        # Return prediction result
        return {"prediction": int(prediction)}

    except Exception as e:
        # Log error details for debugging
        logging.error(str(e))

        # Return HTTP 500 if prediction fails
        raise HTTPException(status_code=500, detail="Prediction failed")
