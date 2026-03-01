import joblib


def load_model():
    """
    Load the trained machine learning model
    from the serialized joblib file.
    """
    return joblib.load("model.joblib")
