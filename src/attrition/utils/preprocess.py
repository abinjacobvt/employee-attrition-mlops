import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing function.
    Fills missing values with 0.
    """
    df = df.copy()
    df = df.fillna(0)
    return df
