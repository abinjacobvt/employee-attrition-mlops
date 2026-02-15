import pandas as pd

from attrition.utils.preprocess import preprocess_data


def test_preprocess_removes_nulls():
    df = pd.DataFrame({"Age": [25, None, 30], "Salary": [50000, 60000, None]})

    processed = preprocess_data(df)

    assert processed.isnull().sum().sum() == 0
