from pathlib import Path
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi



def download_employee_attrition_dataset():

    # Paths
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
    zip_path = data_dir / "ibm-hr-analytics-attrition-dataset.zip"

    # Authenticate with Kaggle

    api = KaggleApi()
    api.authenticate()

    # Download dataset (ZIP)

    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(
        dataset_name,
        path=data_dir,
        unzip=False
    )

    # Extract ZIP

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

 
    # Rename CSV to standard name
    original_csv = data_dir / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    final_csv = data_dir / "employee_attrition.csv"

    if original_csv.exists():
        original_csv.rename(final_csv)
        print("CSV renamed to employee_attrition.csv")
    else:
        raise FileNotFoundError(
            "Expected CSV file not found after extraction."
        )

    print("Dataset downloaded directly from Kaggle successfully!")


if __name__ == "__main__":
    download_employee_attrition_dataset()
