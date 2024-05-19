""" Download and extract dataset from Kaggle """

import argparse
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_and_extract_dataset():
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Set the dataset name and paths
    dataset_name = 'debashishsau/aslamerican-sign-language-aplhabet-dataset'
    zip_file_name = f"{dataset_name.split('/')[1]}.zip"
    zip_file_path = os.path.join(os.getcwd(), zip_file_name)
    extract_path = os.path.join(os.getcwd(), "dataset")

    # Check if the zip file already exists
    if os.path.exists(zip_file_path):
        print("Dataset zip file already exists. Skipping download.")
    else:
        api.dataset_download_files(dataset_name)

    # Extract zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Remove the zip file after extraction
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract dataset from Kaggle")
    parser.add_argument("--json_path", type=str, required=True, help="Relative path to the Kaggle JSON file")
    args = parser.parse_args()

    # Set KAGGLE_CONFIG_DIR environment variable to the parent directory of the JSON file
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(args.json_path)

    download_and_extract_dataset()
