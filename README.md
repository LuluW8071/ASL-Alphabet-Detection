# American Sign Language Image Classification using EfficientNetB0

To utilize the Kaggle API, ensure you have a Kaggle account and a Kaggle API token saved as a JSON file named `kaggle.json`. Run the script with the `--json_path` argument pointing to the location of your `kaggle.json` file for authentication. For a step-by-step tutorial on obtaining your Kaggle API token, click [here](https://christianjmills.com/posts/kaggle-obtain-api-key-tutorial/).

To separately download the ASL Dataset, you can click [here](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset). 

Alternatively, you can use the provided `dataset_downloader.py` script to automatically download the dataset for you. Simply run the script with the appropriate arguments, and it will handle the download and extraction process for you.

## Uploading Python Scripts and Kaggle API Key to Colab
To begin, navigate to your Colab environment and ensure you have all the necessary Python scripts from the `neuralnet` directory and the `script` directory. Additionally, make sure you have your Kaggle API JSON file ready for uploading.

1. **Accessing Colab Environment**: Open your Colab notebook and ensure you are connected to the runtime.

2. **Upload Python Scripts**: Click on the "Files" icon on the left sidebar and select "Upload". Navigate to the directories containing your Python scripts (`neuralnet` and `script`), select all relevant files, and upload them to your Colab environment.

3. **Upload Kaggle API JSON File**: Follow the same process to upload your Kaggle API JSON file. This file is named `kaggle.json` and contains your Kaggle API token for authentication.

Once all files are uploaded, you can proceed to execute your Python scripts within the Colab environment, utilizing the uploaded scripts and the Kaggle API token for accessing datasets. 

## Train Model
All the required **python commands** to run the scripts are in notebook `ASL_Alphabet_Classification.ipynb`.

### Note
- It may take around 1 hour to train.
- After training is completed, download model file named `efficientnet_model.pth` under `models` directory.

## Training the Model Locally

If you wish to train the model on your local device, please ensure that 
- You have at least **8GB** of **VRAM** on a CUDA-supported GPU device. 
- Kaggle API token (`kaggle.json`) is moved to the directory `"C:/Users/{your_username}/.kaggle/"` on Windows, or `"~/.kaggle/"` on Linux or macOS. 
- To provide the absolute path of the new `kaggle.json` location when running the script with the `--json_path` argument for `dataset_downloader.py
- `Pytorch Cuda` version is installed with appropriate version of `Nvidia CUDA Toolkit`.

## Demo Usage

To run the demo:

- **Install Dependencies**:
    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

    ```bash
    pip install -r requirements.txt
    ```
- **Run `main.py` for Demo**:
    - To run demo in streamlit:
        ```bash
        cd demo
        streamlit run main.py
        ```

    - To run demo in opencv/mediapipe:
        ```bash
        python3 detect.py
        ```

    >Note: logs are created on the file named `action_handler.log`

<i>This project utilizes the `EfficientNetB0` CNN architecture model for image classification. The pre-trained model is available in the `model/` directory. You can load the model file `efficientnet_model.pth` on Colab or a local device to perform inference on American Sign Language images.</i>

---
Feel free to report any issues you encounter. </br>
<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="25px" /> Don't forget to star the repo :)
