# American Sign Language Detection using EfficientNetB0

This project utilizes the **EfficientNetB0** CNN architecture for image classification + detection of American Sign Language (ASL) alphabet images.

## Setup and Installation

### 1. Install Required Dependencies


```bash
pip install -r requirements.txt
pip install kaggle
```

### 2. Download the ASL Alphabet Dataset

To download the dataset directly from Kaggle, ensure that your `kaggle.json` API key is properly set up. Follow these steps:

1. Move your `kaggle.json` file to the directory:
    - On Windows: `C:/Users/{your_username}/.kaggle/`
    - On macOS/Linux: `~/.kaggle/`

2. Run the following command to download the dataset:
    ```bash
    kaggle datasets download -d debashishsau/aslamerican-sign-language-aplhabet-dataset
    ```

3. Unzip the dataset:
    ```bash
    unzip aslamerican-sign-language-aplhabet-dataset.zip
    ```

### 3. Train the Model

You can train the model locally using the dataset:

- Ensure you have at least **8GB of VRAM** on a CUDA-supported GPU.
- Open the `ASL_Alphabet_Classification.ipynb` notebook and follow the instructions for training.

> [!Note]
> Training may take around 1 hour to complete. After training, the model file `efficientnet_model.pth` will be saved under the `models/` directory.

### 4. Running the Demo

#### Streamlit Demo (Image Classification):
```bash
cd demo
streamlit run main.py
```

#### Live Detection Demo:
```bash
python3 detect.py
```

> [!Note]
> If you don't have a webcam, you can use the DroidCam app to turn your mobile phone into a webcam. Logs will be saved in the `action_handler.log` file.

### 5. Pre-trained Model

You can use the pre-trained model `efficientnet_model.pth` located in the `models/` directory to perform inference on ASL images in your local environment or in Google Colab.

---

Feel free to report any issues you encounter.  
Don't forget to ⭐ the repo!
