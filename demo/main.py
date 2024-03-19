import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from utils import *

import torch
from torch import nn
import sys
sys.path.append('../')
from neuralnet import model as nn_model


def main():
    st.title("American Sign Language Classifier")

    # Setting device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # st.write(f"Device: {device}")

    # Load Trained Model
    MODEL_LOAD_PATH = "../model/efficientnet_model.pth"
    model_info = torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu'))

    # Instantiate the EfficientNet model
    model = nn_model.EfficientNetB0(num_classes=29).to(device)

    # Define paths
    data_path = Path("test_images/")

    # Image upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image
        custom_image_path = data_path / uploaded_file.name
        with open(custom_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and preprocess the image
        custom_image_transformed = load_and_preprocess_image(custom_image_path)

        # Load the model
        model.load_state_dict(model_info)
        model.eval()

        # Predict the label for the image
        # Define the class names from 'A' to 'Z'
        letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]

        # Additional class names: 'del', 'nothing', 'space'
        additional_classes = ['del', 'nothing', 'space']

        # Concatenate all class names
        class_names = np.array(letters + additional_classes)
        predicted_label, image_pred_probs = predict_image(model,
                                                          custom_image_transformed,
                                                          class_names)


        # Prediction result section
        st.markdown(
            f'<h3 style="color: green;">Prediction Result</h3>', 
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 3])

        # Display prediction label and confidence rate on the left column
        col1.write(f"Predicted Sign: **{predicted_label[0]}**")
        col1.write(f"Confidence: **{image_pred_probs.max()* 100:.2f}%**")

        # Display the uploaded image on the right column
        with col2:
            image = Image.open(custom_image_path)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
if __name__ == "__main__":
    main()