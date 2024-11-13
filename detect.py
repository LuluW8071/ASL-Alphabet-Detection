import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import json
from actions import ActionHandler

import sys 
sys.path.append('../')
from neuralnet import model as nn_model

# Setting device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the PyTorch model
model_path = 'assets/best_model.pth'
model_info = torch.load(model_path, map_location=torch.device('cpu'))
model = nn_model.EfficientNetB0(num_classes=36).to(device)
model.load_state_dict(model_info)
model.eval()

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Load class labels from the JSON file
with open('assets/class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Define transforms for preprocessing the hand image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Initialize variables
bbox = None
predicted_class = None

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB and process it with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw bounding box around the hand with some padding
            hand_landmarks_array = np.array([[data.x, data.y, data.z] for data in hand_landmarks.landmark])
            x_min, y_min, z_min = np.min(hand_landmarks_array, axis=0)
            x_max, y_max, z_max = np.max(hand_landmarks_array, axis=0)
            padding = 0.05  # Change this value to increase/decrease the padding
            x_min -= padding
            y_min -= padding
            x_max += padding
            y_max += padding
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(1, x_max), min(1, y_max)
            bbox = [int(x_min * frame.shape[1]), int(y_min * frame.shape[0]), int(x_max * frame.shape[1]), int(y_max * frame.shape[0])]

            # Extract the hand image
            hand_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Preprocess hand image to tensors
            pil_img = Image.fromarray(hand_img)
            pil_img = transform(pil_img).unsqueeze(0)

            # Inferencing to predict the class
            with torch.inference_mode():
                outputs = model(pil_img)

                _, predicted = torch.max(outputs, 1)
                confidence_value = F.softmax(outputs, dim=1).max().item()
                predicted_class = predicted.item()

                # Execute the corresponding action
                action = class_labels[str(predicted_class)]  # Convert predicted class to string
                handler = ActionHandler(confidence_value, action)
                handler.execute_action()

    # Draw the bounding box
    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)

    # Display the class name above the bounding box
    if predicted_class is not None:
        text = f"{action}: {confidence_value:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 20), (bbox[0] + text_width + 20, bbox[1]), (255, 255, 255), -1)
        cv2.putText(frame, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), thickness=2,lineType=cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('ASL Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
