# import pyautogui
import os
# from datetime import datetime
# import subprocess
import logging

# from PIL import ImageGrab
# import platform
# import tkinter as tk
# from tkinter import messagebox

class ActionHandler:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.logger = self.setup_logger()
    
    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Create a file handler and set the formatter
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "action_handler.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def execute_action(self):
        try:
            print(f"Predicted class: {self.predicted_class}")
        except Exception as e:
            self.logger.error("Error executing action: %s", e)
            print("An error occurred while executing the action.")

