import os
import logging
from datetime import datetime


class ActionHandler:
    def __init__(self, confidence_value, predicted_class):
        self.confidence_value = confidence_value
        self.predicted_class = predicted_class
        self.logger = self.setup_logger()
    
    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_handler.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def execute_action(self):
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"👀 Predicted class: {self.predicted_class} | 🚀 Confidence rate: {self.confidence_value}")
            self.logger.info("Predicted class: %s | Confidence rate: %s", self.predicted_class, self.confidence_value)
        except Exception as e:
            self.logger.error("Error executing action: %s", e)
            print("An error occurred while executing the action.")