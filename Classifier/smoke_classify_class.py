import torch
import cv2
import time
import logging
from pydantic import BaseModel, field_validator
from typing import Any
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import efficientnet_v2_s  # Import EfficientNet V2


# Assuming these paths and constants are set correctly
MODEL_PATH = r"C:\Users\Spacelab3\Desktop\envs\Classifier\efficientnetv2_finetuned.pth"  # Path to the ResNet model
LOGO_PATH = r"C:\Users\Spacelab3\Desktop\envs\Classifier\assets\logo.png"  # Path to the logo
ICON_PERSONA_PATH = r"C:\Users\Spacelab3\Desktop\envs\Classifier\assets\icons\smoke.png"  # Path to persona icon

class ModelManager:
    """
    Loads the appropriate model for image classification based on the task type
    and allocates the device (GPU or CPU).
    """
    def __init__(self):

        # Define image transformation pipeline for preprocessing
        self.transform =    transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomHorizontalFlip(),
                            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def get_model(self, model_path):
        """
        Returns the ResNet model for classification.
        """
        self.model = efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path,weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

        return self.model

    def predict(self, frame: np.ndarray):
        """
        Predict the class of the image using the ResNet model.

        :param frame: The input image for classification.
        :return: Predicted class label and probability.
        """
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_class].item()
        
        return predicted_class, predicted_prob

class OverlayUtils:
    """
    Handles the addition of logos, icons, and class labels to frames.
    """
    def __init__(self):
        self.logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
        self.icons = {
            "persona": cv2.imread(ICON_PERSONA_PATH, cv2.IMREAD_UNCHANGED),
        }
        
    def add_logo(self, frame: np.ndarray, position='top-right', logo_transparency=0.4) -> np.ndarray:
        """
        Resize and add the logo to the frame with transparency.

        :param frame: The frame to which the overlay will be added.
        :param position: Position to place the logo ('top-left', 'top-right', etc.).
        :param logo_transparency: Transparency level of the logo.
        :return: Frame with the logo overlay.
        """
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        scale = 0.2
        scale_factor = min(frame_width, frame_height) * scale
        logo_size = (int(scale_factor), int(scale_factor))
        logo_resized = cv2.resize(self.logo, logo_size)

        logo_height, logo_width = logo_resized.shape[0], logo_resized.shape[1]
        positions = {
            'top-left': (0, 0),
            'top-right': (frame_width - logo_width - 0, 0),
            'bottom-left': (10, frame_height - logo_height),
            'bottom-right': (frame_width - logo_width, frame_height - logo_height),
        }
        top_left_x, top_left_y = positions.get(position, (0, 0))

        logo_rgb = logo_resized[:, :, :3]
        logo_alpha = logo_resized[:, :, 3] / 255.0 * (1 - logo_transparency)
        
        roi = frame[top_left_y:top_left_y + logo_height, top_left_x:top_left_x + logo_width].copy()
        blended = (logo_rgb * logo_alpha[..., None] + roi * (1 - logo_alpha[..., None])).astype(np.uint8)
        frame[top_left_y:top_left_y + logo_height, top_left_x:top_left_x + logo_width] = blended

        return frame

    def add_class_label(self, frame: np.ndarray, class_name: str, prob: float) -> np.ndarray:
        """
        Adds the predicted class label and probability to the frame.

        :param frame: The frame to which the overlay will be added.
        :param class_name: The name of the predicted class.
        :param prob: The probability of the prediction.
        :return: Frame with the class label overlay.
        """
        label = f"{class_name}: {prob:.2f}"
        cv2.putText(frame, label, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        return frame

class Validator(BaseModel):
    """
    Ensures that the frame is correctly formatted and valid.
    """
    frame: Any

    @field_validator("frame")
    def frame_validation(cls, value):
        if not isinstance(value, np.ndarray) or value.size == 0:
            raise ValueError("Frame must be a non-empty ndarray")
        return value

class ImageClassificationProcessor:
    """
    Processes an image using a ResNet model, applies classification, and overlays results.
    """
    def __init__(self):
        self.model_manager = ModelManager()
        self.active_model = self.model_manager.get_model(model_path=MODEL_PATH)
        self.overlay_utils = OverlayUtils()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a frame, performs image classification, and adds overlays.

        :param frame: The frame to be processed.
        :return: Frame with classification results and overlays.
        """
        #  Convert the NumPy array frame from cv2 to a PIL Image
        image = Image.fromarray(frame)

        predicted_class, predicted_prob = self.model_manager.predict(image)
        
        # Convert predicted class to a human-readable label
        class_array = ["No Smokes","Smokes"]
        class_name = "Class_" + class_array[int(predicted_class)]  # You can map it to a real label if you have the labels.
        
        # Add overlays (logo and class label)
        frame_with_logo = self.overlay_utils.add_logo(frame)
        annotated_frame = self.overlay_utils.add_class_label(frame_with_logo, class_name, predicted_prob)

        return annotated_frame

if __name__ == "__main__":
    # Load the image classification processor class
    ia_frame_processor = ImageClassificationProcessor()

    # Input path of image to process
    IMAGE_PATH = "C:/Users/Spacelab3/Desktop/envs/Classifier/VideoTest2/VideoTestVideo_Frame_67.jpg"

    # Read, classify the image
    frame = cv2.imread(IMAGE_PATH)
    ann_frame = ia_frame_processor.process_frame(frame)

    # Show the image
    plt.imshow(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hides the axes and ticks
    plt.show()
