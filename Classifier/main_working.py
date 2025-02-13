import torch
import cv2
import time
import logging
from pydantic import BaseModel, field_validator
from typing import Any
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from pathlib import Path
from torchvision.models import efficientnet_v2_s  # Import EfficientNet V2


# Assuming these paths and constants are set correctly
MODEL_PATH = r"C:\Users\Spacelab3\Desktop\envs\Classifier\efficientnetv2_finetuned.pth"  # Path to the ResNet model
LOGO_PATH = r"C:\Users\Spacelab3\Desktop\envs\Classifier\assets\logo.png"  # Path to the logo
ICON_PERSONA_PATH = r"C:\Users\Spacelab3\Desktop\envs\Classifier\assets\icons\smoke.png"  # Path to persona icon
ICON_PERSONA_PATH = r"C:\Users\Cesar Spacelab\Documents\Roberto Spacelab\git\Github\SkyProtector-IA\skyprotector-v1\v1\assets\icons\persona.png"
ICON_CARRO_PATH = r"C:\Users\Cesar Spacelab\Documents\Roberto Spacelab\git\Github\SkyProtector-IA\skyprotector-v1\v1\assets\icons\carro.png"
ICON_CAMIONETA_PATH = r"C:\Users\Cesar Spacelab\Documents\Roberto Spacelab\git\Github\SkyProtector-IA\skyprotector-v1\v1\assets\icons\camioneta.png"
ICON_CAMION_PATH = r"C:\Users\Cesar Spacelab\Documents\Roberto Spacelab\git\Github\SkyProtector-IA\skyprotector-v1\v1\assets\icons\camion.png"
ICON_MOTO_PATH = r"C:\Users\Cesar Spacelab\Documents\Roberto Spacelab\git\Github\SkyProtector-IA\skyprotector-v1\v1\assets\icons\moto.png"

class ModelManager:
    """
    Loads the appropriate model for image classification based on the task type
    and allocates the device (GPU or CPU).
    """
    def __init__(self):

        # Define image transformation pipeline for preprocessing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.transform =    transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
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
            "carro": cv2.flip(cv2.imread(ICON_CARRO_PATH, cv2.IMREAD_UNCHANGED), 1),
            "camioneta": cv2.imread(ICON_CAMIONETA_PATH, cv2.IMREAD_UNCHANGED),
            "camion": cv2.imread(ICON_CAMION_PATH, cv2.IMREAD_UNCHANGED),
            "moto": cv2.imread(ICON_MOTO_PATH, cv2.IMREAD_UNCHANGED),
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
    def add_counter(self, frame: np.ndarray, class_counts: dict, bg_transparency=0.6, bg_color=(0, 0, 0)) -> np.ndarray:
        """
        It adjusts the size of the icons and text based on frame size, adds class counters and 
        a background with transparency.

        :param frame: The frame to which the overlay will be added.
        :param class_counts: A dictionary with class names as keys and count values as the values.
        :param bg_transparency:  Transparency of the logo, 0 (fully transparent) to 1.
        :param bg_color: RGB values representing the background color.

        :return: The frame with the counter overlays.

        """
    
        frame_height, frame_width = frame.shape[0], frame.shape[1] 

        reference_width, reference_height = 3840, 2160
        scale_factor = min(frame_width / reference_width, frame_height / reference_height)
        icon_scale = scale_factor + 0.1 

        icon_spacing = int(80 * icon_scale) # 80
        text_scale = 1.2 * icon_scale # 1.2
        text_thickness = int(2.5 * icon_scale) # 3

        icon_width, icon_height = int(50 * icon_scale), int(50 * icon_scale) # 50 
        icon_size = icon_width, icon_height

        padding_x, padding_y = int(30 * icon_scale), int(25 * icon_scale) # 30, 25

        total_icons = len(self.class_names)
        text_width_estimate = int(30 * icon_scale) # 30 
        total_width = (padding_x * 2 + 
                      (icon_width + text_width_estimate) * total_icons + 
                      icon_spacing * (total_icons - 1)) 

        bg_height = icon_height + (padding_y * 2)
        bg_width = total_width + 5
       
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (bg_width, bg_height), bg_color, -1)
        frame = cv2.addWeighted(overlay, bg_transparency, frame, 1 - bg_transparency, 0, frame)

        for i, (class_id, class_name) in enumerate(self.class_names.items()):
            icon = self.icons.get(class_name)
            count = class_counts.get(class_name, 0)

            if icon is not None:
                if icon.shape[2] == 4:
                    icon_rgb = icon[:, :, :3]
                    icon_alpha = icon[:, :, 3] / 255.0
                else:
                    icon_rgb = icon
                    icon_alpha = np.ones(icon_rgb.shape[:2], dtype=float)

                icon_resized = cv2.resize(icon_rgb, icon_size)
                icon_alpha_resized = cv2.resize(icon_alpha, icon_size)

                icon_x_position = padding_x + i * (icon_width + text_width_estimate + icon_spacing)
                icon_y_position = padding_y

                if (icon_x_position + icon_width > frame.shape[1] or 
                    icon_y_position + icon_height > frame.shape[0]):
                    continue
                
                roi = frame[
                    icon_y_position:icon_y_position + icon_height,
                    icon_x_position:icon_x_position + icon_width
                ].copy()

                icon_alpha_3channel = np.stack([icon_alpha_resized] * 3, axis=-1)

                blended = np.uint8(icon_resized * icon_alpha_3channel + 
                                 roi * (1 - icon_alpha_3channel))

                frame[
                    icon_y_position:icon_y_position + icon_height,
                    icon_x_position:icon_x_position + icon_width ] = blended

                count_text = f"{count}"

                text_x_position = icon_x_position + icon_width + int(10 * icon_scale) # 10
                text_y_position = icon_y_position + int(icon_height * 0.75)

                cv2.putText(
                    frame, 
                    count_text,
                    (text_x_position, text_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (0, 0, 0),  
                    text_thickness + int(2 * icon_scale)
                )

                cv2.putText(
                    frame,
                    count_text,
                    (text_x_position, text_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (255, 255, 255),  
                    text_thickness
                )

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
class VideoProcessor:
    """
    Loads a video, applies model, annotates the image,
    and displays results with overlay.

    """

    def __init__(self):

        self.model_manager = ModelManager()
    
        self.active_model = self.model_manager.get_model(model_path=MODEL_PATH)
        self.active_model.eval()

        self.overlay_utils = OverlayUtils()
        #self.palette = self.overlay_utils.class_palette
        #self.class_names = self.overlay_utils.class_names

        # Clase 0: No Smokes
        # Clase 1: Smokes
        self.smoke_class_labels = {0: 'NoSmokes', 1: 'Smokes'}
        self.count = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"



    def preprocess_frame(self, frame):
        frame = Image.fromarray(frame)
        transform = transforms.Compose([
                    transforms.Resize((380, 380)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        return transform(frame).unsqueeze(0).to(self.device)

    def process_video(self, video_path: str, output_folder: str):
        """
        Loads the video, processes each frame with model object detection, 
        adds overlays, and saves the processed video.

        :param video_path: The path to the input video to be processed.
        :param output_path: The path where the output video will be saved.

        :return: The annotated image or None in case of an error.

        """

        video_path = Path(video_path)
        output_folder = Path(output_folder)

        input_video_name = video_path.stem
        input_extension = video_path.suffix
        output_video_path = output_folder / f"{input_video_name}{input_extension}"

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")

            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        reference_width, reference_height = 3840, 2160
        scale_factor = round(min(width / reference_width, height / reference_height),1)
        """
        box_annotator = sv.BoxCornerAnnotator(
            color=self.palette,
            thickness=int(7 * scale_factor),
            corner_length=int(30 * scale_factor),
            color_lookup=sv.ColorLookup.CLASS
        )
        """

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break  

                try:
                    # results = self.eye_v_model(frame, conf=0.40, verbose=False)

                    # detections = sv.Detections.from_ultralytics(results[0])
                    # annotated_frame = box_annotator.annotate(scene=frame, detections=detections)

                    # class_counts = {class_name: 0 for class_name in self.class_names.values()}
                    # for i, class_name in enumerate(self.class_names.values()):
                    #     class_detections = detections[detections.class_id == i]
                    #     class_counts[class_name] = len(class_detections)

                    # validator = Validator(frame=annotated_frame, counts=class_counts)
                    # val_frame_logo = self.overlay_utils.add_logo(validator.frame)

                    # out.write(val_frame_logo)

                    # Fire Classification
                    input_tensor = self.preprocess_frame(frame)#.to(self.model_manager.device)

                    with torch.no_grad():
                        outputs = self.active_model(input_tensor)

                    # Obtener la clase predicha
                    _, predicted_class = torch.max(outputs, 1)
                    fire_text = self.smoke_class_labels[predicted_class.item()]
                    # print(f'Clase predicha: {predicted_class.item()}')
                    print(f"Prediccion: {self.smoke_class_labels[predicted_class.item()]}")
                    self.count += 1

                    if(predicted_class == 1):
                        prediction = False
                    elif(predicted_class == 0):
                        prediction = True

                    # Agrega texto de predicción de fuego
                    text_scale = 5 * scale_factor
                    text_thickness = int(10 * scale_factor)
                    text_color = (0, 0, 255) if prediction else (0, 255, 0)
                    text_position = (200, 50)  # Posición del texto

                    cv2.putText(
                        frame,
                        fire_text,
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        text_color,
                        text_thickness
                    )

                    val_frame_logo = self.overlay_utils.add_logo(frame)

                    # Escribe el frame procesado en el video de salida
                    out.write(val_frame_logo)


                except RuntimeError as e:
                    logging.error(f"Error during video processing: {e}")

                    continue

        finally:
            cap.release()
            out.release()

        return prediction, val_frame_logo
if __name__ == "__main__":
    # Test the image classification processor
    ia_frame_processor = ImageClassificationProcessor()
    IMAGE_PATH = "C:/Users/Spacelab3/Desktop/envs/Classifier/VideoTest2/VideoTestVideo_Frame_67.jpg"
    frame = cv2.imread(IMAGE_PATH)
        
    start_time = time.time()
    ann_frame = ia_frame_processor.process_frame(frame)
    end_time = time.time()

    processing_time = end_time - start_time
    logging.info(f"Total processing time: {processing_time:.2f} seconds")
    """
    resize_ann_frame = cv2.resize(ann_frame, (1280, 720))
    plt.imshow(cv2.cvtColor(resize_ann_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hides the axes and ticks
    plt.show()
    """
    video_processor = VideoProcessor()
    video_path = Path(r"C:\Users\Spacelab3\Desktop\envs\Classifier\Videos\DroneVideoNightCity - Copy.mp4")
    output_folder = Path(r"C:\Users\Spacelab3\Desktop\envs\Classifier\Videos\Test")
    start_time = time.time()  
    prediction_fire, video = video_processor.process_video(video_path, output_folder)
    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Total processing time: {processing_time:.2f} seconds")