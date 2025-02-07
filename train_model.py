# import YOLO model
from ultralytics import YOLO


# Load a model
model = YOLO('C:/Users/Spacelab3/Desktop/envs/Classifier/yolo11n-cls.pt') # load a pretrained model (recommended for training)

# Train the model
model.train(
        task="classify",
        device=0,
        data='C:/Users/Spacelab3/Desktop/envs/Classifier/ClassificationDatasetSplit', 
        epochs=5)
