import os
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2  # OpenCV for capturing video frames
from PIL import Image
from torchvision.models import efficientnet_v2_s  # Import EfficientNet V2

# Load the saved model
# Load the EfficientNetB0 model
#model = models.efficientnet_b2(pretrained=True)

# Modify the classifier to match the number of classes (2 for this example)
#model.classifier = nn.Linear(model.classifier[1].in_features, 2)
### C:\Users\Spacelab3\Desktop\envs\Classifier\efficientnetv2_finetuned.pth
# Load the trained weights          C:\Users\Spacelab3\Desktop\envs\Classifier\image-classification-pytorch\weights\best.pt 'C:/Users/Spacelab3/Desktop/envs/Classifier/image-classification-pytorch/weights/best.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_v2_s(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Load the trained weights
model.load_state_dict(torch.load('C:/Users/Spacelab3/Desktop/envs/Classifier/efficientnetv2_finetuned.pth'))
model = model.to(device)
model.eval()  # Set the model to evaluation mode


# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names
class_names = ['NoSmokes', 'Smokes']

# Initialize OpenCV camera capture (0 for default webcam)
cap = cv2.VideoCapture(0)
image_path = r"C:\Users\Spacelab3\Desktop\envs\Classifier\camera"
# Initialize EMA
alpha = 0.9
new_prediction = []
ema_current = None
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    correct_predictions = 0
    total_predictions = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
        image_store = f"{image_path}\camera_photo{total_predictions}.jpg"
        # Convert the frame to a PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #image2 = Image.open(frame)
        #cv2.imwrite(image_store,frame)
        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        # probabilities = [[0.124, 0.876]]

        predicted_class = torch.argmax(probabilities, dim=1).item()  # 1
        predicted_prob = probabilities[0, predicted_class].item()   # 0.876
        
        # Get the predicted class
        _, predicted_class = output.max(1)
        
        # Map the predicted class to the class name
        predicted_class_name = class_names[predicted_class.item()]
        
        # Assuming true class is 'NoSmokes' (you can modify this logic if needed)
        true_class = "NoSmokes"
        
        if true_class == predicted_class_name:
            correct_predictions += 1
        """
        else:
            # If the prediction is incorrect, show the image with the predicted class
            image = np.array(image)
            plt.imshow(image)
            plt.axis('off')
            plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
            plt.show()
        """
        total_predictions += 1
        # For each new prediction in real-time:
        new_prediction.append(predicted_class)  # Example new prediction
        if ema_current is None:
            ema_current = predicted_class  # Initialize EMA with the first prediction
        elif len(new_prediction) > 2:
            ema_current = alpha * sum(new_prediction) + (1 - alpha) * ema_current
            new_prediction.pop(0)

        # Smoothed prediction for the current frame:
        print(ema_current)

        #ema_current_class_name = class_names[int(ema_current)]


        # Display the result on the camera feed (optional)  , Filtered {ema_current_class_name}
        cv2.putText(frame, f'Predicted: {predicted_class_name}, Probability: {predicted_prob}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame: ", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Calculate and print accuracy
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f'Accuracy: {accuracy:.2f}%  of {total_predictions} frames')

# Release the camera and close any OpenCV windows
cap.release()
