import cv2
import numpy as np


def preprocess_image(image_path):
    """
    1. Load an image using OpenCV.
    2. Convert it to RGB format.
    3. Resize it to 224x224 (required for MobileNetV2).
    4. Normalize pixel values to range [0,1].
    5. Expand dimensions to fit model input shape.
    """
    image = cv2.imread(image_path)  # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (224, 224))  # Resize to 224x224
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
