import tensorflow as tf
import numpy as np
import json
import urllib.request
from src.preprocess import preprocess_image

# Load the MobileNetV2 model (pre-trained on ImageNet)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Load ImageNet class labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(LABELS_URL) as f:
    labels = json.load(f)


def get_label(class_id):
    """Fetch human-readable label from ImageNet class ID."""
    return labels[str(class_id)][1]


def predict_image(image_path):
    """
    1. Preprocess the input image.
    2. Use MobileNetV2 to predict the object class.
    3. Return the predicted label.
    """
    image = preprocess_image(image_path)
    preds = model.predict(image)  # Make a prediction
    top_pred = np.argmax(preds)  # Get the highest probability class
    label = get_label(top_pred)  # Convert to human-readable label
    return label
