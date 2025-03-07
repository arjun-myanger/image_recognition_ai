import os
import cv2
import matplotlib.pyplot as plt
from src.predict import predict_image

# Folder where images are stored
IMAGE_FOLDER = "images/"


def display_prediction(image_path):
    """
    1. Predict the object in the image.
    2. Display the image with the predicted label.
    """
    predicted_label = predict_image(image_path)

    # Load original image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Show image with prediction
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_label}")
    plt.show()

    print(f"Predicted Object: {predicted_label}")


if __name__ == "__main__":
    # Get all image files from the images/ folder
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png"))
    ]

    if not image_files:
        print("No images found in the 'images/' folder. Add some and try again.")
    else:
        for image_file in image_files:
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            print(f"\nProcessing: {image_file}")
            display_prediction(image_path)
