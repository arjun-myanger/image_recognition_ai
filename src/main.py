import os
import sys

# Ensure Python recognizes the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from src.predict import predict_image


# Ensure Python recognizes the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Classifier")
        self.root.geometry("500x600")
        self.root.configure(bg="white")

        # Title Label
        self.label = tk.Label(
            root, text="AI Image Recognition", font=("Arial", 16, "bold"), bg="white"
        )
        self.label.pack(pady=10)

        # Button to select image
        self.upload_btn = tk.Button(
            root,
            text="Select Image",
            command=self.upload_image,
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
        )
        self.upload_btn.pack(pady=10)

        # Label for image display
        self.image_label = tk.Label(root, bg="white")
        self.image_label.pack(pady=10)

        # Prediction Label
        self.prediction_label = tk.Label(
            root, text="", font=("Arial", 14, "bold"), bg="white", fg="black"
        )
        self.prediction_label.pack(pady=10)

    def upload_image(self):
        """Opens file dialog, allows user to select image, and displays prediction."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )
        if not file_path:
            return  # No file selected

        # Predict the image
        predicted_label = predict_image(file_path)

        # Display Image
        image = Image.open(file_path)
        image = image.resize((300, 300))  # Resize for display
        img = ImageTk.PhotoImage(image)

        self.image_label.config(image=img)
        self.image_label.image = img

        # Display Prediction
        self.prediction_label.config(text=f"Prediction: {predicted_label}", fg="green")


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
