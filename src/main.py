import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from src.predict import predict_image

# Ensure Python recognizes the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Classifier")
        self.root.geometry("600x700")
        self.root.configure(bg="#f4f4f4")  # Light gray background

        # Header Label
        title_label = Label(
            root,
            text="🔍 AI Image Recognition",
            font=("Arial", 20, "bold"),
            bg="#f4f4f4",
            fg="#333",
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10, padx=20)

        # Frame for image display
        self.image_frame = Frame(root, bg="white", bd=2, relief="solid")
        self.image_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=20)

        self.image_label = Label(self.image_frame, bg="white")
        self.image_label.pack()

        # Prediction Label
        self.prediction_label = Label(
            root,
            text="Prediction: ",
            font=("Arial", 14, "bold"),
            bg="#f4f4f4",
            fg="black",
        )
        self.prediction_label.grid(row=2, column=0, columnspan=2, pady=10)

        # Buttons Frame
        button_frame = Frame(root, bg="#f4f4f4")
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        # Select Image Button
        self.upload_btn = Button(
            button_frame,
            text="📂 Select Image",
            command=self.upload_image,
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            padx=15,
            pady=5,
            relief="raised",
        )
        self.upload_btn.grid(row=0, column=0, padx=10, pady=5)

        # Exit Button
        self.exit_btn = Button(
            button_frame,
            text="❌ Exit",
            command=root.quit,
            font=("Arial", 12),
            bg="#e74c3c",
            fg="white",
            padx=15,
            pady=5,
            relief="raised",
        )
        self.exit_btn.grid(row=0, column=1, padx=10, pady=5)

    def upload_image(self):
        """Opens file dialog, allows user to select an image, and displays prediction."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )
        if not file_path:
            return  # No file selected

        # Predict the image
        predicted_label = predict_image(file_path)

        # Display Image
        image = Image.open(file_path)
        image = image.resize((350, 350))  # Resize for display
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
