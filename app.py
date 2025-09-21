# app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
import os
from torchvision import datasets

# 🔧 Resolve path to Training folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")

# 📁 Load training dataset just to get class names
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
class_names = train_dataset.classes  # List of folder names = fruit labels

# 🧠 Define deeper CNN model
class FruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FruitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 12 * 12, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 📦 Load saved weights
model = FruitClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "fruit_model.pth"), map_location=torch.device("cpu")))
model.eval()

from PIL import ImageOps
import numpy as np

def preprocess_for_detection(pil_image):
    # Convert to grayscale
    gray = ImageOps.grayscale(pil_image)

    # Convert to NumPy array
    arr = np.array(gray)

    # Threshold to isolate fruit (works well on white backgrounds)
    mask = arr < 240  # anything darker than near-white

    # Find bounding box of non-white pixels
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pil_image.resize((100, 100))  # fallback

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # add 1 to include edge

    # Crop and resize
    cropped = pil_image.crop((x0, y0, x1, y1)).resize((100, 100))
    return cropped

# 🎨 Streamlit UI
st.title("🍎 Fruit Classifier")
st.write("Upload a fruit image and get its predicted class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Original Uploaded Image", use_column_width=True)

    # 🔍 Preprocess image for detection
    processed_image = preprocess_for_detection(image)
    st.image(processed_image, caption="🧠 Preprocessed Image Used for Prediction", use_column_width=True)

    # 🔢 Transform and predict
    input_tensor = transform(processed_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = class_names[predicted.item()]

    st.success(f"🍓 Predicted Fruit: **{predicted_label}**")