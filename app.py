# app.py
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

from torchvision import datasets

# Load training dataset just to get class names
train_dataset = datasets.ImageFolder(root="/Training")
class_names = train_dataset.classes  # List of folder names = fruit labels
# Load model
class FruitClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(FruitClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 25 * 25, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load saved weights
model = FruitClassifier(num_classes=216)
model.load_state_dict(torch.load("fruit_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Streamlit UI
st.title("🍎 Fruit Classifier")
st.write("Upload a fruit image and get its predicted class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
        predicted_label = class_names[predicted.item()]
        st.success(f"Predicted Fruit: {predicted_label}")