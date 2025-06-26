import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys


# Define the PyTorch Classification Model for Inference
class PyTorchClassificationModel:
    def __init__(
        self,
        model_path,
        labels,
        transform,
        classes,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_path: Path to the PyTorch model file (.pth)
            labels: List of label indices
            transform: Transformations to preprocess the input images
            classes: List of class names
            device: 'cuda' or 'cpu'
        """
        self.device = device  # Initialize device
        self.model = self.load_model(model_path, len(classes))  # Load model
        self.model.eval()  # Set model to evaluation mode
        self.labels = labels
        self.transform = transform
        self.classes = classes
        self.model = self.model.to(self.device)

    def load_model(self, model_path, num_classes):
        """Load the trained PyTorch model."""
        model = models.resnet18(weights=None)  # Load ResNet18 architecture
        model.fc = nn.Linear(
            model.fc.in_features, num_classes
        )  # Adjust for the number of classes

        # Load state dict while ignoring size mismatches for final layer
        state_dict = torch.load(model_path, map_location=self.device)
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k not in ["fc.weight", "fc.bias"]
        }
        model.load_state_dict(filtered_state_dict, strict=False)  # Avoid strict mode

        return model

    def preprocess(self, image_path):
        """Preprocess the input image: Resize, Normalize, Convert to Tensor."""
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image.to(self.device)

    def predict(self, image_path):
        """Perform inference on a single image."""
        # Preprocess the image
        input_tensor = self.preprocess(image_path)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(
                outputs[0], dim=0
            )  # Convert logits to probabilities

        # Map probabilities to class labels
        probabilities = probabilities.cpu().numpy()
        label_to_probabilities = [
            (self.labels[i], float(probabilities[i])) for i in range(len(probabilities))
        ]
        sorted_probs = sorted(label_to_probabilities, key=lambda x: x[1], reverse=True)

        # Predicted class
        predicted_label = sorted_probs[0][0]
        predicted_probability = sorted_probs[0][1]
        predicted_class = self.classes[predicted_label]

        return predicted_class, predicted_probability


# Define the transforms for input image
image_transforms = transforms.Compose(
    [
        transforms.Resize(224),  # Resize the image to 224x224
        transforms.CenterCrop(224),  # Crop the center
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize (ImageNet stats)
    ]
)

# Define class labels
labels = list(range(10))  # Adjusted to 10 classes
class_labels = {
    0: "Safe driving",
    1: "Texting (right hand)",
    2: "Talking on the phone (right hand)",
    3: "Texting (left hand)",
    4: "Talking on the phone (left hand)",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger(s)",
}

# Path to the trained model
model_path = r"C:\Users\Amira\Driver-Monitoring-System\activity detection aya's edition\fine_tuned_resnet18_3.pth"

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PyTorchClassificationModel(
    model_path, labels, image_transforms, class_labels, device=device
)


# Function to select an image file using a file dialog
def select_image():
    """Opens a file dialog to select an image file."""
    app = QApplication(sys.argv)
    image_path, _ = QFileDialog.getOpenFileName(
        None, "Select an Image", "", "Images (*.png *.jpg *.jpeg)"
    )
    return image_path


# Select an image manually or use a fixed path
image_path = select_image()  # Let user select image

if not image_path:
    print("No image selected. Exiting.")
    sys.exit()

# Perform inference
predicted_class, predicted_prob = model.predict(image_path)

# Display the image and prediction
image = Image.open(image_path)
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(f"Predicted: {predicted_class} ({predicted_prob * 100:.2f}%)")
plt.axis("off")
plt.show()

# Print result
print(f"Predicted Class: {predicted_class} (Confidence: {predicted_prob * 100:.2f}%)")
