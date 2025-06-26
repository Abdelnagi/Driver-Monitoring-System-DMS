
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Define the PyTorch Classification Model for Inference
class PyTorchClassificationModel:
    def __init__(
        self, model_path, labels, transform, classes, image_size=224, device="CPU"
    ):
        """
        Args:
            model_path: Path to the PyTorch model file (.pth)
            labels: List of label indices
            transform: Transformations to preprocess the input images
            classes: List of class names
            image_size: Size of input images
            device: 'cuda' or 'cpu'
        """
        self.device = device  # Initialize device first
        self.model = self.load_model(
            model_path, len(classes)
        )  # Pass self.device to load_model
        self.model.eval()  # Set model to evaluation mode
        self.labels = labels
        self.image_size = image_size
        self.transform = transform
        self.classes = classes
        self.model = self.model.to(self.device)

    def load_model(self, model_path, num_classes):
        """
        Load the trained PyTorch model.
        """
        model = models.resnet18(weights=None)  # Load ResNet18 architecture
        model.fc = nn.Linear(
            model.fc.in_features, num_classes
        )  # Adjust for the number of classes
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )  # Use self.device here
        return model

    def preprocess(self, image_path):
        """
        Preprocess the input image: Resize, Normalize, and Convert to Tensor.
        """
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image.to(self.device)

    def run_from_filepath(self, image_path):
        """
        Perform inference on a single image from file path.
        Args:
            image_path: Path to the image
        Returns:
            Predicted class and probability
        """
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
            [self.labels[i], float(probabilities[i])] for i in range(len(probabilities))
        ]
        sorted_probs = sorted(label_to_probabilities, key=lambda x: x[1], reverse=True)

        # Predicted class
        predicted_label = sorted_probs[0][0]
        predicted_probability = sorted_probs[0][1]
        predicted_class = self.classes[predicted_label]

        return predicted_class, predicted_probability


# Define a Custom Dataset for Test Images
class CustomTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Custom dataset for loading test images.
        """
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # Return the image and its filename


# Function for Batch Inference
def predict_batch(data_loader, model, device, class_names, inv_transform=None):
    """
    Perform batch inference on a DataLoader of images and display predictions.
    Args:
        data_loader: DataLoader object containing the test images.
        model: PyTorch model to use for inference.
        device: 'cuda' or 'cpu'.
        class_names: List of class names for predictions.
        inv_transform: Optional transform to revert normalization for visualization.
    """
    model.eval()  # Ensure the model is in evaluation mode
    results = []  # To store predictions

    with torch.no_grad():
        for images, filenames in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices

            # Loop through the batch
            for i in range(images.size(0)):
                # Get the predicted class name
                predicted_class = class_names[predicted[i].item()]
                results.append((filenames[i], predicted_class))

                # Optionally visualize images
                if inv_transform:
                    # Revert normalization for visualization
                    image = inv_transform(images[i].cpu())
                    image = image.permute(1, 2, 0).numpy()

                    # Display the image with its prediction
                    plt.imshow(image)
                    plt.title(f"File: {filenames[i]}, Predicted: {predicted_class}")
                    plt.axis("off")
                    plt.show()
    return results


# Define the transforms for test data
test_transforms = transforms.Compose(
    [
        transforms.Resize(224),  # Resize the image to 224x224
        transforms.CenterCrop(224),  # Crop the center
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize (ImageNet stats)
    ]
)

# Define inverse transforms for visualization (optional)
inv_transform = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)

# Paths and class labels
model_path = r"d:\grad project\imgClass_AD\activity detection models\fine_tuned_resnet18_2.pth"
test_image_dir = r"D:\grad project\imgClass_AD\test_img"
labels = list(range(0, 9))
classes = {
    0: "Safe driving",
    1: "Texting(right hand)",
    2: "Talking on the phone (right hand)",
    3: "Texting (left hand)",
    4: "Talking on the phone (left hand)",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger(s)",
}
 


# Create test dataset and DataLoader
test_dataset = CustomTestDataset(test_image_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PyTorchClassificationModel(
    model_path, labels, test_transforms, classes, device=device
).model

# # Perform batch inference
# results = predict_batch(test_loader, model, device, classes, inv_transform=inv_transform)
# print("Batch Results:")
# for file, prediction in results:
#     print(f"File: {file}, Predicted: {prediction}")

# Select a specific batch number
batch_number = 0 # For example, pick the 3rd batch (index starts from 0)

# Iterate through the DataLoader
for batch_idx, (images, filenames) in enumerate(test_loader):
    if batch_idx == batch_number:
        images = images.to(device)  # Move images to the appropriate device (GPU or CPU)
        outputs = model(images)  # Perform inference
        _, predicted = torch.max(outputs, 1)  # Get predicted class indices

        # Display predictions for the selected batch
        for i in range(images.size(0)):
            # Get the predicted class name
            predicted_class = classes[predicted[i].item()]

            # Revert normalization for visualization
            image = inv_transform(images[i].cpu())  # Undo normalization
            image = image.permute(
                1, 2, 0
            ).numpy()  # Convert from (C, H, W) to (H, W, C)

            # Display the image with prediction
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.title(f"File: {filenames[i]}\nPredicted Class: {predicted_class}")
            plt.axis("off")
            plt.show()
        break  # Exit the loop after processing the desired batch

