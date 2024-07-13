import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAKEClassifier(nn.Module):
    def __init__(self):
        super(CIFAKEClassifier, self).__init__()
        # Assuming the input image size is 32x32x3 as per the rescale block in the diagram
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Convolutional layer with 32 outputs
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer with a 2x2 window and stride 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Second convolutional layer with 32 outputs
        # Flatten layer will be applied in the forward pass
        self.fc1 = nn.Linear(32 * 8 * 8, 64)  # Dense layer with 64 units
        self.fc2 = nn.Linear(64, 1)  # Final dense layer with 1 unit for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU activation function after first convolution
        x = self.pool(x)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply ReLU activation function after second convolution
        x = self.pool(x)  # Apply max pooling
        x = torch.flatten(x, 1)  # Flatten the tensor for the dense layer
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after first dense layer
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid activation function for binary classification
        return x