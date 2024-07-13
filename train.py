from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import numpy as np
from CIFAKEClassifier import CIFAKEClassifier
from device import fetchDevice
import dataloader1 
from dataloader1 import CIFAKEDataset
import torch
import torch.nn as nn
import torch.nn.functional as F


data_set = CIFAKEDataset(num_processes=4)
model = CIFAKEClassifier()
model.to(fetchDevice())


# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 5

# Create the training and testing splits
train_size = int(0.95 * len(data_set))
test_size = len(data_set) - train_size
train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

# Dataloader for batch training
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = Adam(model.parameters(), lr=learning_rate)

# Train the model
def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            labels = labels.float()  # BCELoss expects labels to be in float format

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()  # Remove unnecessary dimensions
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Finished Training')

def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images).squeeze()  # Remove unnecessary dimensions
            predicted = torch.round(outputs)  # Round to get binary predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    


train_model(model, train_loader, criterion, optimizer, epochs)
test_model(model, test_loader)


torch.save(model.state_dict(), 'model_weights.pth')
