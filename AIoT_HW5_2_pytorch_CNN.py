# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:54:59 2024

@author: User
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# Step 1: Data Loading and Preprocessing
# Download MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for training and test datasets
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Step 2: Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First Conv Layer + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Second Conv Layer + Pooling
        x = torch.relu(self.conv3(x))             # Third Conv Layer
        x = x.view(-1, 64 * 7 * 7)                # Flatten the tensor
        x = torch.relu(self.fc1(x))               # Fully connected layer
        x = self.fc2(x)                           # Output layer (softmax handled in loss function)
        return x

# Step 3: Initialize the Model, Loss Function, and Optimizer
model = CNN()

# Loss function (CrossEntropyLoss automatically applies softmax)
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the Model
epochs = 5
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Step 5: Evaluate the Model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f'Test accuracy: {test_accuracy:.4f}')

# Step 6: Save the Model
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("Model saved as mnist_cnn_model.pth")

# Step 7: Visualize Training History
# Plot the training accuracy
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training Accuracy Over Epochs')
plt.show()

# Step 8: Predictions on Test Data
sample_images, _ = next(iter(test_loader))
sample_predictions = model(sample_images)

# Display predictions
for i in range(5):
    predicted_class = torch.argmax(sample_predictions[i]).item()
    print(f"Sample {i+1} predicted as: {predicted_class}")
