# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:02:27 2024

@author: User
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets, models

# Step 1: Data Loading and Preprocessing

# Transform: ToTensor + Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG19 requires 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained model stats
])

# Download CIFAR-10 dataset and apply transformations
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader for training and testing
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Step 2: Define the VGG19 Model

# Load VGG19 pretrained on ImageNet
model = models.vgg19(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to match CIFAR-10's 10 classes
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 3: Define Loss Function and Optimizer

# Cross-entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer for fine-tuning the last fully connected layer
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Step 4: Training the Model

epochs = 10
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    correct = 0
    total = 0
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
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

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f'Test accuracy: {test_accuracy:.4f}')

# Step 6: Save the Model

torch.save(model.state_dict(), 'vgg19_cifar10.pth')
print("Model saved as vgg19_cifar10.pth")

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
sample_images = sample_images.to(device)
sample_outputs = model(sample_images)

# Display predictions
for i in range(5):
    predicted_class = torch.argmax(sample_outputs[i]).item()
    print(f"Sample {i+1} predicted as class: {predicted_class}")
