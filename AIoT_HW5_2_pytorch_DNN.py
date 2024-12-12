# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:38:41 2024

@author: User
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 加載數據
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# 3. 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平圖片
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. 初始化模型並移動到 GPU
model = Net().to(device)

# 5. 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 6. 訓練模型
for epoch in range(2):  # 訓練兩個時期
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 將數據移動到 GPU

        # 清零梯度
        optimizer.zero_grad()

        # 前向 + 反向 + 優化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:  # 每 1000 個批次輸出一次
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}")
            running_loss = 0.0

print("Finished Training")

# 7. 測試模型
correct = 0
total = 0
with torch.no_grad():  # 在測試階段不計算梯度
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # 將數據移動到 GPU
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

# 8. 顯示一張圖片
dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)  # 確保將圖片移動到 GPU
plt.imshow(images[0].cpu().numpy().squeeze(), cmap='gray')  # 回到 CPU 再顯示
plt.title(f"Label: {labels[0].item()}")
plt.show()
