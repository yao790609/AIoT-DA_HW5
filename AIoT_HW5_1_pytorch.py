# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:26:50 2024

@author: User
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import pandas as pd
import numpy as np

# Step 1: Business Understanding
# 我們的目標是分類 Iris 花的種類 (Setosa, Versicolour, Virginica)

# Step 2: Data Understanding
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

print("資料前五列:")
print(data.head())
print("\n資料摘要:")
print(data.describe())

# Step 3: Data Preparation
# Features (X) and Target (y)
X = iris.data
y = iris.target

# 資料標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot 編碼目標變數
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 轉換成 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 4: Modeling
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(X.shape[1], 16)  # 第一層
        self.layer2 = nn.Linear(16, 8)  # 第二層
        self.output = nn.Linear(8, 3)  # 輸出層 (3 類)

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # 激活函數
        x = torch.relu(self.layer2(x))  # 激活函數
        x = torch.softmax(self.output(x), dim=1)  # 輸出層
        return x

# 初始化模型
model = IrisModel()

# 定義損失函數和優化器
criterion = nn.BCELoss()  # 使用 BCELoss 處理 one-hot 編碼的多分類問題
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training
epochs = 50
batch_size = 16

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 訓練
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Step 6: Evaluation
model.eval()

# 評估模型
with torch.no_grad():
    output = model(X_test)
    _, predicted = torch.max(output, 1)
    y_test_max = torch.max(y_test, 1)[1]  # 找出 one-hot 編碼的最大值，對應實際標籤

    correct = (predicted == y_test_max).sum().item()
    accuracy = correct / len(y_test)

print(f"測試集準確率: {accuracy}")

# Step 7: Deployment
# 儲存模型
torch.save(model.state_dict(), 'iris_classification_model.pth')
print("模型已保存為 iris_classification_model.pth")

# 測試預測
sample = X_test[:5]
with torch.no_grad():
    predictions = model(sample)
    _, predicted_classes = torch.max(predictions, 1)

    for i, pred_class in enumerate(predicted_classes):
        predicted_class_name = iris.target_names[pred_class.item()]
        print(f"樣本 {i+1} 預測為: {predicted_class_name}")
