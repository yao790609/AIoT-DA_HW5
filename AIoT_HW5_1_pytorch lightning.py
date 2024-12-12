# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:28:26 2024

@author: User
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
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

# Step 4: Modeling with PyTorch Lightning
class IrisModel(pl.LightningModule):
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

    def configure_optimizers(self):
        # 使用 Adam 優化器
        return optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        X, y = batch
        output = self(X)
        loss = nn.BCELoss()(output, y)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        output = self(X)
        loss = nn.BCELoss()(output, y)
        return loss

    def test_step(self, batch, batch_idx):
        # 新增 test_step 方法來支持測試
        X, y = batch
        output = self(X)
        loss = nn.BCELoss()(output, y)
        return loss

    def train_dataloader(self):
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        return torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    def val_dataloader(self):
        val_data = torch.utils.data.TensorDataset(X_test, y_test)
        return torch.utils.data.DataLoader(val_data, batch_size=16)

    def test_dataloader(self):
        test_data = torch.utils.data.TensorDataset(X_test, y_test)  # 測試資料也可以使用 X_test 和 y_test
        return torch.utils.data.DataLoader(test_data, batch_size=16)

# Step 5: Setup PyTorch Lightning Trainer
model = IrisModel()  # 確保這裡是實例化的模型

# Step 6: Setup the PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=50)

# Step 7: Training
trainer.fit(model)  # 訓練過程

# Step 8: Evaluation
trainer.test(model)  # 測試過程

# Step 9: Deployment
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
