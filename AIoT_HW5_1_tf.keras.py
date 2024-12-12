# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:19:41 2024

@author: User
"""


import tensorflow as tf
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

# Step 4: Modeling
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_dim=X.shape[1], activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Evaluation
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=2)

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"測試集損失: {loss}")
print(f"測試集準確率: {accuracy}")

# Step 6: Deployment
# 儲存模型
model.save('iris_classification_model.h5')
print("模型已保存為 iris_classification_model.h5")

# 測試預測
sample = X_test[:5]
predictions = model.predict(sample)
for i, pred in enumerate(predictions):
    predicted_class = iris.target_names[np.argmax(pred)]
    print(f"樣本 {i+1} 預測為: {predicted_class}")
