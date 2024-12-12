# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:38:19 2024

@author: User
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Business Understanding
# 目標是訓練一個模型來識別手寫數字

# Step 2: Data Understanding
# 載入 MNIST 資料集，包含 60,000 個訓練圖片和 10,000 個測試圖片
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 資料維度
print("訓練集維度: ", x_train.shape)
print("測試集維度: ", x_test.shape)

# 顯示一張圖片
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Step 3: Data Preparation
# 資料預處理：將圖片數據標準化並將目標變數 one-hot 編碼

# Rescale 影像資料，使其從 [0, 255] 範圍變為 [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten 28x28 圖像為一維向量，這是對 DNN 模型所需的處理
x_train_flatten = x_train.reshape(-1, 28*28)
x_test_flatten = x_test.reshape(-1, 28*28)

# One-hot 編碼目標變數
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

# Step 4: Modeling

# 建立 DNN 模型
dnn_model = models.Sequential([
    layers.Dense(128, activation='relu', input_dim=28*28),  # 第一層: 128 個神經元，ReLU 激活
    layers.Dense(64, activation='relu'),  # 第二層: 64 個神經元，ReLU 激活
    layers.Dense(10, activation='softmax')  # 輸出層: 10 個神經元（對應 10 類數字），softmax 激活
])

# 編譯模型
dnn_model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',  # 多類別交叉熵損失函數
                  metrics=['accuracy'])  # 評估指標為準確率

# Step 5: Training the DNN model
dnn_history = dnn_model.fit(x_train_flatten, y_train_one_hot, epochs=10, batch_size=128, validation_data=(x_test_flatten, y_test_one_hot))

# Step 6: 評估 DNN 模型
dnn_loss, dnn_accuracy = dnn_model.evaluate(x_test_flatten, y_test_one_hot, verbose=0)
print(f"DNN 測試集損失: {dnn_loss}")
print(f"DNN 測試集準確率: {dnn_accuracy}")

# 測試預測 (使用 DNN 模型)
sample = x_test_flatten[:5]  # 測試集中的前 5 張圖片
sample_labels = y_test[:5]  # 這些圖片的實際標籤
sample_pred = dnn_model.predict(sample)

# 顯示預測結果
for i in range(5):
    predicted_class = np.argmax(sample_pred[i])  # 取得預測的類別
    actual_class = sample_labels[i]  # 取得實際標籤
    print(f"樣本 {i+1} 預測為: {predicted_class}, 實際為: {actual_class}")

# 儲存模型
dnn_model.save('dnn_mnist_model.h5')
print("DNN 模型已儲存為 'dnn_mnist_model.h5'")
