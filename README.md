# Machine Learning Homework

This repository contains solutions to three machine learning tasks using different frameworks such as `tf.keras`, `PyTorch`, and `PyTorch Lightning`. The tasks involve solving classic machine learning problems, including Iris classification, handwritten digit recognition, and image classification using a pretrained VGG19 model.

---

## Table of Contents

- [HW5-1: Iris Classification Problem](#hw5-1-iris-classification-problem-tfkeras-pytorch-pytorch-lightning)
- [HW5-2: Handwritten Digit Recognition (Dense NN, CNN)](#hw5-2-handwritten-digit-recognition-dense-nn-cnn-tfkeras-pytorch-or-pytorch-lightning)
- [HW5-3: CIFAR Image Classification with VGG19 Pretrained Model](#hw5-3-cifar-image-classification-with-vgg19-pretrained-model-tfkeras-or-pytorch-lightning)
---

## HW5-1: Iris Classification Problem (tf.keras, PyTorch, PyTorch Lightning)

### Problem Description:
This task involves implementing an Iris classification problem using three different frameworks: `tf.keras`, `PyTorch`, and `PyTorch Lightning`. The goal is to build a neural network for classifying the Iris dataset using dropout, batch normalization, and other advanced training techniques such as EarlyStopping and Learning Rate Scheduler.

### Approach:
We will first implement the Iris classification problem using `tf.keras`, followed by a PyTorch implementation and finally a PyTorch Lightning version. We will also add necessary callback functions like EarlyStopping and LearningRateScheduler for better model performance during training.

### Prompt Summary:
- **tf.keras Implementation**:
  - Train a model for Iris classification using `tf.keras` with dropout, batch normalization, and other necessary callbacks such as `EarlyStopping` and `LearningRateScheduler`.
  
- **PyTorch Implementation**:
  - Implement the Iris classification model using PyTorch, applying dropout and batch normalization, with the use of an early stopping callback and learning rate scheduler.

- **PyTorch Lightning Implementation**:
  - Convert the model to PyTorch Lightning, making use of its built-in hooks for early stopping and learning rate scheduling.

---

## HW5-2: Handwritten Digit Recognition (Dense NN, CNN) - tf.keras, PyTorch or PyTorch Lightning

### Problem Description:
This task involves handwritten digit recognition using both Dense Neural Networks (DNN) and Convolutional Neural Networks (CNN). We will implement this problem using three frameworks: `tf.keras`, `PyTorch`, and `PyTorch Lightning`.

### Approach:
We will first implement a Dense Neural Network (DNN) followed by a Convolutional Neural Network (CNN) for handwritten digit classification (using the MNIST dataset). Both models will be implemented using `tf.keras`, `PyTorch`, and `PyTorch Lightning`.

- **Dense NN**: A simple fully connected neural network that works well for digit classification.
- **CNN**: A Convolutional Neural Network, typically more effective for image data like MNIST.

### Prompt Summary:
- **tf.keras Implementation**:
  - Create DNN and CNN models using `tf.keras` for handwritten digit classification.
  - Use necessary layers like `Dense`, `Conv2D`, and activation functions to build the networks.

- **PyTorch Implementation**:
  - Implement the models using `torch.nn.Module` for both Dense NN and CNN architectures.
  - Use necessary layers like `Linear`, `Conv2d`, and activation functions like `ReLU`.

- **PyTorch Lightning Implementation**:
  - Implement the models using PyTorch Lightning.
  - Use built-in support for easy scaling and training loops, including validation and testing.

---

## HW5-3: CIFAR Image Classification with VGG19 Pretrained Model (tf.keras or PyTorch Lightning)

### Problem Description:
This task involves CIFAR image classification using a VGG19 pretrained model. We will use either `tf.keras` or `PyTorch Lightning` to implement this model and fine-tune it for the CIFAR-10 dataset.

### Approach:
We will leverage the VGG19 model, pretrained on ImageNet, and modify the final layers to adapt to the 10 output classes of the CIFAR-10 dataset. This will involve:

- Freezing the earlier layers of the network (since they are already pretrained on ImageNet) and fine-tuning only the final layers.
- Using a dropout layer to regularize the model.
- Adding a learning rate scheduler to improve training.

### Prompt Summary:
- **tf.keras Implementation**:
  - Load the pretrained VGG19 model from Keras applications.
  - Fine-tune the model by adjusting the last layer for CIFAR-10 classification.
  - Use callbacks like EarlyStopping and LearningRateScheduler.
  
- **PyTorch Lightning Implementation**:
  - Load the pretrained VGG19 model from PyTorch's `torchvision.models`.
  - Modify the final layer to match CIFAR-10 output classes.
  - Use PyTorch Lightning's hooks for training, validation, and callbacks such as EarlyStopping.

---



















