# MlOps-kartik-m25CSA025
# MLOps Lab – CNN on CIFAR-10

The objective of this lab was to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and track different training metrics using basic MLOps practices.

---

## Contents
- PyTorch code for training CNN on CIFAR-10
- Custom DataLoader implementation
- FLOPs calculation
- Gradient and weight flow tracking
- WandB experiment logs
  
---

## Dataset
- **CIFAR-10**
- 60,000 color images (32x32)
- 10 classes

The dataset is downloaded automatically using torchvision.

---

## Model
A simple CNN architecture was used:
- Two convolution layers
- ReLU activation
- Max pooling
- Fully connected layers for classification

The model was intentionally kept simple for learning purposes.

---

## Training Details
- Optimizer: Adam  
- Loss Function: Cross Entropy Loss  
- Epochs: 25  
- Batch Size: 128  

Training was performed on GPU when available.

---

## Experiment Tracking
Weights & Biases (WandB) was used to track:
- Training loss
- Accuracy
- Gradient flow
- Weight updates
- FLOPs

---

## Links
- **GitHub Repository**:  
  https://github.com/KartikIyer27/MlOps-kartik-m25CSA025/tree/lab2_worksheet

- **WandB Project**:  
  https://wandb.ai/m25csa025-iit-jodhpur/mlops-lab-cifar10/workspace?nw=nwuserm25csa025

- **Colab Notebook**:  
  https://colab.research.google.com/drive/1WhlUvZvw5s8h_fVTIDNHKpi2pGDMjuPF#scrollTo=VNvXd0ZcYJld

---

Name: **S Kartik Iyer**  
Roll No: **M25CSA025**
