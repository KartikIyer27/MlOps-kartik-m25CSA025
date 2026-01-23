# MlOps-kartik-m25CSA025
# DL-Ops Lab Assignment 1

## Overview
This repository contains experiments performed as part of **DL-Ops Lab Assignment 1**.  
The goal is to study the impact of **batch size, optimizer, learning rate, model architecture, and compute platform (CPU vs GPU)** on image classification performance.

Experiments are conducted on:
- **MNIST**
- **FashionMNIST**

using:
- **Deep Learning models** (ResNet-18, ResNet-50)
- **Classical ML models** (SVM with Polynomial and RBF kernels)

---


## Deep Learning Experiments

### Hyperparameters Explored
- Batch sizes: `16`, `32`
- Optimizers: `SGD`, `Adam`
- Learning rates: `0.001`, `0.0001`
- models : `resnet18` , `resnet50`

---

## Results: MNIST

| Batch Size | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%) |
|-----------|----------|---------------|--------------|---------------|
| 16 | SGD | 0.001 | 98.97 | 98.86 |
| 16 | SGD | 0.0001 | 96.37 | 94.77 |
| 16 | Adam | 0.001 | 98.87 | 98.71 |
| 16 | Adam | 0.0001 | **99.17** | 98.36 |
| 32 | SGD | 0.001 | 98.38 | 98.36 |
| 32 | Adam | 0.0001 | 99.08 | 98.81 |

---

## Results: FashionMNIST

| Batch Size | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%) |
|-----------|----------|---------------|--------------|---------------|
| 16 | SGD | 0.001 | 90.42 | 89.64 |
| 16 | SGD | 0.0001 | 83.36 | 76.22 (3 ep) / 86.39 (8 ep) |
| 16 | Adam | 0.001 | 90.59 | 89.22 |
| 16 | Adam | 0.0001 | **91.62** | **90.20** |
| 32 | SGD | 0.001 | 89.27 | 87.89 |
| 32 | Adam | 0.0001 | 91.04 | 89.41 |

---

## Training & Validation Curves

The following graphs correspond to the **best configuration**:
- Batch size = `16`
- Optimizer = `Adam`
- Learning rate = `0.0001`
- Model = `ResNet-18`
- Dataset = `MNIST`

### Training vs Validation Accuracy
![Training vs Validation Accuracy](accuracy.png)

### Training vs Validation Loss
![Training vs Validation Loss](loss.png)



---

## SVM Classification Results

| Dataset | Kernel | Accuracy (%) | Training Time (ms) |
|-------|--------|--------------|--------------------|
| MNIST | Polynomial | 91.70 | 4578 |
| MNIST | RBF | **93.15** | 4172 |
| FashionMNIST | Polynomial | 81.55 | 3928 |
| FashionMNIST | RBF | **85.15** | 3791 |

---

## CPU vs GPU Performance

| Compute | Model | Accuracy (%) | Train Time (ms) | FLOPs |
|-------|------|--------------|-----------------|-------|
| CPU | ResNet-18 | 85.95 | 54075 | 1.824 |
| CPU | ResNet-50 | 80.62 | 130139 | 4.132 |
| GPU | ResNet-18 | **90.59** | **44224** | 1.824 |
| GPU | ResNet-50 | **89.64** | **76061** | 4.132 |

---


## Links
- 📓 Colab Notebook: [Open in Colab](https://colab.research.google.com/drive/14PQwbYVtsiNgUgT1_LfDSVID2RHZJTuJ?usp=sharing)
  
