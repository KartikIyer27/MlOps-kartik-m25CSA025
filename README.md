# 📘 Assignment 5 - Deep Learning

This project covers:

-   **Q1:** Vision Transformer fine-tuning using LoRA\
-   **Q2:** Adversarial attacks (FGSM, PGD, BIM) and detection model

------------------------------------------------------------------------

## ⚙️ Installation & Setup

### 🔹 Clone Repository

``` bash
git clone <your-repo-link>
cd Assignment5
```

### 🔹 Install Dependencies

``` bash
pip install -r requirements.txt
```

### 🔹 (Optional) Docker Setup

``` bash
docker build -t assignment5 .
docker run -it --rm -v $(pwd):/app assignment5
```

------------------------------------------------------------------------

## 📁 Project Structure

    Assignment5/
    │
    ├── src/
    │   ├── q1_baseline.py
    │   ├── q1_lora.py
    │   ├── q1_optuna.py
    │   ├── q2_resnet18.py
    │   ├── q2_fgsm_scratch.py
    │   ├── q2_fgsm_art.py
    │   ├── q2_attacks.py
    │   ├── q2_detector.py
    │
    ├── outputs/
    │   ├── best_lora_model.pth
    │   ├── resnet18.pth
    │   ├── detector_resnet34.pth
    │
    ├── report.pdf
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# 🧠 Q1: Vision Transformer + LoRA

## 🎯 Objective

Fine-tune **DeiT-small** on CIFAR-100 using:

-   Full fine-tuning\
-   LoRA (Low-Rank Adaptation)\
-   Optuna hyperparameter tuning

------------------------------------------------------------------------

## 🚀 How to Run Q1

### 1. Baseline Model

``` bash
python src/q1_baseline.py
```

### 2. LoRA Experiments

``` bash
python src/q1_lora.py
```

### 3. Optuna Hyperparameter Search

``` bash
python src/q1_optuna.py
```

------------------------------------------------------------------------

## 📊 Q1 Results

### Baseline

-   **Accuracy:** 68.97%\
-   **Parameters:** \~22M

### 🏆 Best Configuration

-   **Rank:** 8\
-   **Alpha:** 8\
-   **Accuracy:** 57.61%

------------------------------------------------------------------------

# ⚔️ Q2: Adversarial Attacks

## 🎯 Objective

-   Evaluate robustness of ResNet18\
-   Generate adversarial examples\
-   Train a detection model

------------------------------------------------------------------------

## 🚀 How to Run Q2

``` bash
python src/q2_resnet18.py
python src/q2_fgsm_scratch.py
python src/q2_fgsm_art.py
python src/q2_attacks.py
python src/q2_detector.py
```

------------------------------------------------------------------------

## 📊 Q2 Results

-   **ResNet18 Accuracy:** 75.93%

------------------------------------------------------------------------

## 📈 WandB

🔗 `<ADD YOUR LINK HERE>`

------------------------------------------------------------------------

## 🤗 HuggingFace

🔗 `<ADD YOUR LINK HERE>`

------------------------------------------------------------------------

## 👤 Author

-   **Name:** `<Your Name>`{=html}\
-   **Course:** `<Your Course>`{=html}
