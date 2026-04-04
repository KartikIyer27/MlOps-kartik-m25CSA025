"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD BASE MODEL (ResNet18)
# -----------------------
base_model = models.resnet18(pretrained=False)
base_model.fc = nn.Linear(base_model.fc.in_features, 10)
base_model.load_state_dict(torch.load("outputs/resnet18.pth"))
base_model.to(DEVICE)
base_model.eval()

classifier = PyTorchClassifier(
    model=base_model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(base_model.parameters()),
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(-1, 1)
)

# -----------------------
# DATA
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# -----------------------
# STRONG ATTACKS
# -----------------------
pgd = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.2,
    eps_step=0.02,
    max_iter=10
)

bim = BasicIterativeMethod(
    estimator=classifier,
    eps=0.2
)

# -----------------------
# GENERATE TRAINING DATA (PGD)
# -----------------------
clean_images = []
adv_images = []
labels = []

print("\n--- Generating Training Data ---")

for i, (images, _) in enumerate(loader):
    if i == 30:   # 🔥 increased data (~4000 samples)
        break

    images_np = images.numpy()

    adv = pgd.generate(x=images_np)

    clean_images.append(images)
    adv_images.append(torch.tensor(adv))

    labels += [0]*len(images) + [1]*len(images)

clean_images = torch.cat(clean_images)
adv_images = torch.cat(adv_images)

print("Clean:", len(clean_images), "Adv:", len(adv_images))

X = torch.cat([clean_images, adv_images])
y = torch.tensor(labels)

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -----------------------
# DETECTOR MODEL (ResNet34)
# -----------------------
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# TRAIN DETECTOR
# -----------------------
print("\n--- Training Detector ---")

for epoch in range(10):   # 🔥 increased epochs
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}, Accuracy: {correct/total:.4f}")

# -----------------------
# EVALUATION FUNCTION
# -----------------------
def evaluate_detector(model, attack, name):
    model.eval()
    correct = 0
    total = 0

    for i, (images, _) in enumerate(loader):
        if i == 30:  # 🔥 same data size
            break

        images_np = images.numpy()

        adv = attack.generate(x=images_np)

        clean_tensor = images.to(DEVICE)
        adv_tensor = torch.tensor(adv).to(DEVICE)

        X = torch.cat([clean_tensor, adv_tensor])
        y = torch.tensor([0]*len(images) + [1]*len(images)).to(DEVICE)

        outputs = model(X)
        preds = outputs.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"{name} Detection Accuracy: {acc:.4f}")
    return acc

# -----------------------
# FINAL EVALUATION
# -----------------------
print("\n--- Final Evaluation ---")

pgd_acc = evaluate_detector(model, pgd, "PGD")
bim_acc = evaluate_detector(model, bim, "BIM")

# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "outputs/detector_resnet34.pth")

print("\nDetector training complete and saved!")
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset, random_split
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD BASE MODEL (ResNet18)
# -----------------------
base_model = models.resnet18(pretrained=False)
base_model.fc = nn.Linear(base_model.fc.in_features, 10)
base_model.load_state_dict(torch.load("outputs/resnet18.pth", map_location=DEVICE))
base_model.to(DEVICE)
base_model.eval()

classifier = PyTorchClassifier(
    model=base_model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(base_model.parameters()),
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(-1, 1),
)

# -----------------------
# DATA
# -----------------------
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_base = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader_base = DataLoader(train_base, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader_base = DataLoader(test_base, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------
# STRONG ATTACKS
# -----------------------
pgd = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.12,
    eps_step=0.01,
    max_iter=20,
    targeted=False,
)

bim = BasicIterativeMethod(
    estimator=classifier,
    eps=0.12,
    eps_step=0.01,
    max_iter=20,
    targeted=False,
)


def extract_features(batch_images):
    with torch.no_grad():
        logits = base_model(batch_images)
        probs = torch.softmax(logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        max_prob = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1)
        logit_norm = logits.norm(dim=1)
        return torch.stack([max_prob, margin, entropy, logit_norm], dim=1)


def build_detector_dataset(source_loader, max_batches):
    feature_rows = []
    labels = []

    print("\n--- Generating Detector Dataset ---")

    for i, (images, true_labels) in enumerate(source_loader):
        if i == max_batches:
            break

        images_np = images.numpy()
        labels_np = true_labels.numpy()

        adv_pgd = pgd.generate(x=images_np, y=labels_np)
        adv_bim = bim.generate(x=images_np, y=labels_np)

        clean_tensor = images.to(DEVICE)
        pgd_tensor = torch.tensor(adv_pgd, dtype=images.dtype).to(DEVICE)
        bim_tensor = torch.tensor(adv_bim, dtype=images.dtype).to(DEVICE)

        feature_rows.append(extract_features(clean_tensor).cpu())
        feature_rows.append(extract_features(pgd_tensor).cpu())
        feature_rows.append(extract_features(bim_tensor).cpu())

        labels.extend([0] * len(images))
        labels.extend([1] * len(images))
        labels.extend([1] * len(images))

    X = torch.cat(feature_rows)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X, y)


detector_dataset = build_detector_dataset(train_loader_base, max_batches=150)
train_size = int(0.85 * len(detector_dataset))
val_size = len(detector_dataset) - train_size
train_dataset, val_dataset = random_split(
    detector_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


class FeatureDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.network(x)


def run_epoch(dataloader, model, criterion, optimizer, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    for features, labels in dataloader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            outputs = model(features)
            loss = criterion(outputs, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total


def evaluate_detector(eval_model, attack, name, source_loader, max_batches=40):
    eval_model.eval()
    correct = 0
    total = 0

    for i, (images, true_labels) in enumerate(source_loader):
        if i == max_batches:
            break

        images_np = images.numpy()
        labels_np = true_labels.numpy()

        adv = attack.generate(x=images_np, y=labels_np)

        clean_tensor = images.to(DEVICE)
        adv_tensor = torch.tensor(adv, dtype=images.dtype).to(DEVICE)

        clean_features = extract_features(clean_tensor)
        adv_features = extract_features(adv_tensor)

        X = torch.cat([clean_features, adv_features], dim=0).to(DEVICE)
        y = torch.tensor([0] * len(images) + [1] * len(images), dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            outputs = eval_model(X)
            preds = outputs.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"{name} Detection Accuracy: {acc:.4f}")
    return acc


model = FeatureDetector(input_dim=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

print("\n--- Training Detector ---")

best_val_acc = 0.0
for epoch in range(25):
    train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer, train_mode=True)
    val_loss, val_acc = run_epoch(val_loader, model, criterion, optimizer, train_mode=False)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "outputs/detector_resnet34.pth")

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

print("\n--- Final Evaluation ---")

best_model = FeatureDetector(input_dim=4).to(DEVICE)
best_model.load_state_dict(torch.load("outputs/detector_resnet34.pth", map_location=DEVICE))

pgd_acc = evaluate_detector(best_model, pgd, "PGD", test_loader_base)
bim_acc = evaluate_detector(best_model, bim, "BIM", test_loader_base)

print("\nDetector training complete and saved!")
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader, TensorDataset
# from art.estimators.classification import PyTorchClassifier
# from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
# import numpy as np
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# # -----------------------
# # LOAD BASE MODEL (ResNet18)
# # -----------------------
# base_model = models.resnet18(pretrained=False)
# base_model.fc = nn.Linear(base_model.fc.in_features, 10)
# base_model.load_state_dict(torch.load("outputs/resnet18.pth"))
# base_model.to(DEVICE)
# base_model.eval()
#
# classifier = PyTorchClassifier(
#     model=base_model,
#     loss=nn.CrossEntropyLoss(),
#     optimizer=torch.optim.Adam(base_model.parameters()),
#     input_shape=(3, 32, 32),
#     nb_classes=10,
#     clip_values=(-1, 1)
# )
#
# # -----------------------
# # DATA
# # -----------------------
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
# loader = DataLoader(dataset, batch_size=128, shuffle=True)
#
# # -----------------------
# # STRONG ATTACKS
# # -----------------------
# pgd = ProjectedGradientDescent(
#     estimator=classifier,
#     eps=0.2,
#     eps_step=0.02,
#     max_iter=10
# )
#
# bim = BasicIterativeMethod(
#     estimator=classifier,
#     eps=0.2
# )
#
# # -----------------------
# # GENERATE TRAINING DATA (PGD)
# # -----------------------
# clean_images = []
# adv_images = []
# labels = []
#
# print("\n--- Generating Training Data ---")
#
# for i, (images, _) in enumerate(loader):
#     if i == 30:   # 🔥 increased data (~4000 samples)
#         break
#
#     images_np = images.numpy()
#
#     adv = pgd.generate(x=images_np)
#
#     clean_images.append(images)
#     adv_images.append(torch.tensor(adv))
#
#     labels += [0]*len(images) + [1]*len(images)
#
# clean_images = torch.cat(clean_images)
# adv_images = torch.cat(adv_images)
#
# print("Clean:", len(clean_images), "Adv:", len(adv_images))
#
# X = torch.cat([clean_images, adv_images])
# y = torch.tensor(labels)
#
# train_dataset = TensorDataset(X, y)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#
# # -----------------------
# # DETECTOR MODEL (ResNet34)
# # -----------------------
# model = models.resnet34(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 2)
# model.to(DEVICE)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # -----------------------
# # TRAIN DETECTOR
# # -----------------------
# print("\n--- Training Detector ---")
#
# for epoch in range(10):   # 🔥 increased epochs
#     model.train()
#     correct = 0
#     total = 0
#
#     for images, labels in train_loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         preds = outputs.argmax(1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#
#     print(f"Epoch {epoch+1}, Accuracy: {correct/total:.4f}")
#
# # -----------------------
# # EVALUATION FUNCTION
# # -----------------------
# def evaluate_detector(model, attack, name):
#     model.eval()
#     correct = 0
#     total = 0
#
#     for i, (images, _) in enumerate(loader):
#         if i == 30:  # 🔥 same data size
#             break
#
#         images_np = images.numpy()
#
#         adv = attack.generate(x=images_np)
#
#         clean_tensor = images.to(DEVICE)
#         adv_tensor = torch.tensor(adv).to(DEVICE)
#
#         X = torch.cat([clean_tensor, adv_tensor])
#         y = torch.tensor([0]*len(images) + [1]*len(images)).to(DEVICE)
#
#         outputs = model(X)
#         preds = outputs.argmax(1)
#
#         correct += (preds == y).sum().item()
#         total += y.size(0)
#
#     acc = correct / total
#     print(f"{name} Detection Accuracy: {acc:.4f}")
#     return acc
#
# # -----------------------
# # FINAL EVALUATION
# # -----------------------
# print("\n--- Final Evaluation ---")
#
# pgd_acc = evaluate_detector(model, pgd, "PGD")
# bim_acc = evaluate_detector(model, bim, "BIM")
#
# # -----------------------
# # SAVE MODEL
# # -----------------------
# torch.save(model.state_dict(), "outputs/detector_resnet34.pth")
#
# print("\nDetector training complete and saved!")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset, random_split
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
import matplotlib.pyplot as plt
import wandb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "").strip()
try:
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY, relogin=True)
    else:
        wandb.login(relogin=False)
except Exception as exc:
    print(f"W&B login failed, continuing without W&B: {exc}")

wandb.init(project="assignment5-q2-detector", name="detector-feature-based", reinit=True)


def denormalize(image_tensor):
    return torch.clamp(image_tensor * 0.5 + 0.5, 0, 1)

# -----------------------
# LOAD BASE MODEL (ResNet18)
# -----------------------
base_model = models.resnet18(weights=None)
base_model.fc = nn.Linear(base_model.fc.in_features, 10)
base_model.load_state_dict(torch.load("outputs/resnet18.pth", map_location=DEVICE))
base_model.to(DEVICE)
base_model.eval()

classifier = PyTorchClassifier(
    model=base_model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(base_model.parameters()),
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(-1, 1),
)

# -----------------------
# DATA
# -----------------------
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_base = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader_base = DataLoader(train_base, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader_base = DataLoader(test_base, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------
# STRONG ATTACKS
# -----------------------
pgd = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.12,
    eps_step=0.01,
    max_iter=20,
    targeted=False,
)

bim = BasicIterativeMethod(
    estimator=classifier,
    eps=0.12,
    eps_step=0.01,
    max_iter=20,
    targeted=False,
)


def build_detector_dataset(source_loader, max_batches):
    clean_images = []
    adv_images = []
    labels = []

    print("\n--- Generating Detector Dataset ---")

    for i, (images, _) in enumerate(source_loader):
        if i == max_batches:
            break

        images_np = images.numpy()

        adv_pgd = pgd.generate(x=images_np)
        adv_bim = bim.generate(x=images_np)

        clean_images.append(images)
        adv_images.append(torch.tensor(adv_pgd, dtype=images.dtype))
        adv_images.append(torch.tensor(adv_bim, dtype=images.dtype))

        labels += [0] * len(images) + [1] * len(images) + [1] * len(images)

    clean_images = torch.cat(clean_images)
    adv_images = torch.cat(adv_images)

    X = torch.cat([clean_images, adv_images])
    y = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(X, y)


detector_dataset = build_detector_dataset(train_loader_base, max_batches=120)
train_size = int(0.9 * len(detector_dataset))
val_size = len(detector_dataset) - train_size
train_dataset, val_dataset = random_split(
    detector_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------
# DETECTOR MODEL (Feature-Based)
# -----------------------


class FeatureDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.network(x)


def extract_features(batch_images):
    with torch.no_grad():
        logits = base_model(batch_images)
        probs = torch.softmax(logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        max_prob = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1)
        logit_norm = logits.norm(dim=1)
        feature_tensor = torch.stack([max_prob, margin, entropy, logit_norm], dim=1)
    return feature_tensor


def build_detector_dataset(source_loader, max_batches):
    feature_rows = []
    labels = []

    print("\n--- Generating Detector Dataset ---")

    for i, (images, true_labels) in enumerate(source_loader):
        if i == max_batches:
            break

        images_np = images.numpy()
        labels_np = true_labels.numpy()

        adv_pgd = pgd.generate(x=images_np, y=labels_np)
        adv_bim = bim.generate(x=images_np, y=labels_np)

        clean_tensor = images.to(DEVICE)
        pgd_tensor = torch.tensor(adv_pgd, dtype=images.dtype).to(DEVICE)
        bim_tensor = torch.tensor(adv_bim, dtype=images.dtype).to(DEVICE)

        feature_rows.append(extract_features(clean_tensor).cpu())
        feature_rows.append(extract_features(pgd_tensor).cpu())
        feature_rows.append(extract_features(bim_tensor).cpu())

        labels.extend([0] * len(images))
        labels.extend([1] * len(images))
        labels.extend([1] * len(images))

    X = torch.cat(feature_rows)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X, y)


def save_wandb_samples(source_loader, max_batches=1):
    sample_images = []

    for i, (images, true_labels) in enumerate(source_loader):
        if i == max_batches:
            break

        images_np = images.numpy()
        labels_np = true_labels.numpy()
        adv_pgd = pgd.generate(x=images_np, y=labels_np)
        adv_bim = bim.generate(x=images_np, y=labels_np)

        for sample_idx in range(min(10, len(images))):
            clean_image = denormalize(images[sample_idx]).permute(1, 2, 0).numpy()
            pgd_image = denormalize(torch.tensor(adv_pgd[sample_idx])).permute(1, 2, 0).numpy()
            bim_image = denormalize(torch.tensor(adv_bim[sample_idx])).permute(1, 2, 0).numpy()

            figure = plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(clean_image)
            plt.title("Clean")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(pgd_image)
            plt.title("PGD")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(bim_image)
            plt.title("BIM")
            plt.axis("off")

            sample_path = f"outputs/detector_sample_{i}_{sample_idx}.png"
            figure.savefig(sample_path)
            plt.close(figure)
            sample_images.append(wandb.Image(sample_path, caption=f"sample_{i}_{sample_idx}"))

    wandb.log({"detector_samples": sample_images})


save_wandb_samples(test_loader_base, max_batches=1)


detector_dataset = build_detector_dataset(train_loader_base, max_batches=150)
train_size = int(0.85 * len(detector_dataset))
val_size = len(detector_dataset) - train_size
train_dataset, val_dataset = random_split(
    detector_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

model = FeatureDetector(input_dim=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)


def run_epoch(dataloader, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    for features, labels in dataloader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            outputs = model(features)
            loss = criterion(outputs, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total


def evaluate_detector(eval_model, attack, name, source_loader, max_batches=40):
    eval_model.eval()
    correct = 0
    total = 0

    for i, (images, true_labels) in enumerate(source_loader):
        if i == max_batches:
            break

        images_np = images.numpy()
        labels_np = true_labels.numpy()

        adv = attack.generate(x=images_np, y=labels_np)

        clean_tensor = images.to(DEVICE)
        adv_tensor = torch.tensor(adv, dtype=images.dtype).to(DEVICE)

        clean_features = extract_features(clean_tensor)
        adv_features = extract_features(adv_tensor)

        X = torch.cat([clean_features, adv_features], dim=0).to(DEVICE)
        y = torch.tensor([0] * len(images) + [1] * len(images), dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            outputs = eval_model(X)
            preds = outputs.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"{name} Detection Accuracy: {acc:.4f}")
    return acc


# -----------------------
# TRAIN DETECTOR
# -----------------------
print("\n--- Training Detector ---")

best_val_acc = 0.0
for epoch in range(25):
    train_loss, train_acc = run_epoch(train_loader, train_mode=True)
    val_loss, val_acc = run_epoch(val_loader, train_mode=False)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "outputs/detector_resnet34.pth")

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )


# -----------------------
# FINAL EVALUATION
# -----------------------
print("\n--- Final Evaluation ---")

best_model = FeatureDetector(input_dim=4).to(DEVICE)
best_model.load_state_dict(torch.load("outputs/detector_resnet34.pth", map_location=DEVICE))

pgd_acc = evaluate_detector(best_model, pgd, "PGD", test_loader_base)
bim_acc = evaluate_detector(best_model, bim, "BIM", test_loader_base)

wandb.log({
    "pgd_detection_accuracy": pgd_acc,
    "bim_detection_accuracy": bim_acc,
    "best_val_accuracy": best_val_acc,
})

print("\nDetector training complete and saved!")

wandb.finish()