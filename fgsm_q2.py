import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs/fgsm_images", exist_ok=True)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("outputs/resnet18.pth"))
model.to(DEVICE)
model.eval()

wandb.init(project="q2-adversarial-visualization", name="FGSM_scratch")

# Data
# transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1)

criterion = nn.CrossEntropyLoss()
epsilon = 0.1

correct_clean = 0
correct_adv = 0

for i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    images.requires_grad = True

    # CLEAN PREDICTION
    outputs = model(images)
    pred_clean = outputs.argmax(1)

    correct_clean += (pred_clean == labels).sum().item()

    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data

    # FGSM attack
    perturbed = images + epsilon * data_grad.sign()
    perturbed = torch.clamp(perturbed, -1, 1)

    outputs_adv = model(perturbed)
    pred_adv = outputs_adv.argmax(1)

    correct_adv += (pred_adv == labels).sum().item()

    # Save first 10 images
    if i < 10:
        orig = images.squeeze().cpu().detach().permute(1,2,0).numpy()
        adv = perturbed.squeeze().cpu().detach().permute(1,2,0).numpy()

        wandb.log({
            f"FGSM_sample_{i}": [
                wandb.Image(orig, caption="Original"),
                wandb.Image(adv, caption="Adversarial")
            ]
        })

total = len(test_loader)

print("Clean Accuracy:", correct_clean / total)
print("FGSM Accuracy:", correct_adv / total)
print("Accuracy Drop:", (correct_clean - correct_adv)/total)