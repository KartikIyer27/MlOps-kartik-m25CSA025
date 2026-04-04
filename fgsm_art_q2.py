import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import numpy as np
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("outputs/resnet18.pth"))
model.to(DEVICE)
model.eval()

# Wrap model for ART
classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(-1, 1)
)

wandb.init(project="q2-adversarial-visualization", name="FGSM_ART")

# Data
# transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128)

attack = FastGradientMethod(estimator=classifier, eps=0.1)

correct = 0
total = 0

for i, (images, labels) in enumerate(test_loader):

    images_np = images.numpy()
    labels_np = labels.numpy()

    # generate first
    adv_images = attack.generate(x=images_np)

    # logging after generation
    if i < 10:
        orig = images[0].permute(1,2,0).numpy()
        adv = adv_images[0].transpose(1,2,0)

        # optional normalization fix
        orig = (orig + 1) / 2
        adv = (adv + 1) / 2

        wandb.log({
            f"FGSM_ART_sample_{i}": [
                wandb.Image(orig, caption="Original"),
                wandb.Image(adv, caption="Adversarial")
            ]
        })

    preds = classifier.predict(adv_images)
    preds = np.argmax(preds, axis=1)

    correct += (preds == labels_np).sum()
    total += len(labels_np)

print("ART FGSM Accuracy:", correct/total)