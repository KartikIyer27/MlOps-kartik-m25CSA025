import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
import numpy as np
import os
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs/adv_data", exist_ok=True)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("outputs/resnet18.pth"))
model.to(DEVICE)
model.eval()

classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    input_shape=(3,32,32),
    nb_classes=10,
    clip_values=(-1,1)
)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128)

# Attacks
pgd = ProjectedGradientDescent(estimator=classifier, eps=0.1)
bim = BasicIterativeMethod(estimator=classifier, eps=0.1)

pgd_correct = 0
bim_correct = 0
total = 0

wandb.init(project="q2-adversarial-visualization", name="PGD_BIM")

for i, (images, labels) in enumerate(loader):

    images_np = images.numpy()
    labels_np = labels.numpy()

    # -----------------------
    # GENERATE ATTACKS
    # -----------------------
    adv_pgd = pgd.generate(x=images_np)
    adv_bim = bim.generate(x=images_np)

    # -----------------------
    # WANDB LOGGING (FIRST 10 SAMPLES)
    # -----------------------
    if i < 10:
        orig = images[0].permute(1,2,0).numpy()
        pgd_img = adv_pgd[0].transpose(1,2,0)
        bim_img = adv_bim[0].transpose(1,2,0)

        # normalize for visualization
        orig = (orig + 1) / 2
        pgd_img = (pgd_img + 1) / 2
        bim_img = (bim_img + 1) / 2

        wandb.log({
            f"PGD_sample_{i}": [
                wandb.Image(orig, caption="Original"),
                wandb.Image(pgd_img, caption="PGD")
            ],
            f"BIM_sample_{i}": [
                wandb.Image(orig, caption="Original"),
                wandb.Image(bim_img, caption="BIM")
            ]
        })

    # -----------------------
    # EVALUATE PGD
    # -----------------------
    preds_pgd = classifier.predict(adv_pgd)
    preds_pgd = np.argmax(preds_pgd, axis=1)
    pgd_correct += (preds_pgd == labels_np).sum()

    # -----------------------
    # EVALUATE BIM
    # -----------------------
    preds_bim = classifier.predict(adv_bim)
    preds_bim = np.argmax(preds_bim, axis=1)
    bim_correct += (preds_bim == labels_np).sum()

    total += len(labels_np)

    # -----------------------
    # LIMIT FOR SPEED
    # -----------------------
    if i == 30:
        break


# for images, labels in loader:
#     images_np = images.numpy()
#     labels_np = labels.numpy()

#     # PGD
#     adv_pgd = pgd.generate(x=images_np)
#     preds_pgd = np.argmax(classifier.predict(adv_pgd), axis=1)
#     pgd_correct += (preds_pgd == labels_np).sum()

#     # BIM
#     adv_bim = bim.generate(x=images_np)
#     preds_bim = np.argmax(classifier.predict(adv_bim), axis=1)
#     bim_correct += (preds_bim == labels_np).sum()

#     total += len(labels_np)

print("PGD Accuracy:", pgd_correct/total)
print("BIM Accuracy:", bim_correct/total)