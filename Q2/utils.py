import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_metrics(preds, targets, num_classes=23, smooth=1e-6):
    preds = torch.argmax(preds, dim=1)

    ious = []
    dices = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        pred_sum = pred_cls.sum().item()
        target_sum = target_cls.sum().item()

        if union == 0:
            continue

        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)

        ious.append(iou)
        dices.append(dice)

    miou = np.mean(ious) if len(ious) > 0 else 0.0
    mdice = np.mean(dices) if len(dices) > 0 else 0.0
    return miou, mdice

def save_training_plots(train_losses, train_ious, train_dices, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_ious)
    plt.title("Training mIOU")
    plt.xlabel("Epoch")
    plt.ylabel("mIOU")
    plt.savefig(os.path.join(save_dir, "miou_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_dices)
    plt.title("Training mDice")
    plt.xlabel("Epoch")
    plt.ylabel("mDice")
    plt.savefig(os.path.join(save_dir, "mdice_curve.png"))
    plt.close()