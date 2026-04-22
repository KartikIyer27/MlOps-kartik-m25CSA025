import os
import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json

from dataset import CityscapesDataset
from model import UNet
from utils import compute_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 23

    image_dir = "data/CameraRGB"
    mask_dir = "data/CameraMask"

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    test_dataset = CityscapesDataset(test_imgs, test_masks)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.eval()

    all_ious = []
    all_dices = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            miou, mdice = compute_metrics(outputs.cpu(), masks.cpu(), num_classes)
            all_ious.append(miou)
            all_dices.append(mdice)

    test_miou = sum(all_ious) / len(all_ious)
    test_mdice = sum(all_dices) / len(all_dices)

    print(f"Test mIOU: {test_miou:.4f}")
    print(f"Test mDice: {test_mdice:.4f}")

    with open("outputs/test_metrics.json", "w") as f:
        json.dump({
            "test_miou": round(test_miou, 4),
            "test_mdice": round(test_mdice, 4)
        }, f, indent=4)

if __name__ == "__main__":
    main()