import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import CityscapesDataset
from model import UNet
from utils import compute_metrics, save_training_plots

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 23
    batch_size = 8
    epochs = 15
    lr = 1e-3

    image_dir = "data/CameraRGB"
    mask_dir = "data/CameraMask"

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_dataset = CityscapesDataset(train_imgs, train_masks)
    test_dataset = CityscapesDataset(test_imgs, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("outputs", exist_ok=True)

    train_losses = []
    train_ious = []
    train_dices = []

    best_miou = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_ious = []
        epoch_dices = []

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            miou, mdice = compute_metrics(outputs.detach().cpu(), masks.detach().cpu(), num_classes)
            epoch_ious.append(miou)
            epoch_dices.append(mdice)

        avg_loss = running_loss / len(train_loader)
        avg_miou = sum(epoch_ious) / len(epoch_ious)
        avg_mdice = sum(epoch_dices) / len(epoch_dices)

        train_losses.append(avg_loss)
        train_ious.append(avg_miou)
        train_dices.append(avg_mdice)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} mIOU: {avg_miou:.4f} mDice: {avg_mdice:.4f}")

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), "outputs/best_model.pth")

    save_training_plots(train_losses, train_ious, train_dices)

    print("Training complete.")
    print("Best training mIOU:", best_miou)

if __name__ == "__main__":
    main()