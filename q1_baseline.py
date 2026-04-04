import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from tqdm import tqdm
import wandb

# -----------------------
# CONFIG
# -----------------------
BATCH_SIZE = 64
EPOCHS = 5
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# WANDB
# -----------------------
wandb.init(project="assignment5-vit-baseline")

# -----------------------
# DATA
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # IMPORTANT for ViT
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -----------------------
# MODEL
# -----------------------
model = ViTForImageClassification.from_pretrained(
    "facebook/deit-small-patch16-224",
    num_labels=100,
    ignore_mismatched_sizes=True	
)

model.to(DEVICE)

# -----------------------
# OPTIMIZER + LOSS
# -----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# -----------------------
# ACCURACY FUNCTION
# -----------------------
def accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    return (preds == labels).float().mean().item()

# -----------------------
# TRAIN LOOP
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_acc = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs.logits, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            val_loss += loss.item()
            val_acc += accuracy(outputs.logits, labels)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # -----------------------
    # LOG
    # -----------------------
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "outputs/vit_baseline.pth")
