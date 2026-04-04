import os
import json
import torch
import optuna
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# -----------------------
# STABILITY
# -----------------------
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy("file_system")

# -----------------------
# CONFIG
# -----------------------
BATCH_SIZE = 64
OPTUNA_EPOCHS = 3   # keep small for faster search
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/deit-small-patch16-224"

os.makedirs("outputs", exist_ok=True)

# -----------------------
# DATA
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
val_dataset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# -----------------------
# HELPERS
# -----------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# OPTUNA OBJECTIVE
# -----------------------
def objective(trial):
    r = trial.suggest_categorical("r", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])
    lr = trial.suggest_categorical("lr", [1e-4, 5e-5, 2e-5])

    print(f"\nTrial {trial.number}: r={r}, alpha={alpha}, lr={lr}")

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=100,
        ignore_mismatched_sizes=True
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(OPTUNA_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Trial {trial.number} | Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        trial.report(val_acc, step=epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trainable_params = count_trainable_params(model)

    with open("outputs/optuna_trials.txt", "a") as f:
        f.write(
            f"trial={trial.number},r={r},alpha={alpha},lr={lr},"
            f"best_val_acc={best_val_acc:.4f},trainable_params={trainable_params}\n"
        )

    return best_val_acc

# -----------------------
# RUN STUDY
# -----------------------
if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="q1_lora_optuna"
    )

    study.optimize(objective, n_trials=10)

    print("\nBest Trial:")
    print(f"  Value (Best Val Acc): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    with open("outputs/optuna_best.json", "w") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_params
            },
            f,
            indent=4
        )

    df = study.trials_dataframe()
    df.to_csv("outputs/optuna_trials.csv", index=False)

    print("\nSaved:")
    print(" - outputs/optuna_best.json")
    print(" - outputs/optuna_trials.csv")
    print(" - outputs/optuna_trials.txt")