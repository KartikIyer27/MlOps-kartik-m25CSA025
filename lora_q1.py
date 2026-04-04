import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import wandb
import os

# -----------------------
# PERFORMANCE + STABILITY
# -----------------------
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

# -----------------------
# CONFIG
# -----------------------
BATCH_SIZE = 64
EPOCHS = 10  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "").strip()
WANDB_PROJECT = "assignment5-vit-lora"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "").strip() or None
WANDB_ENABLED = True

os.makedirs("outputs", exist_ok=True)

try:
    # Prefer explicit API key from env, but fall back to existing local W&B login.
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY, relogin=True)
    else:
        wandb.login(relogin=False)
except Exception as exc:
    WANDB_ENABLED = False
    print(f"W&B login failed, continuing without W&B: {exc}")

# -----------------------
# DATA
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------
# FUNCTIONS
# -----------------------
def accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    return (preds == labels).float().mean().item()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# EXPERIMENT LOOP
# -----------------------
# FULL RUN
ranks = [2, 4, 8]
alphas = [2, 4, 8]

# TEST RUN
# ranks = [2]
# alphas = [2, 4]

for r in ranks:
    for alpha in alphas:

        print(f"\n===== Running LoRA: r={r}, alpha={alpha} =====")
        run = None

        if WANDB_ENABLED:
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"r{r}_alpha{alpha}",
                reinit=True,
                mode="online",
                config={
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": 5e-5,
                    "lora_r": r,
                    "lora_alpha": alpha,
                    "lora_dropout": 0.1,
                    "model": "facebook/deit-small-patch16-224"
                }
            )
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("epoch")
            wandb.define_metric("epoch/*", step_metric="epoch")
            if run is not None:
                print(f"W&B run URL: {run.url}")

        # -----------------------
        # MODEL
        # -----------------------
        model = ViTForImageClassification.from_pretrained(
            "facebook/deit-small-patch16-224",
            num_labels=100,
            ignore_mismatched_sizes=True
        )

        # -----------------------
        # APPLY LoRA
        # -----------------------
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none"
        )

        model = get_peft_model(model, lora_config)
        model.to(DEVICE)

        # 🔥 Print trainable params
        trainable_params = count_trainable_params(model)
        print(f"Trainable Parameters: {trainable_params}")

        # -----------------------
        # OPTIMIZER
        # -----------------------
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        criterion = torch.nn.CrossEntropyLoss()

        # -----------------------
        # TRAIN LOOP
        # -----------------------
        global_step = 0
        for epoch in range(EPOCHS):

            model.train()
            train_loss, train_acc = 0, 0

            for step, (images, labels) in enumerate(tqdm(train_loader), start=1):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += accuracy(outputs.logits, labels)
                global_step += 1

                if WANDB_ENABLED and (step % 20 == 0 or step == len(train_loader)):
                    wandb.log({
                        "train/step": global_step,
                        "train/loss_step": loss.item(),
                        "train/acc_step": accuracy(outputs.logits, labels)
                    }, step=global_step)

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            # -----------------------
            # VALIDATION
            # -----------------------
            model.eval()
            val_loss, val_acc = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    outputs = model(images)
                    loss = criterion(outputs.logits, labels)

                    val_loss += loss.item()
                    val_acc += accuracy(outputs.logits, labels)

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            print(f"Epoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if WANDB_ENABLED:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/train_loss": train_loss,
                    "epoch/train_acc": train_acc,
                    "epoch/val_loss": val_loss,
                    "epoch/val_acc": val_acc
                }, step=global_step)

        # -----------------------
        # FINAL TEST ACCURACY
        # -----------------------
        test_acc = evaluate(model, val_loader)
        print(f"Final Test Accuracy: {test_acc:.4f}")
        if WANDB_ENABLED:
            wandb.summary["final_test_acc"] = test_acc
            wandb.summary["trainable_params"] = trainable_params

        # -----------------------
        # SAVE MODEL
        # -----------------------
        torch.save(model.state_dict(), f"outputs/lora_r{r}_alpha{alpha}.pth")

        # -----------------------
        # SAVE RESULTS (AUTO TABLE)
        # -----------------------
        with open("outputs/results.txt", "a") as f:
            f.write(f"LoRA(QKV),{r},{alpha},0.1,{test_acc:.4f},{trainable_params}\n")

        if WANDB_ENABLED:
            wandb.finish()