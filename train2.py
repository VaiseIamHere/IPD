import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from MedMamba import VSSM as medmamba

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

    # Transforms
    data_transform = {
        "train": transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # Directories
    train_dir = "/kaggle/input/retinamnist/retinaMNIST/train"
    val_dir = "/kaggle/input/retinamnist/retinaMNIST/val"  # <<< Add your validation folder here

    # Datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # Save class index mapping
    class_to_idx = train_dataset.class_to_idx

    batch_size = int(sys.argv[2])
    num_workers = int(sys.argv[3])
    print(f"Batch Size: {batch_size}, Dataloader Workers: {num_workers}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model, loss, optimizer, scheduler
    num_classes = len(class_to_idx)
    net = medmamba(num_classes=num_classes, activationOption=sys.argv[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Save model details
    model_name = f"mamba_t_{sys.argv[1]}"
    file_path = f"/kaggle/working/{model_name}_details.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            model_details = json.load(f)
    else:
        model_details = {
            "model_name": model_name,
            "training_params": {
                "loss": "CrossEntropy",
                "optimizer": "AdamW",
                "batch_size": batch_size,
                "activationOption": sys.argv[1],
                "num_workers": num_workers
            },
            "metrics": []
        }

    epochs = 100
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} started", flush=True)
        
        # Training
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / train_steps
        train_acc = correct_train / total_train

        # Validation
        net.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / val_steps
        val_acc = correct_val / total_val

        # Step scheduler
        scheduler.step(avg_val_loss)

        # Save metrics
        model_details["metrics"].append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "train_loss": avg_train_loss,
            "val_acc": val_acc,
            "val_loss": avg_val_loss
        })

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"/kaggle/working/checkpoints/{model_name}_checkpoint{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

            with open(file_path, "w") as f:
                json.dump(model_details, f, indent=4)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == '__main__':
    main()
