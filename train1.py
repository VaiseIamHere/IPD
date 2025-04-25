import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import sys
import torch
print("Cuda Device Count: ", torch.cuda.device_count())

import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    print(sys.argv)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # Training dataset and dataloader
    train_dir = "/kaggle/input/mnistdata/extracted_images/train"
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    train_num = len(train_dataset)

    # Save class index mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    batch_size = int(sys.argv[2])
    num_workers = int(sys.argv[3])
    print(f"Batch Size: ", batch_size)
    print(f"Using {num_workers} dataloader workers.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory= True
    )

    print(f"Loaded {train_num} training images.")

    # Model, loss, optimizer
    num_classes = len(class_to_idx)
    net = medmamba(num_classes=num_classes, activationOption=sys.argv[1])
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # Training loop
    epochs = 100
    model_name = "mnist.pth"
    train_steps = len(train_loader)
    file_path = "/kaggle/working/mnist_details.json"
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            model_details = json.load(f)
    else:
        model_details = {
            "model_name": model_name,
            "training_params": {
                "loss": "CrossEntropy",
                "optimizer": "Adam",
                "batch_size": batch_size,
                "activationOption": sys.argv[1],
                "num_workers": num_workers
            },
            "metrics": []
        }

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} started", flush=True)
        net.train()
        running_loss = 0.0
        total_train, correct_train = 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels.to(device)).sum().item()
            total_train += labels.size(0)
        
        train_acc = correct_train / total_train
        avg_loss = running_loss / train_steps

        model_details["metrics"].append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "avg_loss": avg_loss
        })

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"/kaggle/working/checkpoints/mnist_checkpoint{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            with open(file_path, "w") as f:
                json.dump(model_details, f, indent=4)

        print(f"[Epoch {epoch+1}] Average Training Loss: {avg_loss:.3f}", flush=True)

if __name__ == '__main__':
    main()
