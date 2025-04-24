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
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    print(sys.argv)

    # Data transforms
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # Training dataset and dataloader
    train_dir = "/kaggle/input/drdataset/dataset/Training"
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    train_num = len(train_dataset)

    # Save class index mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open('/kaggle/working/class_indices.json', 'w') as json_file:
        json.dump(idx_to_class, json_file, indent=4)

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
    model_name = f"mamba_{sys.argv[1]}"
    train_steps = len(train_loader)
    
    def save_path(epoch):
        return f'/kaggle/working/{model_name}_epoch{epoch}.pth'
    
    model_details = {
        "model_name": model_name,
        "training_params": {
            "loss": "CrossEntropy",
            "optimizer": "Adam",
            "batch_size": batch_size,
            "activationOption": sys.argv[1],
            "num_workers": num_workers
        }
    }
    metrics = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        total_train, correct_train = 0, 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
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

            train_bar.desc = f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.3f}"

        if (epoch + 1) % 25 == 0:
            sp = save_path(epoch+1)
            torch.save(net.state_dict(), sp)
            print(f"Training Epoch: {epoch+1}. Model saved to {sp}")
        
        train_acc = correct_train / total_train
        avg_loss = running_loss / train_steps
        metrics.append({
            "epoch": epoch+1,
            "train_acc": train_acc,
            "avg_loss": avg_loss
        })
        print(f"[Epoch {epoch+1}] Average Training Loss: {avg_loss:.3f}")
    
    model_details["training_metrics"] = metrics
    print("**************************************")
    print(model_details)
    print("**************************************")
    with open(f"/kaggle/working/{sys.argv[1]}_details.json", "w") as f:
        json.dump(model_details, f, indent=8)

if __name__ == '__main__':
    main()
