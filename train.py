import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json

import torch
print("Device Count: ", torch.cuda.device_count())
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba  # import your model

def main():
    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

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
    with open('class_indices.json', 'w') as json_file:
        json.dump(idx_to_class, json_file, indent=4)

    batch_size = 8
    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"Using {0} dataloader workers.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers
        num_workers=0
    )

    print(f"Loaded {train_num} training images.")

    # Model, loss, optimizer
    num_classes = len(class_to_idx)
    net = medmamba(num_classes=num_classes)
    # net = nn.DataParallel(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # Training loop
    epochs = 100
    model_name = 'mamba_vaibhav'
    save_path = f'./{model_name}Net.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.3f}"
            if device == 'cpu':
                print("Training Started you can switch to GPU probably")
                exit(0)

        avg_loss = running_loss / train_steps
        print(f"[Epoch {epoch+1}] Average Training Loss: {avg_loss:.3f}")

    # Save final model
    torch.save(net.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == '__main__':
    main()
