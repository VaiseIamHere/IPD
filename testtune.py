import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import json
import gc

from MedMamba import VSSM as medmamba

num_classes = 5
checkpoints = [10*i for i in range(1, 11)]

print("CheckPoints: ", checkpoints)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dir = "/kaggle/input/mnistdata/extracted_images/test"
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

test_num = len(test_dataset)
print(f"Loaded {test_num} test images.")
metrics = []

activationOption = sys.argv[1]

for checkpoint in checkpoints:
    print(f"Checkpoint: {checkpoint}, ", end=" ", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = medmamba(num_classes=num_classes, activationOption=activationOption)
    net = net.to(device)

    # load_path = f"/kaggle/input/{notebook_name}/checkpoints/mamba_{activationOption}_checkpoint{checkpoint}.pth"
    model_path = f"/kaggle/working/checkpoints/mamba_tuned_checkpoint{checkpoint}.pth"
    f = torch.load(model_path, weights_only=True)
    net.load_state_dict(f["model_state_dict"], strict=True)
    net.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / test_num

    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print("All Predictions: ")
    print(all_predictions)

    print("All Labels: ")
    print(all_labels)

    print(f"Test Accuracy: {accuracy:.2f}%", flush=True)
    print("Precison: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Conf Matrix ->")
    print(conf_matrix)

    metrics.append({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "conf_matrix": conf_matrix.tolist(),
    })

    del net
    torch.cuda.empty_cache()
    gc.collect()

json_path = f"/kaggle/working/mnist_tuned_metrices.json"
with open(json_path, 'w') as f:
    json.dump(metrics, f, indent=4)
