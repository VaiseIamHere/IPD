import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

from MedMamba import VSSM as medmamba

num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = medmamba(num_classes=num_classes)
net = net.to(device)

net.load_state_dict(torch.load('./mamba_vaibhavNet.pth'))
net.eval()

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dir = "/kaggle/input/drdataset/dataset/Testing"
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

test_num = len(test_dataset)
print(f"Loaded {test_num} test images.")

all_labels = []
all_predictions = []

with torch.no_grad():
    test_bar = tqdm(test_loader, file=sys.stdout)
    for images, labels in test_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        test_bar.desc = f"Test Progress: {100 * (len(all_predictions) / test_num):.2f}%"

accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / test_num
print(f"Test Accuracy: {accuracy:.2f}%")

precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

np.savetxt('./confusion_matrix.csv', conf_matrix, delimiter=',', fmt='%d')
print("Confusion matrix saved as 'confusion_matrix.csv'")
