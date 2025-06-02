import time
import threading
import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('/content/working/tiny-imagenet-200'):
    os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P /content/working')
    os.system('unzip -q /content/working/tiny-imagenet-200.zip -d /content/working')

save_dir = '/content/working/checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_dir = '/content/working/tiny-imagenet-200/train'
dataset = datasets.ImageFolder(root=image_dir, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True
)
num_classes = len(dataset.classes)
print(f"Total images: {len(dataset)}, Number of classes: {num_classes}")

models_to_train = [
    {'name': 'MobileNetV2', 'model': timm.create_model('mobilenetv2_100', pretrained=False, num_classes=num_classes)},
    {'name': 'MobileViTV2', 'model': timm.create_model('mobilevitv2_050', pretrained=False, num_classes=num_classes)}
]

def train_model(model, model_name, dataloader, num_epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    epoch_accuracies = []
    epoch_precisions = []
    epoch_recalls = []
    epoch_f1s = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        loop = tqdm(dataloader, desc=f"{model_name} Epoch {epoch}/{num_epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        precision = 100 * precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = 100 * recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = 100 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        epoch_precisions.append(precision)
        epoch_recalls.append(recall)
        epoch_f1s.append(f1)

        print(f"{model_name} Epoch {epoch:3d}/{num_epochs} – "
              f"Loss: {avg_loss:.8f}, "
              f"Accuracy: {accuracy:.8f}%, "
              f"Precision: {precision:.8f}%, "
              f"Recall: {recall:.8f}%, "
              f"F1-Score: {f1:.8f}%")

        if epoch % 5 == 0:
            save_path = os.path.join(save_dir, f'{model_name.lower()}_tinyimagenet_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint at: {save_path}")

    final_save_path = os.path.join(save_dir, f'{model_name.lower()}_pretrained_tinyimagenet.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f"Final {model_name} model saved at: {final_save_path}")

    return {
        'model_name': model_name,
        'losses': epoch_losses,
        'accuracies': epoch_accuracies,
        'precisions': epoch_precisions,
        'recalls': epoch_recalls,
        'f1_scores': epoch_f1s
    }

results = []
for model_info in models_to_train:
    print(f"\nStarting training for {model_info['name']}...")
    result = train_model(model_info['model'], model_info['name'], dataloader, num_epochs=100)
    results.append(result)

print("\n=== Final Performance Comparison ===")
for result in results:
    final_accuracy = result['accuracies'][-1]
    final_precision = result['precisions'][-1]
    final_recall = result['recalls'][-1]
    final_f1 = result['f1_scores'][-1]
    print(f"{result['model_name']}: "
          f"Accuracy: {final_accuracy:.8f}%, "
          f"Precision: {final_precision:.8f}%, "
          f"Recall: {final_recall:.8f}%, "
          f"F1-Score: {final_f1:.8f}%")

plt.figure(figsize=(12, 6))
for result in results:
    plt.plot(result['accuracies'], label=f"{result['model_name']} Accuracy")
plt.title('Model Accuracy Comparison on Tiny ImageNet')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'accuracy_comparison_tinyimagenet.png'))
plt.close()

print(f"Accuracy plot saved at: {os.path.join(save_dir, 'accuracy_comparison_tinyimagenet.png')}")