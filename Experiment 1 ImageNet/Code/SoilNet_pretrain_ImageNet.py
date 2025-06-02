import time
import threading
import torch
import torch.nn as nn
import timm
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def keep_alive():
    while True:
        print("Keeping alive...", end='\r')
        time.sleep(600)

alive_thread = threading.Thread(target=keep_alive)
alive_thread.daemon = True
alive_thread.start()

print("Keep-alive thread started.")

!wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
!unzip -q tiny-imagenet-200.zip

from google.colab import drive
drive.mount('/content/drive')

save_dir = '/content/drive/MyDrive/SoilNet_Checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())

image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_dir = './tiny-imagenet-200/train'
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
print(f"Total images: {len(dataset)}, Number of class: {num_classes}")

class SoilNetDualHead(nn.Module):
    def __init__(self, num_classes=200, use_light=False):
        super().__init__()
        self.use_light = use_light
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.mnv2_block1 = nn.Sequential(*list(
            timm.create_model("mobilenetv2_100.ra_in1k", pretrained=True).blocks.children())[0:3]
        )
        self.channel_adapter = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        self.mobilevit_full = timm.create_model("mobilevitv2_050", pretrained=True)
        self.mobilevit_encoder = self.mobilevit_full.stages
        self.mvit_to_mnv2 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.mnv2_block2 = nn.Sequential(*list(
            timm.create_model("mobilenetv2_100.ra_in1k", pretrained=True).blocks.children())[3:7]
        )
        self.final_conv = nn.Conv2d(320, 1280, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if use_light:
            self.light_dense = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
            self.reg_head = nn.Sequential(
                nn.Linear(1280 + 32, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)
            )
        self.cls_head = nn.Sequential(
            nn.Linear(1280 if not use_light else 1280 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_light=None):
        x = self.initial_conv(x_img)
        x = self.mnv2_block1(x)
        x = self.channel_adapter(x)
        x = self.mobilevit_encoder(x)
        x = self.mvit_to_mnv2(x)
        x = self.mnv2_block2(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x_img_feat = torch.flatten(x, 1)

        if self.use_light and x_light is not None:
            x_light_feat = self.light_dense(x_light)
            x_concat = torch.cat([x_img_feat, x_light_feat], dim=1)
            reg_out = self.reg_head(x_concat)
            cls_out = self.cls_head(x_concat)
            return reg_out, cls_out
        else:
            cls_out = self.cls_head(x_img_feat)
            return cls_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SoilNetDualHead(num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
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

    loop = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
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

    print(f"Epoch {epoch:3d}/{num_epochs} – "
          f"Loss: {avg_loss:.8f}, "
          f"Accuracy: {accuracy:.8f}%, "
          f"Precision: {precision:.8f}%, "
          f"Recall: {recall:.8f}%, "
          f"F1-Score: {f1:.8f}%")

    if epoch % 5 == 0:
        save_path = os.path.join(save_dir, f'soilnet_imagenet_epoch_{epoch}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint tại: {save_path}")

final_save_path = os.path.join(save_dir, 'Soilnet_pretrained_ImageNet.pth')
torch.save(model.state_dict(), final_save_path)

