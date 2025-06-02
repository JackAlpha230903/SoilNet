import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from tqdm.notebook import tqdm
import timm
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

df = pd.read_csv(".../labels_new.csv")
print(f"Total labeled images before check: {len(df)}")

required_columns = ['path', 'SM_0', 'SM_20', 'light_value', 'moisture_class']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"DataFrame missing columns: {missing_columns}")

fallback_dir = ".../augmented_fallback"
os.makedirs(fallback_dir, exist_ok=True)
replaced_count = 0
augment = transforms.ColorJitter(brightness=0.2, contrast=0.2)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = row["path"]
    try:
        img = Image.open(path).convert("RGB")
        img.verify()
    except:
        print(f"Image error: {path}")
        folder = os.path.dirname(path)
        all_images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        good_images = [f for f in all_images if f != os.path.basename(path)]
        if not good_images:
            print(f"No replacement images in: {folder}")
            continue
        candidate = random.choice(good_images)
        candidate_path = os.path.join(folder, candidate)
        try:
            img = Image.open(candidate_path).convert("RGB")
            img_aug = augment(img)
            new_filename = f"aug_{os.path.basename(path)}"
            new_path = os.path.join(fallback_dir, new_filename)
            img_aug.save(new_path)
            df.at[idx, "path"] = new_path
            replaced_count += 1
            print(f"Replaced with: {new_path}")
        except:
            print(f"Replacement image error: {candidate_path}")
            continue
print(f"Replaced {replaced_count} invalid images with augmented versions.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SoilDualTaskDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        light = torch.tensor([row["light_value"] / 100.0], dtype=torch.float32)
        regression_target = torch.tensor([row["SM_0"] / 100.0, row["SM_20"] / 100.0], dtype=torch.float32)
        class_target = torch.tensor(row["moisture_class"], dtype=torch.long)
        return {
            "image": img,
            "light": light,
            "regression_target": regression_target,
            "class_target": class_target
        }

class SoilNetDualHead(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
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
        self.light_dense = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_light):
        x = self.initial_conv(x_img)
        x = self.mnv2_block1(x)
        x = self.channel_adapter(x)
        x = self.mobilevit_encoder(x)
        x = self.mvit_to_mnv2(x)
        x = self.mnv2_block2(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x_img_feat = torch.flatten(x, 1)
        x_light_feat = self.light_dense(x_light)
        x_concat = torch.cat([x_img_feat, x_light_feat], dim=1)
        reg_out = self.reg_head(x_concat)
        cls_out = self.cls_head(x_concat)
        return reg_out, cls_out

dataset = SoilDualTaskDataset(df, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True if device.type == "cuda" else False, num_workers=0)

num_classes = len(df["moisture_class"].unique())
model = SoilNetDualHead(num_classes=num_classes).to(device)

checkpoint_path = ".../"Model_name".pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_state_dict.items() 
                       if k in model_dict and model_dict[k].shape == v.shape}
    temp_model_dict = model_dict.copy()
    temp_model_dict.update(pretrained_dict)
    missing_keys, unexpected_keys = model.load_state_dict(temp_model_dict, strict=False)
    print(f"Loaded pretrained weights from {checkpoint_path}")
    if missing_keys:
        print(f"Some keys missing in checkpoint and not loaded: {missing_keys}")
    if unexpected_keys:
        print(f"Some keys in checkpoint not in model and ignored: {unexpected_keys}")
except FileNotFoundError:
    print(f"Checkpoint not found: {checkpoint_path}. Using random initialization.")
except Exception as e:
    print(f"Error loading checkpoint: {e}. Using random initialization.")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
reg_criterion = nn.MSELoss()
cls_criterion = nn.CrossEntropyLoss()

num_epochs = 100
loss_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss, total_reg_loss, total_cls_loss = 0, 0, 0
    for batch in tqdm(dataloader, leave=False):
        x_img = batch["image"].to(device)
        x_light = batch["light"].to(device)
        y_reg = batch["regression_target"].to(device)
        y_cls = batch["class_target"].to(device)
        optimizer.zero_grad()
        pred_reg, pred_cls = model(x_img, x_light)
        loss_reg = reg_criterion(pred_reg, y_reg)
        loss_cls = cls_criterion(pred_cls, y_cls)
        loss = loss_reg + loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_reg_loss += loss_reg.item()
        total_cls_loss += loss_cls.item()
    avg_total = total_loss / len(dataloader)
    loss_history.append((avg_total, total_reg_loss / len(dataloader), total_cls_loss / len(dataloader)))
    print(f"Loss → Total: {avg_total:.4f}, Regression: {total_reg_loss / len(dataloader):.4f}, Classification: {total_cls_loss / len(dataloader):.4f}")
    model.eval()
    y_true_reg, y_pred_reg = [], []
    y_true_cls, y_pred_cls = [], []
    with torch.no_grad():
        for batch in dataloader:
            x_img = batch["image"].to(device)
            x_light = batch["light"].to(device)
            y_reg = batch["regression_target"].to(device)
            y_cls = batch["class_target"].to(device)
            pred_reg, pred_cls = model(x_img, x_light)
            y_true_reg.extend(y_reg.cpu().numpy())
            y_pred_reg.extend(pred_reg.cpu().numpy())
            y_true_cls.extend(y_cls.cpu().numpy())
            y_pred_cls.extend(pred_cls.argmax(dim=1).cpu().numpy())
    y_true_reg = np.array(y_true_reg) * 100
    y_pred_reg = np.array(y_pred_reg) * 100
    metrics = {}
    for i, label in enumerate(["SM_0", "SM_20"]):
        metrics[label] = {
            "RMSE": mean_squared_error(y_true_reg[:, i], y_pred_reg[:, i]) ** 0.5,
            "MAE": mean_absolute_error(y_true_reg[:, i], y_pred_reg[:, i]),
            "ME": np.mean(y_pred_reg[:, i] - y_true_reg[:, i]),
        }
    metrics["Classification"] = {
        "Accuracy": accuracy_score(y_true_cls, y_pred_cls),
        "F1-score": f1_score(y_true_cls, y_pred_cls, average="weighted")
    }
    for label in ["SM_0", "SM_20"]:
        print(f"{label} → RMSE: {metrics[label]['RMSE']:.8f}, MAE: {metrics[label]['MAE']:.8f}, ME: {metrics[label]['ME']:.8f}")
    print(f"Classification → Accuracy: {metrics['Classification']['Accuracy']:.8f}, F1-score: {metrics['Classification']['F1-score']:.8f}")

torch.save(model.state_dict(), ".../"Model_name"_finetuned.pth")