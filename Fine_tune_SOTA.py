import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import timm
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import json
from google.colab import drive

drive.mount('/content/drive')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-4
NUM_CLASSES = 10
MODEL_DIR = Path("/content/drive/MyDrive/SoilNet_Checkpoints/trained_models_SOTA")
MODEL_DIR.mkdir(exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

df = pd.read_csv("/content/drive/MyDrive/SoilMoistureProject/label_updated_colab.csv")

fallback_dir = "/content/drive/MyDrive/SoilMoistureProject/data/labeled_data/augmented_fallback"
os.makedirs(fallback_dir, exist_ok=True)

augment = transforms.ColorJitter(brightness=0.2, contrast=0.2)
replaced_count = 0
print("Checking and replacing invalid images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = row["path"]
    try:
        img = Image.open(path).convert("RGB")
        img.verify()
    except (UnidentifiedImageError, FileNotFoundError, OSError):
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
        except Exception as e:
            print(f"Replacement image error: {candidate_path} ({e})")
print(f"Replaced {replaced_count} invalid images with augmented versions.")

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class SoilDualTaskDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe
        self.t = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        img = self.t(img)
        light = torch.tensor([r["light_value"]/100], dtype=torch.float32)
        y_reg = torch.tensor([r["SM_0"]/100, r["SM_20"]/100], dtype=torch.float32)
        y_cls = torch.tensor(r["moisture_class"], dtype=torch.long)
        return {"image": img, "light": light, "y_reg": y_reg, "y_cls": y_cls}

ds = SoilDualTaskDataset(df, tfm)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

class SoilNetDualHead(nn.Module):
    def __init__(self, backbone_name="mobilevitv2_050", num_classes=10, light_feat_dim=32):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")
        if hasattr(self.backbone, 'num_features'):
            backbone_out = self.backbone.num_features
        elif hasattr(self.backbone, 'feature_info'):
            backbone_out = self.backbone.feature_info.channels()[-1]
        else:
            raise ValueError(f"Cannot determine output feature of backbone '{backbone_name}'")
        self.light_fc = nn.Sequential(
            nn.Linear(1, light_feat_dim),
            nn.ReLU()
        )
        total_feat = backbone_out + light_feat_dim
        self.reg_head = nn.Sequential(
            nn.Linear(total_feat, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(total_feat, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_light):
        feat = self.backbone(x_img)
        light_feat = self.light_fc(x_light)
        x = torch.cat([feat, light_feat], dim=1)
        return self.reg_head(x), self.cls_head(x)

MODELS = {
    "mobilevitv2_050": lambda: SoilNetDualHead("mobilevitv2_050"),
    "mobilevit_s": lambda: SoilNetDualHead("mobilevit_s"),
    "efficientnet_b0": lambda: SoilNetDualHead("efficientnet_b0"),
    "mobilenetv2_100": lambda: SoilNetDualHead("mobilenetv2_100"),
}

def train_epoch(model, loader, opt, scaler, reg_crit, cls_crit):
    model.train()
    tot = reg = cls = 0
    pbar = tqdm(loader, leave=False)
    for b in pbar:
        x = b["image"].to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        l = b["light"].to(DEVICE, non_blocking=True)
        yr = b["y_reg"].to(DEVICE, non_blocking=True)
        yc = b["y_cls"].to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            pr, pc = model(x, l)
            loss_r = reg_crit(pr, yr)
            loss_c = cls_crit(pc, yc)
            loss = loss_r + loss_c
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item()
        reg += loss_r.item()
        cls += loss_c.item()
        pbar.set_description(f"L={loss.item():.3f}")
    n = len(loader)
    return tot/n, reg/n, cls/n

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    ytr, ypr, ytc, ypc = [], [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(DEVICE)
            l = b["light"].to(DEVICE)
            pr, pc = model(x, l)
            ytr.append(b["y_reg"].cpu())
            ytc.append(b["y_cls"].cpu())
            ypr.append(pr.cpu())
            ypc.append(pc.argmax(1).cpu())
    ytr = torch.cat(ytr).numpy() * 100
    ypr = torch.cat(ypr).numpy() * 100
    ytc = torch.cat(ytc).numpy()
    ypc = torch.cat(ypc).numpy()
    metric = {}
    for i, lab in enumerate(["SM_0", "SM_20"]):
        metric[lab] = {
            "RMSE": mean_squared_error(ytr[:, i], ypr[:, i]) ** 0.5,
            "MAE": mean_absolute_error(ytr[:, i], ypr[:, i]),
            "ME": float(np.mean(ypr[:, i] - ytr[:, i]))
        }
    metric["Acc"] = accuracy_score(ytc, ypc)
    metric["F1"] = f1_score(ytc, ypc, average="weighted")
    return metric

def save_everything(model, name, optimizer, loss_history, save_dir="/content/drive/MyDrive/SoilNet_Checkpoints/trained_models"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"{name}.pt")
    torch.save(model, save_path / f"{name}_entire.pth")
    torch.save({
        "model_name": name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss_history": loss_history
    }, save_path / f"{name}_full.pth")
    with open(save_path / f"{name}_loss.json", "w") as f:
        json.dump(loss_history, f)
    print(f"Saved model '{name}' to '{save_dir}'")

criterion_reg = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()
all_hist = {}

for name, fn in MODELS.items():
    print(f"Training {name} …")
    model = fn().to(DEVICE, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
    scaler = GradScaler()
    hist = []
    early_stop = False
    for ep in range(NUM_EPOCHS):
        t0 = time.time()
        tot, reg, cls = train_epoch(model, loader, opt, scaler, criterion_reg, criterion_cls)
        hist.append((tot, reg, cls))
        metric = evaluate(model, loader)
        print(f"Epoch {ep+1}/{NUM_EPOCHS}")
        print(f"Loss → Total: {tot:.8f}, Regression: {reg:.8f}, Classification: {cls:.8f}")
        for sm in ["SM_0", "SM_20"]:
            rmse = metric[sm]["RMSE"]
            mae = metric[sm]["MAE"]
            me = metric[sm].get("ME", 0.0)
            print(f"{sm} → RMSE: {rmse:.8f}, MAE: {mae:.8f}, ME: {me:.8f}")
        acc = metric["Acc"]
        f1 = metric["F1"]
        print(f"Classification → Accuracy: {acc:.8f}, F1-score: {f1:.8f} | Time: {time.time()-t0:.1f}s")
        if ep > 0:
            prev_loss = hist[ep-1][0]
            loss_increase = tot - prev_loss
            if loss_increase > 0.2:
                print(f"Early stopping at epoch {ep+1}: Loss increased from {prev_loss:.8f} → {tot:.4f} (Δ={loss_increase:.8f})")
                early_stop = True
                break
    save_everything(model, name, opt, hist)
    all_hist[name] = hist
    del model
    torch.cuda.empty_cache()