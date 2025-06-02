import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_images(df):
    new_df = df.copy()
    deleted_count = 0
    valid_indices = []
    for idx, row in tqdm(new_df.iterrows(), total=len(new_df)):
        path = row["path"]
        try:
            img = Image.open(path).convert("RGB")
            img.verify()
            valid_indices.append(idx)
        except:
            if os.path.exists(path):
                os.remove(path)
                deleted_count += 1
                print(f"Deleted image: {path}")
    new_df = new_df.loc[valid_indices].reset_index(drop=True)
    print(f"Deleted {deleted_count} invalid labeled images. {len(new_df)} images remain.")
    return new_df

df = pd.read_csv(".../labels_new.csv")
print(f"Total labeled images before check: {len(df)}")
df = preprocess_images(df)
print(f"Total labeled images after check: {len(df)}")

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

class SoilNetDualHead(nn.Module):
    def __init__(self, num_classes):
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
        return reg_out, cls_out, x_concat

num_classes = len(df["moisture_class"].unique())
model = SoilNetDualHead(num_classes=num_classes).to(device)
checkpoint_path = ".../SoilNet_best.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded fine-tuned model from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

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

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = SoilDualTaskDataset(train_df, transform)
test_dataset = SoilDualTaskDataset(test_df, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def extract_features(loader, model, device):
    features_cls = []
    features_reg = []
    reg_targets = []
    cls_targets = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, unit="batch"):
            x_img = batch["image"].to(device)
            x_light = batch["light"].to(device)
            y_reg = batch["regression_target"]
            y_cls = batch["class_target"]
            reg_out, cls_out, _ = model(x_img, x_light)
            features_reg.extend(reg_out.cpu().numpy())
            features_cls.extend(cls_out.cpu().numpy())
            reg_targets.extend(y_reg.numpy())
            cls_targets.extend(y_cls.numpy())
    return (
        np.array(features_cls),
        np.array(features_reg),
        np.array(reg_targets),
        np.array(cls_targets)
    )

X_train_cls, X_train_reg, y_train_reg, y_train_cls = extract_features(train_loader, model, device)
X_test_cls, X_test_reg, y_test_reg, y_test_cls = extract_features(test_loader, model, device)

scaler_cls = StandardScaler()
X_train_cls_std = scaler_cls.fit_transform(X_train_cls)
X_test_cls_std = scaler_cls.transform(X_test_cls)

scaler_reg = StandardScaler()
X_train_reg_std = scaler_reg.fit_transform(X_train_reg)
X_test_reg_std = scaler_reg.transform(X_test_reg)

y_train_reg = y_train_reg * 100
y_test_reg = y_test_reg * 100

target_scaler_sm0 = StandardScaler()
target_scaler_sm20 = StandardScaler()
y_train_reg_sm0_scaled = target_scaler_sm0.fit_transform(y_train_reg[:, 0].reshape(-1, 1))
y_train_reg_sm20_scaled = target_scaler_sm20.fit_transform(y_train_reg[:, 1].reshape(-1, 1))

classifiers = {
    "SVM": (
        SVC(kernel="rbf", C=1, gamma='scale', class_weight='balanced', random_state=42, probability=True),
        SVR(kernel="rbf", C=1, gamma='scale', epsilon=0.1)
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    ),
    "GBM": (
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    ),
    "Extra Trees": (
        ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=42),
        ExtraTreesRegressor(n_estimators=100, max_depth=None, random_state=42)
    ),
    "AdaBoost": (
        AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42)
    ),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(max_iter=5000, learning_rate=0.01, max_depth=6, random_state=42),
        HistGradientBoostingRegressor(max_iter=5000, learning_rate=0.01, max_depth=6, random_state=42)
    ),
}

results = {}
for name, (cls_model, reg_model) in tqdm(classifiers.items(), unit="model"):
    print(f"Training {name}...")
    cls_model.fit(X_train_cls_std, y_train_cls)
    y_pred_cls = cls_model.predict(X_test_cls_std)
    class_report = classification_report(y_test_cls, y_pred_cls, output_dict=True)
    weighted_avg = class_report["weighted avg"]
    y_pred_reg = np.zeros_like(y_test_reg)
    for i, label in enumerate(["SM_0", "SM_20"]):
        if name == "SVM":
            if label == "SM_0":
                reg_model.fit(X_train_reg_std, y_train_reg_sm0_scaled.ravel())
                y_pred_reg_scaled = reg_model.predict(X_test_reg_std).reshape(-1, 1)
                y_pred_reg[:, i] = target_scaler_sm0.inverse_transform(y_pred_reg_scaled).ravel()
            else:
                reg_model.fit(X_train_reg_std, y_train_reg_sm20_scaled.ravel())
                y_pred_reg_scaled = reg_model.predict(X_test_reg_std).reshape(-1, 1)
                y_pred_reg[:, i] = target_scaler_sm20.inverse_transform(y_pred_reg_scaled).ravel()
        else:
            reg_model.fit(X_train_reg_std, y_train_reg[:, i])
            y_pred_reg[:, i] = reg_model.predict(X_test_reg_std)
    y_pred_reg = np.clip(y_pred_reg, 0, 100)
    regression_metrics = {
        "SM_0": {
            "RMSE": np.sqrt(mean_squared_error(y_test_reg[:, 0], y_pred_reg[:, 0])),
            "MAE": mean_absolute_error(y_test_reg[:, 0], y_pred_reg[:, 0]),
            "MSE": mean_squared_error(y_test_reg[:, 0], y_pred_reg[:, 0])
        },
        "SM_20": {
            "RMSE": np.sqrt(mean_squared_error(y_test_reg[:, 1], y_pred_reg[:, 1])),
            "MAE": mean_absolute_error(y_test_reg[:, 1], y_pred_reg[:, 1]),
            "MSE": mean_squared_error(y_test_reg[:, 1], y_pred_reg[:, 1])
        }
    }
    results[name] = {
        "Classification": {
            "Accuracy": accuracy_score(y_test_cls, y_pred_cls),
            "F1-Score": weighted_avg["f1-score"],
            "Precision": weighted_avg["precision"],
            "Recall": weighted_avg["recall"]
        },
        "Regression": regression_metrics
    }

print("So sánh hiệu suất (Using cls_head and reg_head features):")
for name in results:
    print(f"{name}:")
    print("Classification:")
    print(f"  Accuracy: {results[name]['Classification']['Accuracy']:.4f}")
    print(f"  Precision: {results[name]['Classification']['Precision']:.4f}")
    print(f"  Recall: {results[name]['Classification']['Recall']:.4f}")
    print(f"  F1-Score: {results[name]['Classification']['F1-Score']:.4f}")
    print("Regression:")
    for label in ["SM_0", "SM_20"]:
        reg = results[name]["Regression"][label]
        print(f"  {label}: RMSE: {reg['RMSE']:.4f}, MAE: {reg['MAE']:.4f}")