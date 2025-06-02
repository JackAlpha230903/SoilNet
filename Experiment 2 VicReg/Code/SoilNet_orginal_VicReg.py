import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import glob
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import logging

from google.colab import drive
drive.mount('/content/drive')

!apt-get install unrar -y > /dev/null
rar_path = "/content/drive/MyDrive/All_SOIL.rar"
extract_dir = "/content/drive/MyDrive/SoilMoistureProject/data/unlabeled_images"
os.makedirs(extract_dir, exist_ok=True)
!unrar x -o+ "{rar_path}" "{extract_dir}/"

RAR_LABELED = "/content/drive/MyDrive/Soil_Labeled_Data.rar"
EXTRACT_LABELED = "/content/drive/MyDrive/SoilMoistureProject/data/labeled_data"
os.makedirs(EXTRACT_LABELED, exist_ok=True)
!unrar x -o+ "{RAR_LABELED}" "{EXTRACT_LABELED}/"

logging.basicConfig(filename='image_check.log', level=logging.INFO, format='%(asctime)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            logging.info(f"Image error: {path} ({str(e)})")
            try:
                if os.path.exists(path):
                    os.remove(path)
                    deleted_count += 1
                    logging.info(f"Deleted image: {path}")
            except (PermissionError, OSError) as e:
                logging.error(f"Cannot delete image {path}: {str(e)}")
    new_df = new_df.loc[valid_indices].reset_index(drop=True)
    logging.info(f"Deleted {deleted_count} labeled images. {len(new_df)} images remain.")
    return new_df

def preprocess_unlabeled_images(image_paths):
    new_image_paths = []
    deleted_count = 0
    for path in tqdm(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            img.verify()
            new_image_paths.append(path)
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            logging.info(f"Image error: {path} ({str(e)})")
            try:
                if os.path.exists(path):
                    os.remove(path)
                    deleted_count += 1
                    logging.info(f"Deleted image: {path}")
            except (PermissionError, OSError) as e:
                logging.error(f"Cannot delete image {path}: {str(e)}")
    logging.info(f"Deleted {deleted_count} unlabeled images. {len(new_image_paths)} images remain.")
    return new_image_paths

image_dir = "/content/drive/MyDrive/SoilMoistureProject/data/unlabeled_images"
image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
image_paths = preprocess_unlabeled_images(image_paths)

label_csv_path = "/content/drive/MyDrive/SoilMoistureProject/label_updated_colab.csv"
fallback_dir = "/content/drive/MyDrive/SoilMoistureProject/data/labeled_data/augmented_fallback"
os.makedirs(fallback_dir, exist_ok=True)
df = pd.read_csv(label_csv_path)
df = preprocess_images(df)

augment = transforms.ColorJitter(brightness=0.2, contrast=0.2)
replaced_count = 0
for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = row["path"]
    try:
        img = Image.open(path).convert("RGB")
        img.verify()
    except (UnidentifiedImageError, FileNotFoundError, OSError):
        folder = os.path.dirname(path)
        all_images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        good_images = [f for f in all_images if f != os.path.basename(path)]
        if not good_images:
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
        except Exception as e:
            continue

image_size = 224
transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labeled_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        except Exception as e:
            logging.error(f"Error reading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

class LabeledImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        try:
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
        except Exception as e:
            logging.error(f"Error reading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.df))
        humidity = torch.tensor([row['SM_0'] / 100, row['SM_20'] / 100], dtype=torch.float32)
        class_label = torch.tensor(row["moisture_class"], dtype=torch.long)
        return img, humidity, class_label

unlabeled_dataset = UnlabeledImageDataset(image_paths, transform)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
labeled_dataset = LabeledImageDataset(df, labeled_transform)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

class SoilNetDualHead(nn.Module):
    def __init__(self, num_classes=10, simclr_mode=False):
        super().__init__()
        self.simclr_mode = simclr_mode
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
        if self.simclr_mode:
            return x_img_feat
        x_light_feat = self.light_dense(x_light)
        x_concat = torch.cat([x_img_feat, x_light_feat], dim=1)
        reg_out = self.reg_head(x_concat)
        cls_out = self.cls_head(x_concat)
        return reg_out, cls_out

class Projector(nn.Module):
    def __init__(self, input_dim=1280, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

class OnlineLinearRegression(nn.Module):
    def __init__(self, input_dim=1280, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

class OnlineClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

def vicreg_loss(z1, z2, lambda_=25.0, mu=25.0, nu=1.0, epsilon=1e-4):
    invariance_loss = F.mse_loss(z1, z2)
    def variance_term(z):
        z_std = torch.sqrt(z.var(dim=0) + epsilon)
        return torch.mean(F.relu(1 - z_std))
    var_loss = variance_term(z1) + variance_term(z2)
    def covariance_term(z):
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.shape[0] - 1)
        off_diag = cov - torch.diag(cov.diag())
        return off_diag.pow(2).sum() / z.shape[1]
    cov_loss = covariance_term(z1) + covariance_term(z2)
    return lambda_ * invariance_loss + mu * var_loss + nu * cov_loss

model = SoilNetDualHead(num_classes=10, simclr_mode=True).to(device)
projector = Projector(input_dim=1280, proj_dim=128).to(device)
linear_reg = OnlineLinearRegression(input_dim=1280, output_dim=2).to(device)
classifier = OnlineClassifier(input_dim=1280, num_classes=10).to(device)

try:
    model.load_state_dict(torch.load('/content/drive/MyDrive/SoilNet_Checkpoints/Soinet_orginal.pth', map_location=device))
except FileNotFoundError:
    pass

optimizer_vicreg = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=1e-4)
optimizer_linear = torch.optim.Adam(linear_reg.parameters(), lr=1e-3)
optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)

checkpoint_dir = "/content/drive/MyDrive/SoilNet_Checkpoints/checkpoints_VicReg"
os.makedirs(checkpoint_dir, exist_ok=True)

def train_vicreg_with_mu(mu=25.0, num_epochs=50):
    vicreg_losses, mse_losses, rmse_losses, mae_losses, accuracy_scores, f1_scores = [], [], [], [], [], []
    labeled_iterator = iter(labeled_dataloader)
    for epoch in range(1, num_epochs + 1):
        model.train()
        projector.train()
        linear_reg.train()
        classifier.train()
        running_vicreg_loss = running_mse = running_mae = running_accuracy = running_f1 = 0.0
        num_batches = 0
        for img1, img2 in tqdm(unlabeled_dataloader, leave=False):
            img1, img2 = img1.to(device), img2.to(device)
            feat1 = model(img1, x_light=None)
            feat2 = model(img2, x_light=None)
            z1 = projector(feat1)
            z2 = projector(feat2)
            vicreg_loss_val = vicreg_loss(z1, z2, mu=mu)
            optimizer_vicreg.zero_grad()
            vicreg_loss_val.backward()
            optimizer_vicreg.step()
            try:
                labeled_img, humidity, class_label = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(labeled_dataloader)
                labeled_img, humidity, class_label = next(labeled_iterator)
            labeled_img, humidity, class_label = labeled_img.to(device), humidity.to(device), class_label.to(device)
            with torch.no_grad():
                feat = model(labeled_img, x_light=None)
            pred_humidity = linear_reg(feat)
            mse_loss = F.mse_loss(pred_humidity, humidity)
            optimizer_linear.zero_grad()
            mse_loss.backward()
            optimizer_linear.step()
            pred_logits = classifier(feat)
            cls_loss = F.cross_entropy(pred_logits, class_label)
            optimizer_classifier.zero_grad()
            cls_loss.backward()
            optimizer_classifier.step()
            pred_humidity_np = pred_humidity.detach().cpu().numpy() * 100
            humidity_np = humidity.detach().cpu().numpy() * 100
            mse = mean_squared_error(humidity_np, pred_humidity_np)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(humidity_np, pred_humidity_np)
            pred_classes = torch.argmax(pred_logits, dim=1).detach().cpu().numpy()
            true_classes = class_label.detach().cpu().numpy()
            accuracy = accuracy_score(true_classes, pred_classes)
            f1 = f1_score(true_classes, pred_classes, average='weighted')
            running_vicreg_loss += vicreg_loss_val.item()
            running_mse += mse
            running_mae += mae
            running_accuracy += accuracy
            running_f1 += f1
            num_batches += 1
        avg_vicreg_loss = running_vicreg_loss / num_batches
        avg_mse = running_mse / num_batches
        avg_rmse = np.sqrt(avg_mse)
        avg_mae = running_mae / num_batches
        avg_accuracy = running_accuracy / num_batches
        avg_f1 = running_f1 / num_batches
        vicreg_losses.append(avg_vicreg_loss)
        mse_losses.append(avg_mse)
        rmse_losses.append(avg_rmse)
        mae_losses.append(avg_mae)
        accuracy_scores.append(avg_accuracy)
        f1_scores.append(avg_f1)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'projector_state_dict': projector.state_dict(),
            'linear_reg_state_dict': linear_reg.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'vicreg_loss': avg_vicreg_loss,
            'mse_loss': avg_mse,
            'rmse_loss': avg_rmse,
            'mae_loss': avg_mae,
            'accuracy': avg_accuracy,
            'f1_score': avg_f1
        }
        checkpoint_path = os.path.join(checkpoint_dir, f'vicreg_mu_{mu}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
        final_model_path = os.path.join(checkpoint_dir, f'vicreg_model_final_mu_{mu}.pth')
        final_projector_path = os.path.join(checkpoint_dir, f'vicreg_projector_final_mu_{mu}.pth')
        final_linear_reg_path = os.path.join(checkpoint_dir, f'vicreg_linear_reg_final_mu_{mu}.pth')
        final_classifier_path = os.path.join(checkpoint_dir, f'vicreg_classifier_final_mu_{mu}.pth')
        torch.save(model.state_dict(), final_model_path)
        torch.save(projector.state_dict(), final_projector_path)
        torch.save(linear_reg.state_dict(), final_linear_reg_path)
        torch.save(classifier.state_dict(), final_classifier_path)
        logging.info(f"Saved final models (mu={mu}): {final_model_path}, {final_projector_path}, {final_linear_reg_path}, {final_classifier_path}")

for mu_val in [25.0]:
    train_vicreg_with_mu(mu=mu_val, num_epochs=50)