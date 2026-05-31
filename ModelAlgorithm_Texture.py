# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 17:02:09 2026

@author: Resfys
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:08:03 2025

@author: Resfys
"""


# %% 1 - Importation of the librairies and base parameters

import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


import segmentation_models_pytorch as smp

from skimage.feature import local_binary_pattern

import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------
# ---------------------- CONFIGURATION ------------------------
# -------------------------------------------------------------
# Set these paths if your data lives elsewhere
DATA_DIR = Path(r"your_folder_path")            # working directory ("." by default)
IMAGES_DIR = DATA_DIR / "Images"
MASKS_RGB_DIR = DATA_DIR / "Mask"
CLEANED_MASKS_DIR = DATA_DIR / "Cleaned_mask"

# Pre‑processing flags
CONVERT_MASKS = True            # set False if Cleaned_mask already exists
SPLIT_DATASET = True            # set False if *_train / *_val folders already exist
TRAIN_RATIO = 0.8
SEED = 42

# Training hyper‑parameters
BATCH_SIZE = 16
LR = 5e-4
NUM_EPOCHS = 100
PATIENCE = 50               # early stopping patience
ALPHA_CE = 0.5                # CE weight in combined loss



# -------------------------------------------------------------
# ---------------------- OUTPUT FILES --------------------------
# -------------------------------------------------------------
# >>> EDIT ONLY THIS PART TO CHOOSE WHERE TO SAVE EVERYTHING <<<
OUTPUT_DIR = Path(r"your_output_folder")
RUN_NAME = "Model_Texture_424Img"

MODEL_SAVE_PATH = OUTPUT_DIR / f"{RUN_NAME}.pth"
EXCEL_SAVE_PATH = OUTPUT_DIR / f"{RUN_NAME}_training_metrics.xlsx"
PLOT_LOSS_PATH = OUTPUT_DIR / f"{RUN_NAME}_loss_curve.png"
PLOT_DICE_PATH = OUTPUT_DIR / f"{RUN_NAME}_dice_per_class.png"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)



# %% 2 - Transformation of the RGB-Masks to Class-Masks

COLOR2CLASS = {
    (153,   51,   102): 0,  # Background
    (255, 255,   0): 1,  # H2
    (0,   255, 255): 2,  # Rock
    (0,   255,   0): 3,  # Sulfite
    (255,   0,   0): 4,  # Calcite
    (255,   127,   39): 5,  # Biofilm
}

UNKNOWN = 255
NUM_CLASSES = len(COLOR2CLASS)
CLASS_NAMES = ["Background", "H2", "Rock", "Sulfite", "Calcite", "Biofilm"]


# --- 1) Conversion and recognition of the unknown --- Transform an RGB image into a mask of class, and setting every parasite 
#pixels at 255 because we don´t know there class for the moment

def rgb_to_class_with_unknown(rgb_mask: Image.Image) -> np.ndarray:
    
    arr = np.array(rgb_mask) #tranform the rgb image into a numpy array [height, width, nb of canals]
    
    h, w, _ = arr.shape #recuperate the height and width
    
    class_mask = np.full((h, w), UNKNOWN, dtype=np.uint8) #creates a matrix of pixels at 255 with the dimensions of the masks
    
    for rgb, cls in COLOR2CLASS.items():     #For each class defined in COLOR2CLASS (exemple: rbg :(0,0,0) correspond to cls : 1)
        match = np.all(arr == rgb, axis=-1) #ask for each canal to correspond to the rgb color of the class 
        class_mask[match] = cls #All the pixels corresponding to the good color are considered as part of the class cls
        
    return class_mask, arr   # We obtain the class img and the original image


# --- 2) Major smoothing (5×5 window) ---replace the 255 pixels by the dominating class in a 5x5 window --> first treatment

def majority_fill(mask: np.ndarray, window=5, thresh=0.5) -> np.ndarray:
    
    #the function resolve chose which class for each unknown pixels by looking at its nearest neightbour
    def resolve(window_vals):
        
        window_vals = window_vals[window_vals != UNKNOWN] #We ignore the 255 pixels
        
        if window_vals.size == 0: #if the window contain only 255 pixels, we´ll see later
            return UNKNOWN
        
        counts = np.bincount(window_vals.astype(np.int32), minlength=NUM_CLASSES) #For each window it count the number of time each class is present
        maj = counts.max()
        
        if maj / window_vals.size >= thresh: #If a class is dominent enough
            return counts.argmax()
        
        return UNKNOWN #Else we keep the pixel at 255
    
    result = generic_filter(mask, resolve, size=window, mode="nearest")

    return result.astype(np.uint8)

# --- 3) Chose the class of the unknown by their distance in the color spectrum ---> Second treatment after majority_fill


def fill_by_color_distance(mask, rgb_arr):
    
    unknown_mask = (mask == UNKNOWN) #We locate every residual 255 pixels

    if not np.any(unknown_mask):#If everything is already good we don´t use this function
        return mask
    
    rows, cols = np.where(unknown_mask)
    flat_rgb = rgb_arr[rows, cols]
    
    ref_colors = np.array(list(COLOR2CLASS.keys())) #Numpy array of the class color
    ref_classes = np.array(list(COLOR2CLASS.values())) #Numpy array of the class
    

    dists = np.linalg.norm(flat_rgb[:, None, :] - ref_colors[None, :, :], axis=2) #Calculation of the euclidian distance between 
    #parasite pixels and each class color
    
    best_match = ref_classes[np.argmin(dists, axis = 1)]
    
    mask[rows, cols] = best_match #Take the smaller distance
    
    return mask

# --- 4) Complete pipeline ---> Application of the functions defined before
def clean_rgb_mask(rgb_img: Image.Image) -> np.ndarray:
    
    class_mask, rgb_arr = rgb_to_class_with_unknown(rgb_img)
    
    class_mask = majority_fill(class_mask, window=5, thresh=0.5)
    
    class_mask = fill_by_color_distance(class_mask, rgb_arr)
    
    # 4b) Petit filtre « mode » de 3×3 pour lisser les points isolés
    class_mask = majority_fill(class_mask, window=3, thresh=0.34)
    
    if np.any(class_mask == UNKNOWN):
        rows, cols = np.where(class_mask == UNKNOWN)
        flat_rgb = rgb_arr[rows, cols]
        ref_colors = np.array(list(COLOR2CLASS.keys()))
        ref_classes = np.array(list(COLOR2CLASS.values()))
        dists = np.linalg.norm(flat_rgb[: , None, :] - ref_colors[None, :, :], axis = 2)
        best_match = ref_classes[np.argmin(dists, axis = 1)]
        class_mask[rows, cols] = best_match
                
    
    return class_mask.astype(np.uint8)

def compute_lbp_percentile(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    p1, p99 = np.percentile(lbp, (1, 99))
    lbp = np.clip((lbp - p1) / (p99 - p1 + 1e-8), 0, 1)
    return lbp.astype(np.float32)


if CONVERT_MASKS:
    print("[1/3] cleaning masks")
    CLEANED_MASKS_DIR.mkdir(exist_ok = True, parents = True)
    for file in os.listdir(MASKS_RGB_DIR):
        if file.lower().endswith(".png"):
            rgb_img = Image.open(MASKS_RGB_DIR / file).convert("RGB")
            cleaned_mask = clean_rgb_mask(rgb_img)
            out_path = CLEANED_MASKS_DIR / file
            
            print(f"{file} : dtype = {cleaned_mask.dtype}, unique values = {np.unique(cleaned_mask)}")
            #print(f"Shape = {cleaned_mask.shape}")
            #print(f"Any Nan = {np.isnan(cleaned_mask).any()}")
            Image.fromarray(cleaned_mask).save(out_path)
            
    print(f"[1/3] Cleaned masks saved to: {CLEANED_MASKS_DIR}")
    
else: 
    print("[1/3] Skipping mask cleaning (flag CONVERT_MASKS = False)")


# %% 3 - Data splitting

TRAIN_IMG_DIR = DATA_DIR / "Images_train"
VAL_IMG_DIR = DATA_DIR / "Images_val"
TRAIN_MASK_DIR = DATA_DIR / "Masques_cleaned_train"
VAL_MASK_DIR = DATA_DIR / "Masques_cleaned_val"


def split_dataset(images_dir: Path, masks_dir: Path):
    random.seed(SEED)
    filenames = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])
    random.shuffle(filenames)

    split_idx = int(len(filenames) * TRAIN_RATIO)
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR]:
        d.mkdir(exist_ok=True, parents=True)

    for f in train_files:
        shutil.copy(images_dir / f, TRAIN_IMG_DIR / f)
        shutil.copy(masks_dir / f, TRAIN_MASK_DIR / f)

    for f in val_files:
        shutil.copy(images_dir / f, VAL_IMG_DIR / f)
        shutil.copy(masks_dir / f, VAL_MASK_DIR / f)

    print(f"Dataset split → {len(train_files)} train | {len(val_files)} val")


if SPLIT_DATASET:
    print("[2/3] Splitting dataset …")
    split_dataset(IMAGES_DIR, CLEANED_MASKS_DIR)
else:
    print("[2/3] Skipping dataset split (flag SPLIT_DATASET=False)")
    
    
# %% 4 -  Albumentations transforms & Dataset class

#the goal is to prepare the images before they enter in the U-Net model
#We`ll define the transformations in order to avoid overlearning, to adapt the image sizes to the model and normalize the 
#RGB canals to ResNet50

#It´ll create a class that gives the pairs (image, mask) ready to be use


#We define a gaussian tranformation for the training

class CustomGaussianNoise(A.ImageOnlyTransform):
    def __init__(self, mean = 0.0 , std = 10.0, always_apply = False, p=0.5):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std
        
    def apply(self, image, **params):
        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        
        return np.clip(noisy, 0 , 255).astype(np.uint8)

#Here, we do some modifications to the dataset in order to have a resilient model
train_transform = A.Compose([
    A.Resize(256, 256), #resize if needed
    A.HorizontalFlip(p=0.5), #50% of the images are fliped
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p = 0.5),
    A.ColorJitter(brightness= 0.5, contrast= 0.5, saturation= 0.5 , hue= 0.2, p = 0.8),
    A.RandomBrightnessContrast(p=0.6), #20% of the images have changement in their birghtness/contrast
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), #some zoom and rotations on 50% of the images
    A.Normalize(mean=(0.559, 0.477, 0.491), std=(0.135, 0.121, 0.125)),
    A.GaussianBlur(kernel_size = 3, sigma = (0.1 , 0.7) ,p = 0.2),
    CustomGaussianNoise(mean =0 , std =15, p=0.2),
    #A.RandomResizedCrop(256,256, scale = (0.8, 1.0)),
    #A.RandomCrop(height = 224, width = 224, p =0.5),
    ToTensorV2(), #Convert the images and masks into PyTorch tensor 
])


#No random transformations here because we want a stable and repeatable evaluation 
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5530443241628572, 0.47017516525331043, 0.48376285187456325), std=(0.13426006720699943, 0.12038174593484816, 0.12349954462710282)),
    ToTensorV2(),
])

#This is oriented object programation, self is the dataset considered as a SegmentationDataset class
#Self.something refered to the attribute something of the Dataset
#This class will be use right after to build the train_loader and val_loader
#Those datasets gives automaticaly the mini-batch to the model fot the training


class SegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    #Tells the number of images
    def __len__(self):
        return len(self.images)

    #Here we charge the RGB image and the mask
    def __getitem__(self, idx):
        img_path = self.image_dir / self.images[idx]
        mask_path = self.mask_dir / self.images[idx]

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        
        
        # Añadido por IA
        gray = np.mean(img, axis=2).astype(np.uint8)
        lbp = compute_lbp_percentile(gray)
        
        #We apply the transformation
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image'] #The same tranformation are applied to the image and the associated mask
            lbp = np.array(
                Image.fromarray((lbp * 255).astype(np.uint8)).resize(
                    (img.shape[2], img.shape[1]), Image.BILINEAR
                    )
                ) / 255.0
            lbp = torch.from_numpy(lbp).unsqueeze(0).float()
            img = torch.cat([img, lbp], dim=0)
            
            mask = augmented['mask']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
            mask = torch.from_numpy(mask)

        return img, mask.long()
    
def check_class_presence(dataset, name="dataset"):
    import numpy as np
    from collections import Counter

    counter = Counter()

    for _, mask in dataset:
        unique, counts = np.unique(mask.numpy(), return_counts=True)
        counter.update(dict(zip(unique, counts)))

    print(f"\nClass pixel counts in {name}:")
    for cls in range(NUM_CLASSES):
        print(f"  Class {cls}: {counter.get(cls, 0)} pixels")
        
        

# %% 5 - Dataloaders

train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

# Check TRAIN and VAL
check_class_presence(train_dataset, name="TRAIN")
check_class_presence(val_dataset, name="VAL")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[3/3] Dataloaders ready → {len(train_dataset)} train | {len(val_dataset)} val")


# %% 6 - Loss functions

#This part guide the learning by telling how much the model is wrong at each step
#The loss function measure the difference between the model prediction and the corresppnding mask


#Diceloss is a specialized loss function for image segmentation that predict the difference between the predicted mask and 
#the real one usingthe Dice Coef
#Very adapted for rare objects segmentation

class DiceLoss(nn.Module):    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        #inputs : (batch, classes, height, width) --> the type of images used
        #targets : true mask containing the shapes with the true class numbers
        
        inputs = F.softmax(inputs, dim=1) #each pixel has a probability per class
        
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        
        cardinality = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1 - dice.mean()


#the goal here is to combine the stability and the rigor for each pixel of the cross-entropy and the power of rare object 
#recognition of the diceloss

#We are adding different weights to each class, to favor rare claa (here sulfite, calcite)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
CE_WEIGHTS = torch.tensor([1.5, 1.2, 1.2, 2.5, 4, 4], dtype = torch.float).to(device)

#Then we can obtain a loss function efficient for big areas andlittle objects
#We can modify the coef ALPHA_CE to adapt the ratio between diceloss and cross entropy loss
def combined_loss(preds, masks, alpha=ALPHA_CE):
    ce = nn.CrossEntropyLoss(weight = CE_WEIGHTS)(preds, masks)
    dice = DiceLoss()(preds, masks)
    return alpha * ce + (1 - alpha) * dice

# %% 7 - Model, optimiser, scheduler

#We define here the neural network that we want to train, how it learns and its rythm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#here we use the Unet neural network which predict a class for each pixel. The resnet50 model is a model that already recognize 
#some general structures 
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=4,
    classes=len(COLOR2CLASS),
).to(device)

# Load ImageNet weights except for the first layer
state = torch.load(
    r"C:\Users\Resfys\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth",
    map_location="cpu"
)

model_dict = model.encoder.state_dict()

# Filtrar conv1.weight
state = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}

model_dict.update(state)
model.encoder.load_state_dict(model_dict)


# #This part guarantees that it´ll work on every machine
# state = torch.load(r"C:\Users\Resfys\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth", map_location= "cpu")
# model.encoder.load_state_dict(state, strict=False)
# print("ImageNet weight injected into the encoder")

#The optimizer fits the weight in function of the gradient loss
#lr is the learning rate that can be adapted
optimizer = optim.Adam(model.parameters(), lr=LR)

#The scheduler adapt automaticaly the learning rate if the validation loss stop its upgrading
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# %% 8 - Training loop

#That´s where the Unet model learn how to segment image after image, mask after mask


def validate(model_, loader_):
    model_.eval()
    loss_total = 0.0
    with torch.no_grad():
        for imgs, masks in loader_:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model_(imgs)
            loss = combined_loss(preds, masks)
            loss_total += loss.item()
    return loss_total / len(loader_)

def compute_dice_per_class(model_, loader_, num_classes):
    model_.eval()
    dices = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for imgs, masks in loader_:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model_(imgs)
            pred_labels = torch.argmax(preds, dim = 1)
            
            for cls in range(num_classes):
                pred_cls = (pred_labels == cls)
                true_cls = (masks == cls)
                
                inter = (pred_cls & true_cls).sum().float()
                union = pred_cls.sum() + true_cls.sum()
                
                dice = (2. * inter + 1e-5)/(union + 1e-5)
                
                dices[cls].append(dice.item())
                
    return [np.mean(cls_scores) for cls_scores in dices]
            

best_val_loss = float("inf") #keep the best validation loss that has been reach
trigger_times = 0 #keep how many epochs without improvement


#Graph the metrics druing epochs
train_losses = []
val_losses = []
dice_history = [[] for _ in range(NUM_CLASSES)]

#Epoch loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    #Mini-batch loop
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    val_loss = validate(model, val_loader)
    dice_scores = compute_dice_per_class(model, val_loader, num_classes = NUM_CLASSES)
    scheduler.step(val_loss)
    
    
    # Save history
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    for cls in range(NUM_CLASSES):
        dice_history[cls].append(dice_scores[cls])


    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train: {epoch_loss:.4f} | Val: {val_loss:.4f}")
    print("Dice par classe: ", ["{:.3f}".format(d) for d in dice_scores])

    # keep the best model
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("  🔥 New best model saved")
        
    #early stopping
    else:
        trigger_times += 1
        if trigger_times >= PATIENCE:
            print("  ⏹ Early stopping")
            break

print("Training complete → best val loss:", best_val_loss)


# %% 9 - Save plots and Excel with training history

epochs_ran = len(train_losses)
epoch_index = np.arange(1, epochs_ran + 1)

# -------------------- Plot: Train Loss vs Val Loss --------------------
plt.figure(figsize=(8, 5))
plt.plot(epoch_index, train_losses, marker='o', label="Train Loss")
plt.plot(epoch_index, val_losses, marker='o', label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_LOSS_PATH, dpi=300)
plt.show()

# -------------------- Plot: Dice Score per class --------------------
plt.figure(figsize=(10, 6))
for i in range(NUM_CLASSES):
    plt.plot(epoch_index, dice_history[i], marker='o', label=f"{CLASS_NAMES[i]}")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Dice Score vs Epoch for Each Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DICE_PATH, dpi=300)
plt.show()

# -------------------- Save metrics to Excel --------------------
metrics_data = {
    "Epoch": epoch_index,
    "Train_Loss": train_losses,
    "Val_Loss": val_losses,
}

for i in range(NUM_CLASSES):
    metrics_data[f"Dice_{CLASS_NAMES[i]}"] = dice_history[i]

df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_excel(EXCEL_SAVE_PATH, index=False, engine="openpyxl")

print(f"Best model saved in: {MODEL_SAVE_PATH}")
print(f"Loss plot saved in: {PLOT_LOSS_PATH}")
print(f"Dice plot saved in: {PLOT_DICE_PATH}")
print(f"Excel file saved in: {EXCEL_SAVE_PATH}")