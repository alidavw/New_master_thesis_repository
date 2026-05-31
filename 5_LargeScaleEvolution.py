# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 09:22:07 2025

@author: Resfys
"""

#%% Librairies importation

import czifile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import segmentation_models_pytorch as smp
import skimage
from matplotlib.colors import ListedColormap
from skimage import io
import matplotlib.cm as cm
import imageio.v3 as iio
from tqdm import tqdm

#%% Loading the model

NUM_CLASSES = 6  # Names of the classes you used during training

# Rebuild the architecture
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=4,
    classes=NUM_CLASSES
)

# Load the trained weights
model.load_state_dict(torch.load(r"you_folder_path.pth",map_location='cpu'))
model.eval()


#%% Functions to apply the model


transform = A.Compose([

    A.Normalize(mean=(0.5530443241628572, 0.47017516525331043, 0.48376285187456325), std=(0.13426006720699943, 0.12038174593484816, 0.12349954462710282)),
    ToTensorV2()
])

from skimage.feature import local_binary_pattern     #Texture IA, now we have 4 chanels R,G,B,Texture)

def compute_lbp_percentile(gray, P=8, R=1):          #(Texture IA)
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    p1, p99 = np.percentile(lbp, (1, 99))
    lbp = np.clip((lbp - p1) / (p99 - p1 + 1e-8), 0, 1)
    return lbp.astype(np.float32)


def applyzone(x, y, width, height, img):
    sous_image = img[x:(x+ width), y:(y + height)]
    

    # input_tensor = transform(image=sous_image)["image"].unsqueeze(0)  # (1, 3, 256, 256)
    
    
    # RGB transform (same as training) #(Texture IA)
    rgb_tensor = transform(image=sous_image)["image"]
    
    # LBP (same as training) #(Texture IA)
    gray = np.mean(sous_image, axis=2).astype(np.uint8)
    lbp = compute_lbp_percentile(gray)    
    
    lbp = np.array(
        Image.fromarray((lbp * 255).astype(np.uint8)).resize(
            (rgb_tensor.shape[2], rgb_tensor.shape[1]), Image.BILINEAR
            )
        ) / 255.0
    
    lbp_tensor = torch.from_numpy(lbp).unsqueeze(0).float() #(Texture IA)
    
    # Concatenate → (4, H, W) #(Texture IA)
    input_tensor = torch.cat([rgb_tensor, lbp_tensor], dim=0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()  # (C,H,W)
        
        plt.imshow(np.argmax(probs, axis=0))
        plt.title("Patch prediction (argmax, debug only)")
        plt.show()
    return probs


def divisionssimg(img, patch_h=256, patch_w=256, overlap_h=0, overlap_w=0):
    #  patch_h=4476, patch_w=5189, overlap_h=64, overlap_w=64
    """
    Split the image into overlapping rectangular patches (sliding window).
    """
    patches = []
    positions = []

    stride_h = patch_h - overlap_h
    stride_w = patch_w - overlap_w

    #rows = (img.shape[0] - overlap_h) // stride_h
    #cols = (img.shape[1] - overlap_w) // stride_w    Commented out by Alida 27.01.26

    img_padded = np.pad(                              #Added by Alida 27.01.26
        img,
        ((0, patch_h), (0, patch_w), (0, 0)),
        mode="reflect"
        )
    
    H_p, W_p = img_padded.shape[:2] #Added by Alida 27.01.26

    rows = (H_p - overlap_h) // stride_h #Added by Alida 27.01.26
    cols = (W_p - overlap_w) // stride_w

    for i in range(rows):
        for j in range(cols):
            x = i * stride_h
            y = j * stride_w
            # Cortar parche y aplicarlo al modelo
            sub_pred = applyzone(x, y, patch_h, patch_w, img)
            patches.append(sub_pred)
            positions.append((x, y))

    return patches, positions, (img.shape[0], img.shape[1])



def reassemble_grid(patches, positions, final_shape,
                    patch_h=256, patch_w=256,
                    overlap_h=0, overlap_w=0):
    """
    Reconstruct the full image from patches using blending
    """
    H, W = final_shape
    full_pred = np.zeros((H, W), dtype=np.float32)
    weight_mask = np.zeros((H, W), dtype=np.float32)

    for patch, (x, y) in zip(patches, positions):
        h, w = patch.shape
        full_pred[x:x+h, y:y+w] += patch
        weight_mask[x:x+h, y:y+w] += 1

    # Avoid division by zero
    weight_mask[weight_mask == 0] = 1
    blended = full_pred / weight_mask

    return blended.astype(np.uint8)

#%% Preparation of the list of images

czi = r"your_file_path.czi"


# =========================================================
# 1) LOAD THE .CZI ONLY ONCE
# =========================================================
with czifile.CziFile(czi) as czi_obj:
    data = czi_obj.asarray()

print("stack shape:", data.shape)


#%% =========================================================
# 2) FUNCTION TO EXTRACT A ROI WITHOUT RELOADING THE FILE
# =========================================================
def extract_roi_from_loaded_czi(data, center_x, center_y, width_zen, height_zen,
                                orig_w=24512, orig_h=20422):
    images = []

    for i in tqdm(range(data.shape[1]), desc="Images loading"):
        image = data[0, i, :, :, :]

        # Adjust shape to (H, W, 3)
        if image.ndim == 5:          # e.g. (1,1,H,W,3)
            image = image[0, 0, :, :, :]
        elif image.ndim == 4:        # e.g. (1,H,W,3)
            image = image[0, :, :, :]
        elif image.ndim == 2:        # grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] != 3:  
            # Rare case: e.g. (H,W,1)
            image = np.repeat(image, 3, axis=-1)

        print(f"Frame {i+1} → Shape after adjustment: {image.shape}")

        if image.size == 0:
            print(f"⚠️ Frame {i+1} empty → omitted")
            continue

        print("Full image shape:", image.shape)

        # Convert center → corners in ZEN coordinates
        x1_zen = center_x - width_zen // 2
        x2_zen = center_x + width_zen // 2
        y1_zen = center_y - height_zen // 2
        y2_zen = center_y + height_zen // 2

        # Scale to current loaded image size
        h, w = image.shape[:2]
        sx = w / orig_w
        sy = h / orig_h

        # Convert coordinates
        x1 = int(x1_zen * sx)
        x2 = int(x2_zen * sx)
        y1 = int(y1_zen * sy)
        y2 = int(y2_zen * sy)

        # Crop image
        roi = image[y1:y2, x1:x2]

        # Normalization
        if roi.dtype != np.uint8:
            roi = ((roi - roi.min()) / (roi.ptp() + 1e-8) * 255).astype(np.uint8)

        plt.imshow(roi)
        plt.title(f"frame {i+1}")
        plt.axis("off")
        plt.show()

        images.append(roi)

    return images

images = extract_roi_from_loaded_czi(
    data,
    center_x=12179,
    center_y=10247,
    width_zen=22005,
    height_zen=17703
)


#%% Application of the model on the list


# =========================
# Parameters
# =========================
PATCH_H = 4476
PATCH_W = 5189
NUM_CLASSES = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Preprocess function
# =========================
def preprocess(img):
    # Make sure RGB
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    # --- RGB ---
    rgb = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    # --- LBP (same as training) ---
    gray = np.mean(img, axis=2).astype(np.uint8)
    lbp = compute_lbp_percentile(gray)

    lbp = torch.from_numpy(lbp).unsqueeze(0).float()

    # --- Concatenation ---
    t = torch.cat([rgb, lbp], dim=0)   # (4,H,W)

    return t

# =========================
# Gaussian weight function
# =========================
def gaussian_weight(h, w, sigma=0.25):
    y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return g / g.max()

# ⬇⬇ Explicit Gaussian weight
weight = gaussian_weight(PATCH_H, PATCH_W, sigma=0.25)

# =========================
# Application of the model
# =========================
classimages = []



for i in tqdm(range(len(images)), desc="Images Treatment"):

    img = images[i]

    patches, positions, final_shape = divisionssimg(
        img,
        patch_h=PATCH_H,
        patch_w=PATCH_W,
        overlap_h = PATCH_H // 2,
        overlap_w = PATCH_W // 2
    )

    print(f"Frame {i}: number of patches =", len(patches))

    H, W = final_shape[:2]

    prob_acc = np.zeros((H, W, NUM_CLASSES), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)

    for patch, (y, x) in zip(patches, positions):

        with torch.no_grad():
            
            patch_prob = np.moveaxis(patch, 0, -1)  # patch is already (C,H,W) (Added as a replacement by Alida)

        
        ph, pw = patch_prob.shape[:2]
        prob_acc[y:y+ph, x:x+pw, :] += patch_prob * weight[:ph, :pw, None]
        weight_acc[y:y+ph, x:x+pw] += weight[:ph, :pw]

    valid = weight_acc > 0
    prob_acc[valid] /= weight_acc[valid, None]
    prob_acc[~valid] = 0   # seguridad
    
    full_mask = np.argmax(prob_acc, axis=-1)

    classimages.append(full_mask)

    
#%% Surface evaluation and graphics

Index = np.arange(1, 35, 1)

Evolution = [[] for _ in range(NUM_CLASSES)]

for i in range(len(Index)):
    for j in range(NUM_CLASSES):
        Evolution[j].append(np.count_nonzero(classimages[i] == j))
        
pixel_size_um = 1.1
pixel_area_um2 = pixel_size_um ** 2  # = 1.21 µm²

for i in range(len(Index)):          # frames
    for j in [1, 3, 4]:              # Just H₂, Sulfite y Calcite
            pixel_count = np.count_nonzero(classimages[i] == j)
            area_um2 = pixel_count * pixel_area_um2
            Evolution[j].append(area_um2)

#  Direction where save the mask of every framework
output_dir = r"your_output_folder_path"


# Colormap: class index → RGB color
CLASS_COLORS = {
    0: (153, 51, 102), # Background
    1: (255, 255, 0), # H₂
    2: (0, 255, 255), # Rock
    3: (0, 255, 0), # Sulfate
    4: (255, 0, 0), # Calcite
    5: (255, 127, 39), # Biofilm
}

#  Save each colored mask
for i in range(len(Index)):
    mask = classimages[i] # Class mask (values 0–4)
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color

    mask_img = Image.fromarray(color_mask)
    mask_img.save(f"{output_dir}/1_CaSO4_H2_SRB_Rep2-01_mask_{i+1:03d}.png")


print("MASK SAVED CORRECTLY")
        
class_names = ["Background", "H2", "Rock", "Sulfate", "Calcite", "Biofilm"]
selected_classes = [1, 3, 4]

# print(len(Index)) to check its value before changing it

Index = np.arange(len(Evolution[j])) #Added by Alida 26.01.26 by help from ChatGPT as 
#ValueError: x and y must have same first dimension, but have shapes (1,) and (2,) 

# print(len(Index)) to check it if changes, which it does (from 1 to 2)
  
plt.figure(figsize=(7, 5))
for j in selected_classes: 
    plt.plot(Index, Evolution[j], 
label=class_names[j])
    
plt.xlabel ("Frame over time")
plt.ylabel ("Area_um2")
plt.title ("Temporal evolution")
plt.legend ()
plt.grid (True)
plt.show()

for j in selected_classes: print(f"{class_names[j]}: {Evolution[j]}")