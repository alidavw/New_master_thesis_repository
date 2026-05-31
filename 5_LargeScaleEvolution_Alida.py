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
model.load_state_dict(torch.load(r"D:\Frank\U_Net_IS\PythonPM\Model_Texture_412Img.pth",map_location='cpu'))
model.eval()


#%% Functions to apply the model


transform = A.Compose([

    A.Normalize(mean=(0.559, 0.477, 0.491), std=(0.135, 0.121, 0.125)),
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
    

#   input_tensor = transform(image=sous_image)["image"].unsqueeze(0)  # (1, 3, 256, 256)

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
    
    # Concatenar → (4, H, W) #(Texture IA)
    input_tensor = torch.cat([rgb_tensor, lbp_tensor], dim=0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (256, 256)
    
        plt.imshow(pred_mask)
        plt.show()
    return pred_mask


def divisionssimg(img, patch_h=256, patch_w=256, overlap_h=0, overlap_w=0):
    #  patch_h=4476, patch_w=5189, overlap_h=64, overlap_w=64
    """
    Split the image into overlapping rectangular patches (sliding window).
    """
    patches = []
    positions = []

    stride_h = patch_h - overlap_h
    stride_w = patch_w - overlap_w

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
            y = i * stride_h   # row index
            x = j * stride_w   # column index
            # Cortar parche y aplicarlo al modelo
            sub_img = img_padded[y:y+patch_h, x:x+patch_w]
            sub_img = sub_img[
                :max(0, img.shape[0] - y),
                :max(0, img.shape[1] - x)
]
            patches.append(sub_img)
            positions.append((y, x))

    return patches, positions, (img.shape[0], img.shape[1])



# def reassemble_grid(patches, positions, final_shape,
#                     patch_h=256, patch_w=256,
#                     overlap_h=0, overlap_w=0):
#     """
#     Reconstruct the full image from patches using blending
#     """
#     H, W = final_shape
#     full_pred = np.zeros((H, W), dtype=np.float32)
#     weight_mask = np.zeros((H, W), dtype=np.float32)

#     for patch, (x, y) in zip(patches, positions):
#         h, w = patch.shape
#         full_pred[x:x+h, y:y+w] += patch
#         weight_mask[x:x+h, y:y+w] += 1

#     # Avoid division by zero
#     weight_mask[weight_mask == 0] = 1
#     blended = full_pred / weight_mask

#     return blended.astype(np.uint8)

#%% Preparation of the list of images

czi = r"D:/Alida/SRB_C/1_Base_lighter.czi"

images = []

with czifile.CziFile(czi) as czi_obj:
    data = czi_obj.asarray()
    print("stack shape", data.shape)

    test_img = data[0, 0, :, :, :]
    print("test_img shape:", test_img.shape)

    for i in tqdm(range(data.shape[1]), desc="Images loading"):
        image = data[0, 0, :, :, :]
        

        #  Adjust shape to (H, W, 3)
        if image.ndim == 5:          # ej. (1,1,H,W,3)
            image = image[0, 0, :, :, :]
        elif image.ndim == 4:        # ej. (1,H,W,3)
            image = image[0, :, :, :]
        elif image.ndim == 2:        # grey scale
            image = np.stack([image]*3, axis=-1)   # Duplicate channel → RGB
        elif image.ndim == 3 and image.shape[-1] != 3:  
            #Rare case: ej. (H,W,1)
            image = np.repeat(image, 3, axis=-1)

        print(f"Frame {i+1} → Shape after adjustment: {image.shape}")

        if image.size == 0:
            print(f"⚠️ Frame {i+1} empty → omitted")
            continue
       
        # --- Crop image ---
        image = image[712:4478, 504:5484]

        # --- CRITICAL CHECK ---
        if image.size == 0:
            print("⚠️ Empty image after crop → frame skipped")
            continue

        print("Image shape after crop:", image.shape)
        
        # image is uint16 here
        image = image.astype(np.float32)

        # robust contrast stretch (microscopy standard)
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1 + 1e-8)

        # convert to uint8 [0–255]
        image_uint8 = (image * 255).astype(np.uint8)

        # --- Safe normalization ---
        if image.dtype != np.uint8:
            ptp = image.ptp()
            if ptp == 0:
                image = np.zeros_like(image, dtype=np.uint8)
        else:
            image = ((image - image.min()) / (ptp + 1e-8) * 255).astype(np.uint8)

            
        #Work image show
        plt.imshow(image)
        plt.title("Preview cropped image")
        plt.axis("off")
        plt.show()  



        images.append(image_uint8)

        
#%% Application of the model on the list

#%% Application of the model on the list


# =========================
# Parameters
# =========================
PATCH_H = 1024
PATCH_W = 1024
OVERLAP = 256
NUM_CLASSES = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Preprocess function
# =========================
def preprocess(img):
    if img.ndim == 2:               # grayscale
        img = np.stack([img]*3, axis=-1)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
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
            tensor = preprocess(patch).unsqueeze(0).to(device)  # (1,3,H,W)
            logits = model(tensor)                               # (1,C,H,W)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C,H,W)

        patch_prob = np.moveaxis(probs, 0, -1)  # (H,W,C)
        
        ph, pw = patch_prob.shape[:2]
        w = weight[:ph, :pw]
        
        assert patch_prob.shape[:2] == w.shape, \
        f"Mismatch: patch {patch_prob.shape}, weight {w.shape}"
        
        prob_acc[y:y+ph, x:x+pw, :] += patch_prob * w[..., None]
        weight_acc[y:y+ph, x:x+pw] += w

    valid = weight_acc > 0
    prob_acc[valid] /= weight_acc[valid, None]
    prob_acc[~valid] = 0   # seguridad
    
    background = prob_acc[..., 0]
    rock       = prob_acc[..., 2]
    calcite    = prob_acc[..., 4]
    
    full_mask = np.argmax(prob_acc, axis=-1)

    # enforce physics
#    full_mask[full_mask == 1] = 0  # gas → background
#    full_mask[full_mask == 3] = 0  # sulfate → background
    
    # refine calcite (overwrite where confident)
#    t_calc = np.percentile(calcite, 97)
#    refined_calcite = (
#    (calcite >= t_calc) &
#    (calcite > background) &
#    (calcite > rock)
#    )
#    full_mask[refined_calcite] = 4


    # --- ALWAYS create a mask ---
    # full_mask = np.zeros(calcite.shape, dtype=np.uint8)
    
    # # --- ROCK / GRAINS ---
    # rock_mask = (
    #     (rock > background) &
    #     (rock > calcite)
    #     )
    # full_mask[rock_mask] = 2

    # if not np.all(calcite == 0):

    #     # percentile-based adaptive threshold
    #     t_calc = np.percentile(calcite, 97)

    #     calcite_mask = (
    #         (calcite >= t_calc) &
    #         (calcite > background) &
    #         (calcite > rock)
    #         )

    #     full_mask[calcite_mask] = 4


    classimages.append(full_mask)

print(f"Frame {i}: appended mask, unique labels =", np.unique(full_mask))

    
#%% Surface evaluation and graphics

Index = np.arange(1, 2, 1)
#Index = np.arange(len(classimages))

#Evolution = [[],[],[],[],[]]
Evolution = [[] for _ in range(NUM_CLASSES)]

print("len(classimages):", len(classimages))
print("len(Evolution):", len(Evolution))

for mask in classimages:
    for j in range(NUM_CLASSES):
        Evolution[j].append(np.count_nonzero(mask == j))
        pixel_count = np.count_nonzero(mask == j)
        Evolution[j].append(pixel_count)

# for i in range(len(Index)):
#     for j in range(classimages):
#         Evolution[j].append(np.count_nonzero(classimages[i] == j))

pixel_size_um = 1.1
pixel_area_um2 = pixel_size_um ** 2  # = 1.21 µm²

# for i in range(len(Index)):          # frames
#     for j in [1, 3, 4]:              # solo H₂, Sulfite y Calcite
#             pixel_count = np.count_nonzero(classimages[i] == j)
#             area_um2 = pixel_count * pixel_area_um2
#             Evolution[j].append(area_um2)


for j in [1, 3, 4]:  # H2, Sulfate, Calcite
    Evolution[j] = [px * pixel_area_um2 for px in Evolution[j]]

#  Direction where save the mask of every framework
output_dir = r"D:/Alida/Test Results"

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
for i, mask in enumerate(classimages):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color

        
    plt.figure(figsize=(6,4))
    plt.imshow(image)
    plt.title("Cropped input image")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.imshow(color_mask)
    plt.title("Predicted mask")
    plt.axis("off")
    plt.show()    

    mask_img = Image.fromarray(color_mask)
    mask_img.save(f"{output_dir}/mask_colored_frame_today.png")


print("MASK SAVED CORRECTLY")
        
class_names = ["Background", "H2", "Rock", "Sulfate", "Calcite"]
selected_classes = [1, 3, 4]


Index = np.arange(len(Evolution[j])) #Added by Alida 26.01.26 as a checkpoint


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

print("Mean prob per class:", probs.mean(axis=(1,2)))
    
#%%

# cmap = ListedColormap([
#     (153/255, 51/255, 102/255),        # 0 - Fond
#     (255/255, 255/255, 0/255),    # 1 - H2
#     (0/255, 255/255, 255/255),    # 2 - Rock
#     (0/255, 255/255, 0/255),      # 3 - Sulfite
#     (255/255, 0/255, 0/255),      # 4 - Calcite
# ])



# colored_img = cm.jet(classimages[0]*64 / 255.0)

# colored_img = (colored_img[:,:,:3] * 255).astype(np.uint8)
# iio.imwrite("first_round_CaSO4.png", colored_img)
