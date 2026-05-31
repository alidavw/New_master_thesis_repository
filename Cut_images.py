# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 14:34:32 2025

@author: edviv4919
"""

import numpy as np
import matplotlib.pyplot as plt
from czifile import CziFile



import os
from skimage.util import view_as_blocks

# # ==============================
# # 1. LOAD AND VIEW IMAGE
# # ==============================


# CZI file path
czi_path = r"your_file_path.czi"


with CziFile(czi_path) as czi:
    img = czi.asarray()

print("Original Shape of the CZI:", img.shape)
print("Number of dimensions:", img.ndim)

# ==============================
# EXTRACT CORRECT YX PLAN
# ==============================

img = img.astype(np.float32)

if img.ndim == 2:
    img2d = img

elif img.ndim == 3:
    # (C, Y, X) o (Z, Y, X)
    img2d = img[0]

elif img.ndim == 4:
    # (Z, C, Y, X) o (T, Y, X, C)
    img2d = img[0, 0]

elif img.ndim >= 5:
    # Caso típico CZI: (T, C, Z, Y, X)
    img2d = img[0, 0, 0, :, :]
   # img2d = img[0, 1, 0, :, :]  # otro canal
else:
    raise ValueError("Unrecognized CZI structure")

# Normalization for visualization
img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min())

plt.figure(figsize=(8, 8))
plt.imshow(img2d, cmap="gray")
plt.title("CZI Image – correct visualization")
plt.axis("off")
plt.show()

print("Shape final 2D:", img2d.shape)

# %%
# ===================================
# 2. DEFINE AREA OF INTEREST (ROI)
# ===================================

# Manually define the ROI (in pixels)
# y0:y1 -> rows
# x0:x1 -> columns
y0, y1 = 17389, 14982
x0, x1 = 12284, 18240

roi = img[y0:y1, x0:x1]

plt.figure(figsize=(6, 6))
plt.imshow(roi, cmap="gray")
plt.title("Selected region of interest (ROI)")
plt.axis("off")
plt.show()

print("Shape del ROI:", roi.shape)


# %%
# ===================================
# 3. CUT INTO PATCHES AND STORE
# ===================================

# Patch size
patch_size = 256

# Output folder
output_dir = r"your_output_directory"
os.makedirs(output_dir, exist_ok=True)

# Ensure that the ROI is divisible by 256
h, w = roi.shape
h_crop = (h // patch_size) * patch_size
w_crop = (w // patch_size) * patch_size
roi_cropped = roi[:h_crop, :w_crop]

# Create patches
patches = view_as_blocks(
    roi_cropped,
    block_shape=(patch_size, patch_size)
)

patch_id = 0
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j]
        patch_path = os.path.join(output_dir, f"patch_{patch_id:05d}.png")
        plt.imsave(patch_path, patch, cmap="gray")
        patch_id += 1

print(f"Saved patches: {patch_id}")
print("folder:", output_dir)
