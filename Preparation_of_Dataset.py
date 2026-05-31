# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:47:01 2025

@author: Pierre-Marie
"""

#%%
import numpy as np
import skimage
import matplotlib.pyplot as plt

Icolor = skimage.io.imread("your_picture_file.png")
plt.figure(figsize = (24,18))
plt.imshow(Icolor)
plt.show()

#%%

import cv2
import os

def split_and_save_image( output_dir, tile_size=256):
    # Create the output folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Loads the image in greyscale or color
    image = skimage.io.imread("your_file_path") # For grayscale: add cv2.IMREAD_GRAYSCALE
    height, width = image.shape[:2]

    count = 1
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Make sure the patch is the correct size
            if y + tile_size <= height and x + tile_size <= width:
                tile = image[y:y+tile_size, x:x+tile_size]
                filename = f"img{count}.png"
                cv2.imwrite(os.path.join(output_dir, filename), tile)
                count += 1

    print(f"{count-1} sub-images saved in {output_dir}")

# Utilisation
output_dir = r"your_folder_path"  # <-- folder where sub-images are saved
split_and_save_image( output_dir)

#%%

from pathlib import Path

def rename_images(folder: Path, start_index: int = 1) -> None:
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    images = sorted([p for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower() in image_exts])

    index = start_index
    for img in images:
        new_ext = img.suffix.lower()
        new_name = f"img{index}{new_ext}"
        new_path = img.with_name(new_name)
        if new_path.exists():
            raise FileExistsError(f"{new_path} already exists !")
        img.rename(new_path)
        print(f"{img.name}  →  {new_path.name}")
        index += 1

    print(f"🎉  {len(images)} images renamed.")

# Path to adapt
folder_path = Path(r"your_file_path")
start_index = 401
rename_images(folder_path, start_index)
