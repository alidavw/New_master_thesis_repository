# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:13:19 2025

@author: Resfys
"""

from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def compute_mean_std(image_folder):
    means = []
    stds = []
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".png"))]
    
    for filename in tqdm(image_files):
        path = os.path.join(image_folder, filename)
        img = Image.open(path).convert("RGB")
        img_array = np.array(img) / 255.0
        
        means.append(np.mean(img_array, axis = (0,1)))
        stds.append(np.std(img_array, axis = (0,1)))
        
    mean = np.mean(means, axis = 0)
    std = np.mean(stds, axis = 0)
    
    
    return mean.tolist(), std.tolist()

folder = r"your_folder_path"

mean, std = compute_mean_std(folder)

print("Mean :" , mean)
print("Std :" , std)