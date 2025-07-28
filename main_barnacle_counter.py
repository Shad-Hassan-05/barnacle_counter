# Author: Shad Hassan
# Date: July 27th 2025

# This program uses funtions from functions.py and imports SAM model to 
# count barnacle on two unseen images utalizing segmentation. The program also
# visulaizes desity and size of barnacles detected. 

# READ: before running make sure you pip install segment-anything and
# git+https://github.com/facebookresearch/segment-anything.git along with 
# open cv cv2 library, pyTorch, numpy, and matplotlib

# import models and packages needed
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np
from functions import *

# specific SAM model. Using the smaller vit_b since larger models require more
# RAM.
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Loads model to device being used. Use a GPU if available.
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# get max and min from mask 1
train_mask = cv2.imread("mask1.png")
true_areas = extract_mask_features(train_mask)

# learn min/max barnacle area from mask data of mask1.
min_area_1 = np.percentile(true_areas, 10)
max_area_1 = np.percentile(true_areas, 90)

# get max and min using mask 2.
train_mask = cv2.imread("mask2.png")
true_areas = extract_mask_features(train_mask)

# learn min/max barnacle area from mask2.
min_area_2 = np.percentile(true_areas, 10)
max_area_2 = np.percentile(true_areas, 90)

# get the average min/max from training data
min_area = (min_area_1 + min_area_2) / 2
max_area = (max_area_1 + max_area_2) / 2

# Run on unseen image 1.
count1 = count_barnacles("unseen_img1.png")

# was running into memory issues so clearing GPU cache before running
# counter on second unseen image.
torch.cuda.empty_cache()
count2 = count_barnacles("unseen_img2.png")
