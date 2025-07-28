# Author: Shad Hassan
# Date: July 27th, 2025

# This Python file is a library of functions needed for the main_barnacle_counter.py
# program to function

# READ: before running, make sure you pip install segment-anything and
# git+https://github.com/facebookresearch/segment-anything.git along with 
# open cv cv2 library, pyTorch, numpy, and matplotlib

# import models and packages needed
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
from matplotlib import pyplot as plt

# function resizes image. I made this in hopes that a better
# resolution on unseen_image1 would result in
# a correct count. However, the problem lies not in
# the resolution, but the size of the segments.
def load_and_resize(path, max_dim=2048):  # Increase to 2048 or even higher if memory allows
    image = cv2.imread(path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:  # Only resize if image is too large
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# function returns a range of areas for valid barnicle used in
# used ChatGPT for parts of these functions, sections are labeled
def extract_mask_features(mask):

  # convert the blue mask to gray scale, then invert the  mask so that
  # background is black and contours are white
  gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  inverted = cv2.bitwise_not(gray)

  # binarize image and fill object contours, used ChatGPT: how can I turn
  # contours into segments to train a SAM model
  _, binary = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
  kernel = np.ones((3, 3), np.uint8)
  closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

  # find and store valid segments
  contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  areas = [cv2.contourArea(c) for c in contours]

  return areas

# function takes a barnacle image and returns
# number of barnacles counted by the SAM model
def count_barnacles(img_path):
    image = load_and_resize(img_path, max_dim=2048)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # create mask using SAM
    masks = mask_generator.generate(image)

    # filter, count, and calculate average size of valid barnicles
    barnacle_masks = []
    barnacle_areas = []
    barnacle_x = []
    barnacle_y = []
    sum = 0
    for m in masks:
      if min_area < np.sum(m['segmentation']) < max_area:
        sum += np.sum(m['segmentation'])
        c_x = m['bbox'][0] + (m['bbox'][2] / 2)
        c_y = m['bbox'][1] + (m['bbox'][3] / 2)
        barnacle_masks.append(m)
        barnacle_areas.append(np.sum(m["segmentation"]))
        barnacle_x.append(c_x)
        barnacle_y.append(c_y)

    # calculate count and average
    count = len(barnacle_masks)
    average = sum / count


    # shows the mask created. can switch to showing the original image.
    plt.imshow(image)
    for m in barnacle_masks:
        plt.imshow(m['segmentation'], alpha=0.10)
    plt.title(f"Barnacle Count: {count} and Average Barnacle Pixel Size: {average}")
    plt.axis('off')
    plt.show()

    # plots the masks based on their centers and sizes
    areas = np.array(barnacle_areas)
    # Plot data
    plt.figure(figsize=(8, 8))
    plt.scatter( barnacle_x, barnacle_y,
    s=areas * 0.10,           # Scale point size based on area
    c='red', alpha=0.6, edgecolors='k')
    plt.title("Barnacle Centroids (Size and Area)")
    plt.gca().invert_yaxis()  # flip Y-axis to match image coordinates
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    return count

