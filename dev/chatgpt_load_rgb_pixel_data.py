import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict

def load_rgb_pixel_data(base_dir="dev/eos_labeled_crops", max_points_per_class=2000):
    pixel_data = defaultdict(list)
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            img_paths.extend(glob(os.path.join(base_dir, subdir, ext)))
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img_rgb.reshape(-1, 3)
            pixel_data[subdir].append(pixels)

    for subdir in pixel_data:
        pixels = np.vstack(pixel_data[subdir])
        if len(pixels) > max_points_per_class:
            idx = np.random.choice(len(pixels), max_points_per_class, replace=False)
            pixels = pixels[idx]
        pixel_data[subdir] = pixels

    return pixel_data
