# Written with assistance from Chat GPT

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap
from glob import glob, iglob
from collections import defaultdict

# ========== Step 1: Load all images ==========
def load_images(base_dir):
    image_dict = {}
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in '{base_dir}'")
        return image_dict

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        # img_paths = glob(os.path.join(path, "*.jpg"))
        img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG'):
            img_paths.extend(iglob(os.path.join(path, ext)))
        if not img_paths:
            print(f"No .jpg images found in '{path}'")
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_dict[(img_name, subdir)] = img_rgb
    return image_dict

def load_images_rgb(base_dir):
    rgb_dict = {}
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        img_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG'):
            img_paths.extend(glob(os.path.join(path, ext)))

        if not img_paths:
            print(f"No image files found in '{path}'")

        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_dict[(img_name, subdir)] = img_rgb

    return rgb_dict


# ========== Step 2: Convert images to HSV ==========
def rgb_to_hsv(image_dict):
    hsv_dict = {}
    for key, img_rgb in image_dict.items():
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv_dict[key] = hsv
    return hsv_dict

# ========== Step 3: Flatten and group pixels by subdirectory ==========
def group_pixels_by_subdir(hsv_dict):
    subdir_pixels = defaultdict(list)
    for (img_name, subdir), hsv_img in hsv_dict.items():
        pixels = hsv_img.reshape(-1, 3)  # flatten to (num_pixels, 3)
        subdir_pixels[subdir].append(pixels)
    
    # Concatenate all image pixel arrays for each subdir
    for subdir in subdir_pixels:
        subdir_pixels[subdir] = np.vstack(subdir_pixels[subdir])
    return subdir_pixels

def group_rgb_pixels_by_subdir(rgb_dict):
    subdir_pixels = defaultdict(list)
    for (img_name, subdir), rgb_img in rgb_dict.items():
        pixels = rgb_img.reshape(-1, 3)  # shape: (num_pixels, 3)
        subdir_pixels[subdir].append(pixels)
    
    for subdir in subdir_pixels:
        subdir_pixels[subdir] = np.vstack(subdir_pixels[subdir])
    return subdir_pixels

# ========== Step 4: Visualize with UMAP and PCA ==========
def plot_embeddings(subdir_pixels, max_points_per_class=2000):
    all_pixels = []
    labels = []

    for subdir, pixels in subdir_pixels.items():
        n = len(pixels)
        if n == 0:
            continue
        if n > max_points_per_class:
            idx = np.random.choice(n, max_points_per_class, replace=False)
            selected = pixels[idx]
        else:
            selected = pixels
        all_pixels.append(selected)
        labels.extend([subdir] * len(selected))

    if not all_pixels:
        print("No valid pixel data to plot.")
        return

    all_pixels = np.vstack(all_pixels)
    labels = np.array(labels)

    # PCA
    pca_ = PCA(n_components=2)
    pca_result = pca_.fit_transform(all_pixels)

    # UMAP
    umap_ = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_.fit_transform(all_pixels)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, ax=axes[0], s=3, palette='tab10')
    axes[0].set_title("PCA of HSV Pixels")

    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=labels, ax=axes[1], s=3, palette='tab10')
    axes[1].set_title("UMAP of HSV Pixels")

    plt.tight_layout()
    plt.show()

def plot_rgb_embeddings(subdir_pixels, max_points_per_class=2000):
    all_pixels = []
    labels = []

    for subdir, pixels in subdir_pixels.items():
        n = len(pixels)
        if n == 0:
            continue
        if n > max_points_per_class:
            idx = np.random.choice(n, max_points_per_class, replace=False)
            selected = pixels[idx]
        else:
            selected = pixels
        all_pixels.append(selected)
        labels.extend([subdir] * len(selected))

    if not all_pixels:
        print("No valid pixel data to plot.")
        return

    all_pixels = np.vstack(all_pixels)
    labels = np.array(labels)

    # PCA
    pca_ = PCA(n_components=2)
    pca_result = pca_.fit_transform(all_pixels)

    # UMAP
    umap_ = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_.fit_transform(all_pixels)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, ax=axes[0], s=3, palette='tab10')
    axes[0].set_title("PCA of RGB Pixels")

    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=labels, ax=axes[1], s=3, palette='tab10')
    axes[1].set_title("UMAP of RGB Pixels")

    plt.tight_layout()
    plt.show()


def plot_rgb_and_hsv_pca(rgb_pixels_by_subdir, hsv_pixels_by_subdir, max_points_per_class=2000):
    def sample_pixels(pixel_dict):
        sampled_pixels = []
        labels = []
        for subdir, pixels in pixel_dict.items():
            n = len(pixels)
            if n == 0:
                continue
            if n > max_points_per_class:
                idx = np.random.choice(n, max_points_per_class, replace=False)
                selected = pixels[idx]
            else:
                selected = pixels
            sampled_pixels.append(selected)
            labels.extend([subdir] * len(selected))
        return np.vstack(sampled_pixels), np.array(labels)
    
    # Sample RGB and HSV pixels equally
    rgb_data, labels = sample_pixels(rgb_pixels_by_subdir)
    hsv_data, _ = sample_pixels(hsv_pixels_by_subdir)  # same label order is guaranteed

    # PCA
    pca_rgb = PCA(n_components=2).fit_transform(rgb_data)
    pca_hsv = PCA(n_components=2).fit_transform(hsv_data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=pca_rgb[:, 0], y=pca_rgb[:, 1], hue=labels, ax=axes[0], s=3, palette='tab10')
    axes[0].set_title("PCA of RGB Pixels")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sns.scatterplot(x=pca_hsv[:, 0], y=pca_hsv[:, 1], hue=labels, ax=axes[1], s=3, palette='tab10')
    axes[1].set_title("PCA of HSV Pixels")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.13, 1))
    axes[0].legend().remove()
    axes[1].legend().remove()

    plt.tight_layout()
    plt.show()


# ========== Run Pipeline ==========
if __name__ == "__main__":
    base_dir = "dev/eos_labeled_crops"  # e.g., "./images"
    
    rgb_dict = load_images_rgb(base_dir)
    hsv_dict = rgb_to_hsv(rgb_dict)  # if you need HSV from RGB

    rgb_pixels = group_rgb_pixels_by_subdir(rgb_dict)
    hsv_pixels = group_pixels_by_subdir(hsv_dict)

    plot_rgb_and_hsv_pca(rgb_pixels, hsv_pixels, max_points_per_class=2000)
