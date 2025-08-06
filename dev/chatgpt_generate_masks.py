import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Load model ---
target_class = "nucleus"
model_path = f"dev/pixel_classifier_{target_class}.joblib"
model = joblib.load(model_path)

# --- Parameters ---
input_dir = "dev/eos"
output_dir = "dev/eos_masks"
thresholds = [0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]

os.makedirs(output_dir, exist_ok=True)

# --- Process each image ---
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_dir, img_name)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Flatten pixels and predict
    pixels = img_rgb.reshape(-1, 3)
    probs = model.predict_proba(pixels)[:, 1]

    for t in thresholds:
        mask = (probs > t).astype(np.uint8).reshape(h, w) * 255

        # Save mask
        mask_name = f"{os.path.splitext(img_name)[0]}_mask_thresh_{int(t*1000)}.png"
        cv2.imwrite(os.path.join(output_dir, mask_name), mask)

    print(f"Processed {img_name}")
