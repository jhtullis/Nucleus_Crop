import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Load your pixel data ---
# rgb_pixels_by_subdir: Dict[subdir_name] = np.ndarray of shape (n_pixels, 3)
from chatgpt_load_rgb_pixel_data import load_rgb_pixel_data  # <-- you'll define this

# --- Define target class ---
target_class = "nucleus"
max_points_per_class = 2000

# --- Load data ---
rgb_pixels_by_subdir = load_rgb_pixel_data(max_points_per_class=max_points_per_class)

X = []
y = []

for subdir, pixels in rgb_pixels_by_subdir.items():
    label = 1 if subdir == target_class else 0
    X.append(pixels)
    y.extend([label] * len(pixels))

X = np.vstack(X)
y = np.array(y)

# --- Train classifier ---
model = LogisticRegression(class_weight={0: 1.0, 1: 5.0}, solver="liblinear", max_iter=200)
model.fit(X, y)

# --- Save model ---
joblib.dump(model, f"pixel_classifier_{target_class}.joblib")
print("Model trained and saved.")
