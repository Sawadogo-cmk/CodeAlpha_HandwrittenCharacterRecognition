"""
Export a few MNIST test images as PNG files into samples/.
"""
import os
import numpy as np
from PIL import Image
from preprocess import load_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

(_, _), (x_test, y_test) = load_data()

# Prendre 5 images avec des labels variés
indices = [0, 1, 2, 3, 4]
for i, idx in enumerate(indices):
    img = (x_test[idx].squeeze() * 255).astype("uint8")
    # MNIST est blanc sur noir → on inverse pour un rendu "naturel" noir sur blanc
    img_inv = 255 - img
    label = y_test[idx]
    out_path = os.path.join(SAMPLES_DIR, f"digit_{label}_{i}.png")
    Image.fromarray(img_inv).resize((280, 280), Image.LANCZOS).save(out_path)
    print(f"Saved {out_path} (label={label})")

print(f"\n{len(indices)} images sauvegardées dans samples/")