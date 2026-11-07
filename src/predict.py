"""
Predict a single handwritten digit from an image file.
Usage: python predict.py --image path/to/digit.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def preprocess_image(path):
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img, dtype="float32") / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = load_model(os.path.join(MODELS_DIR, "cnn_mnist.h5"))
    x = preprocess_image(args.image)

    proba = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(proba)); conf = float(proba[pred])

    print(f"Predicted digit: {pred}")
    print(f"Confidence: {conf:.4f}")
    top3 = np.argsort(proba)[::-1][:3]
    print(f"Top-3 digits: {top3.tolist()} "
          f"({[round(float(proba[i]),4) for i in top3]})")

    plt.figure(figsize=(3, 3))
    plt.imshow(x.squeeze(), cmap="gray")
    plt.title(f"Predicted: {pred} ({conf:.2%})")
    plt.axis("off")
    out_path = os.path.join(RESULTS_DIR, "prediction.png")
    plt.savefig(out_path); plt.close()
    print(f"Visual saved -> {out_path}")


if __name__ == "__main__":
    main()