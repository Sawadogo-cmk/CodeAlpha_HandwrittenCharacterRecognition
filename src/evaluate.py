"""
Evaluate the trained CNN on the MNIST test set.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix — MNIST Test Set")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_roc(y_true_cat, y_proba, out_path):
    plt.figure(figsize=(8, 6))
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — One-vs-Rest")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def main():
    print("Loading data and model...")
    (_, _), (x_test, y_test) = load_data()
    model = load_model(os.path.join(MODELS_DIR, "cnn_mnist.h5"))

    print("Predicting...")
    y_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    accuracy = (y_pred == y_test).mean()

    report = classification_report(y_test, y_pred, digits=4)
    print("\n=== Classification Report ===")
    print(report)

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_confusion(y_test, y_pred, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_roc(to_categorical(y_test, 10), y_proba,
             os.path.join(RESULTS_DIR, "roc_curves.png"))
    print("Metrics + plots saved in results/")


if __name__ == "__main__":
    main()