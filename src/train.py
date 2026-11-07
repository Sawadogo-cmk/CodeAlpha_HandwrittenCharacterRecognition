"""
Build and train the CNN on MNIST.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks, optimizers

# allow import from preprocess when run as 'python train.py' from src/
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def main():
    print("[1/4] Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()

    print("[2/4] Building model...")
    model = build_model()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[3/4] Training...")
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=[early_stop],
        verbose=2,
    )

    print("[4/4] Saving...")
    model_path = os.path.join(MODELS_DIR, "cnn_mnist.h5")
    model.save(model_path)
    print(f"Model saved -> {model_path}")

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    plot_history(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    print("Training curves saved -> results/training_curves.png")


if __name__ == "__main__":
    main()