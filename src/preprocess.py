"""
Data loading and preprocessing for MNIST.
"""
import numpy as np
from tensorflow.keras.datasets import mnist


def load_data():
    """
    Load MNIST dataset and normalize to [0, 1].
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_tr, y_tr), (x_te, y_te) = load_data()
    print(f"Train: {x_tr.shape}, labels: {y_tr.shape}")
    print(f"Test : {x_te.shape}, labels: {y_te.shape}")
    print(f"Pixel range: [{x_tr.min():.2f}, {x_tr.max():.2f}]")