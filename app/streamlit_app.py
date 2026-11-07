"""
Streamlit app for Handwritten Digit Recognition.
Run from project root: streamlit run app/streamlit_app.py
"""
import os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_mnist.h5")

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
)


@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


def preprocess_mnist_style(img_array):
    """Crop, resize to 20x20, center in 28x28 — MNIST-style."""
    if img_array.ndim == 3:
        img = img_array[:, :, 0]
    else:
        img = img_array

    img = img.astype("float32")

    # Invert if dark digit on light background
    if img.mean() > 127:
        img = 255.0 - img

    mask = img > 50
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img[rmin:rmax + 1, cmin:cmax + 1]

    pil = Image.fromarray(cropped.astype("uint8")).resize((20, 20), Image.LANCZOS)
    resized = np.array(pil, dtype="float32")

    out = np.zeros((28, 28), dtype="float32")
    out[4:24, 4:24] = resized
    return out / 255.0


def predict(model, img_28x28):
    x = img_28x28.reshape(1, 28, 28, 1)
    proba = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(proba))
    return pred, float(proba[pred]), proba


def show_prediction(pred, conf, proba, processed_img):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Vue du CNN")
        st.image(processed_img, caption="28×28 normalisé", width=200, clamp=True)

    with col2:
        st.subheader("Prédiction")
        st.markdown(f"## 🔢 Chiffre : **{pred}**")
        st.markdown(f"### 🎯 Confiance : **{conf:.2%}**")
        st.progress(conf)

    st.markdown("### Probabilités par classe")
    fig, ax = plt.subplots(figsize=(9, 3))
    colors = ["#ff6b6b" if i == pred else "#4dabf7" for i in range(10)]
    ax.bar(range(10), proba, color=colors)
    ax.set_xticks(range(10))
    ax.set_xlabel("Chiffre")
    ax.set_ylabel("Probabilité")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("✍️ Reconnaissance de chiffres manuscrits")
    st.markdown(
        "**CodeAlpha ML Internship** — Sawadogo Noé\n\n"
        "Modèle CNN entraîné sur MNIST — **99,31 % d'accuracy** sur 10 000 images de test."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Modèle introuvable : `{MODEL_PATH}`\n\n"
            "Entraînez d'abord le modèle : `cd src && python train.py`"
        )
        return

    model = load_trained_model()

    mode = st.radio(
        "Mode d'entrée :",
        ["✏️ Dessiner", "📁 Uploader une image"],
        horizontal=True,
    )

    if mode == "✏️ Dessiner":
        if not CANVAS_AVAILABLE:
            st.error(
                "Canvas indisponible. Installez : `pip install streamlit-drawable-canvas`\n\n"
                "Ou utilisez le mode Upload."
            )
            return

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=22,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            return_image_data=True,      # ← correction ici
            key="canvas",
        )

        if canvas_result.image_data is not None:
            processed = preprocess_mnist_style(canvas_result.image_data[:, :, :3])
            if processed is None:
                st.info("Dessinez un chiffre dans le canvas.")
            else:
                pred, conf, proba = predict(model, processed)
                show_prediction(pred, conf, proba, processed)

    else:
        uploaded = st.file_uploader(
            "Choisissez une image (PNG/JPG)", type=["png", "jpg", "jpeg"]
        )
        if uploaded is not None:
            pil = Image.open(uploaded).convert("L")
            processed = preprocess_mnist_style(np.array(pil, dtype="float32"))
            if processed is None:
                st.warning("Impossible de détecter un chiffre. Essayez une image plus nette.")
            else:
                pred, conf, proba = predict(model, processed)
                show_prediction(pred, conf, proba, processed)


if __name__ == "__main__":
    main()