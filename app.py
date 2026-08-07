import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from notebook.class_names import CLASS_NAMES


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# Automatically detect input size (224 or 300)
IMG_SIZE = model.input_shape[1]


def predict(image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    pred_idx = np.argmax(prediction)
    confidence = prediction[pred_idx]

    return pred_idx, confidence


st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image and the AI model will identify the disease."
)


uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    pred_idx, confidence = predict(image)

    disease = CLASS_NAMES[pred_idx]
    disease = disease.replace("___", " → ")
    disease = disease.replace("_", " ")

    col1, col2 = st.columns([1, 1.5])

    with col1:

        st.image(
            image,
            width=250,
            caption="Uploaded Image"
        )

    with col2:

        st.subheader("Prediction")

        st.success(disease)

        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )


st.sidebar.title("🌿 About")

st.sidebar.write("""
### Plant Disease Detection

**Model:** TensorFlow / Keras

**Dataset:** PlantVillage (38 Classes)

Upload a plant leaf image to identify the disease using AI.
""")