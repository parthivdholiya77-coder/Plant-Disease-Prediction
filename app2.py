import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from notebook.class_names import CLASS_NAMES


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_efficientnetb3.keras")
    return model


model = load_model()

# Automatically detect input size (224 or 300)
IMG_SIZE = model.input_shape[1]


def predict(image):

    image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image).astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    pred_idx = np.argmax(prediction)

    confidence = prediction[pred_idx]

    return pred_idx, confidence



st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a leaf image and the AI model will predict the plant disease."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Create two columns
    col1, col2 = st.columns([1, 1.5])

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            width=250
        )

    with col2:

        pred_idx, confidence = predict(image)

        disease = CLASS_NAMES[pred_idx]
        disease = disease.replace("___", " → ")
        disease = disease.replace("_", " ")

        st.subheader("Prediction")

        st.success(f"🌿 {disease}")

        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )


st.sidebar.title("About")

st.sidebar.write("""
### Plant Disease Detection

This application detects plant diseases using a deep learning model based on **EfficientNetB3**.

**Framework:** TensorFlow/Keras

**Dataset:** PlantVillage (38 Classes)

Developed using Streamlit.
""")