import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load model
model = load_model("my_mnist_model.h5")

st.title("🧠 Handwritten Digit Classifier")
st.write("Upload a **28x28px** handwritten digit image (black digit on white background).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert background to match MNIST
    image = image.resize((28, 28))
    st.image(image, caption='Uploaded Image', width=150)

    img_array = np.array(image).reshape(1, 28, 28) / 255.0

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.success(f"✅ Prediction: **{predicted_digit}**")
