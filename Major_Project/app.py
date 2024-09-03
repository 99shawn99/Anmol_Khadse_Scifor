# -*- coding: utf-8 -*-


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model("/content/drive/MyDrive/face_mask_detector_model.keras")

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to 224x224 pixels
    image = img_to_array(image)  # Convert the image to a numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image to the range [0, 1]
    return image

# Streamlit app title
st.title("Face Mask Detection")

# File uploader to allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(preprocessed_image)

    # Display the result
    if prediction[0][0] > 0.5:
        st.error("No Mask Detected")
    else:
        st.success("Mask Detected")
