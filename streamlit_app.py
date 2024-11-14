import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Digit Recognition App",
    page_icon="✏️",
    layout="centered"
)

# Title and description
st.title("Handwritten Digit Recognition")
st.write("Draw a digit or upload an image to predict the number!")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_model.h5')
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('L').resize((28, 28)))
    
    # Normalize and reshape
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image', width=200)
    with col2:
        st.write("## Prediction:")
        st.write(f"This looks like the digit: **{predicted_digit}**")
        st.write("Confidence scores:")
        st.bar_chart(prediction[0])

# Add information about the app
st.markdown("""
---
### About
This app uses a Convolutional Neural Network trained on the MNIST dataset to recognize handwritten digits.
""")