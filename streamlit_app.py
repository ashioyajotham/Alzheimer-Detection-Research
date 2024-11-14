import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Alzheimer's Detection",
    page_icon="🧠",
    layout="centered"
)

# Title and description
st.title("Alzheimer's Disease Detection")
st.write("Upload a brain MRI scan to detect the stage of Alzheimer's Disease")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('alzheimers_cnn_model.h5')
    return model

try:
    model = load_model()
except:
    st.error("Failed to load model. Please check if the model file exists.")
    st.stop()

# Define classes
classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Scan', width=300)
        
        # Preprocess the image
        img_array = np.array(image.resize((176, 176)))
        # Convert to grayscale if image is RGB or RGBA
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # Normalize and reshape
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 176, 176)
        
        # Make prediction
        with st.spinner('Analyzing the MRI scan...'):
            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
        
        # Display results
        st.write("## Analysis Results")
        st.write(f"**Diagnosis:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        # Show probability distribution
        st.write("### Probability Distribution")
        prob_dict = {class_name: float(prob) * 100 for class_name, prob in zip(classes, prediction[0])}
        st.bar_chart(prob_dict)
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.stop()

# Add information about the app
st.markdown("""
---
### About
This application uses deep learning to analyze brain MRI scans and detect different stages of Alzheimer's Disease. 
The model has been trained on a dataset of brain MRI scans and can classify them into four categories:
- Non Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

### Important Note
This tool is for educational purposes only and should not be used as a substitute for professional medical diagnosis.
Please consult with healthcare professionals for medical advice.
""")

# Add sidebar with additional information
with st.sidebar:
    st.header("How to Use")
    st.write("""
    1. Upload a brain MRI scan image
    2. Wait for the analysis
    3. View the results and probability distribution
    
    For best results:
    - Use clear, high-quality MRI scans
    - Images should be properly oriented
    - Ensure the scan shows the full brain structure
    """)