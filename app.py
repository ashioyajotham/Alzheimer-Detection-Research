# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('alzheimers_cnn_model.h5')

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    class_names = ['Mild Demented', 'Moderate Demented', 'Non-Demented', 'Very Mild Demented']
    result = {
        'prediction': class_names[np.argmax(prediction)],
        'confidence': float(np.max(prediction)),
        'scores': {name: float(score) for name, score in zip(class_names, prediction[0])}
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)