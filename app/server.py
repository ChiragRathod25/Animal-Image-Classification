from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
import pickle
import os

app = Flask(__name__, template_folder='templates')

IMG_SIZE = 128

# Load Model & Label Encoder 
model = tf.keras.models.load_model('model/cnn_model.h5')
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_index = np.argmax(pred)
    label = le.inverse_transform([class_index])[0]

    return jsonify({
        'predicted_category': label,
        'confidence': float(np.max(pred))
    })

if __name__ == '__main__':
    app.run(debug=True)
