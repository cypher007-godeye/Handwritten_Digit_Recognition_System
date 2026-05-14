from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import base64
from tensorflow import keras

# --- Core Logic from MNIST.py ---
def preprocess_image_array(img_array):
    """
    Preprocesses a numpy array image for MNIST digit classification.
    Expects img_array to be a grayscale image.
    """
    # Resize to 28x28
    img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to range [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0

    # Add channel and batch dimensions (1, 28, 28, 1)
    img_final = np.expand_dims(img_normalized, axis=(0, -1))

    return img_final

class DigitPredictor:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, processed_image):
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence_score = np.max(prediction) * 100
        return int(predicted_digit), float(confidence_score)

# --- Flask App ---
app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join("mnist_model", "mnist_classifier.keras")
predictor = DigitPredictor(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Get base64 image string
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)

        # Convert bytes to numpy array using OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Preprocess and Predict
        processed_img = preprocess_image_array(img)
        digit, confidence = predictor.predict(processed_img)

        return jsonify({
            'digit': digit,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
