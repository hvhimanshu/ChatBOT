from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging
import os
from waitress import serve  # Production-ready server

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "pneumonia_classification_model.h5"

# Global variable for model (loaded once at startup)
model = None

def load_model():
    """Load model with memory optimization"""
    global model
    try:
        # Reduce TensorFlow log level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Configure TensorFlow to use less memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        
        model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        raise

# Load model when app starts
load_model()

def preprocess_image(image_bytes):
    """Safe image preprocessing with validation"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Validate image
        if img.format not in ['JPEG', 'PNG']:
            raise ValueError("Only JPEG/PNG images supported")
            
        img = img.convert('L').resize((128, 128))
        img_array = np.array(img)
        
        # Handle grayscale conversion
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
            
        return np.expand_dims(img_array / 255.0, axis=0)
    except Exception as e:
        app.logger.error(f"Image processing error: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks"""
    return jsonify({"status": "healthy", "model_loaded": bool(model)})

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file'].read()
        
        # Validate file size (max 5MB)
        if len(file) > 5 * 1024 * 1024:
            return jsonify({"error": "File too large (max 5MB)"}), 400
            
        img = preprocess_image(file)
        prediction = float(model.predict(img, verbose=0)[0][0])  # Disable TF logging
        
        return jsonify({
            "result": "Pneumonia detected" if prediction > 0.5 else "Normal",
            "confidence": round(prediction, 4),
            "model": os.path.basename(MODEL_PATH)
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    # Use Waitress for production (better than Flask's dev server)
    serve(app, host='0.0.0.0', port=port, threads=4)