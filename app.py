# from flask import Flask, request, jsonify
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io
# import logging
# import os
# from waitress import serve  # Production-ready server

# app = Flask(__name__)
# logging.basicConfig(level=logging.INFO)

# MODEL_URL = "https://drive.google.com/uc?export=download&id=1BCV-dXoruc2roqbL_xSuxx5qQxHiWj5b"
# MODEL_PATH = "pneumonia_classification_model.h5"

# # Global variable for model (loaded once at startup)
# model = None

# def load_model():
#     """Load model with memory optimization"""
#     global model
#     try:
#         # Reduce TensorFlow log level
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
#         # Configure TensorFlow to use less memory
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             tf.config.experimental.set_memory_growth(gpus[0], True)
        
#         model = tf.keras.models.load_model(MODEL_PATH)
#         app.logger.info("Model loaded successfully")
#     except Exception as e:
#         app.logger.error(f"Model loading failed: {str(e)}")
#         raise

# # Load model when app starts
# load_model()

# def preprocess_image(image_bytes):
#     """Safe image preprocessing with validation"""
#     try:
#         img = Image.open(io.BytesIO(image_bytes))
        
#         # Validate image
#         if img.format not in ['JPEG', 'PNG']:
#             raise ValueError("Only JPEG/PNG images supported")
            
#         img = img.convert('L').resize((128, 128))
#         img_array = np.array(img)
        
#         # Handle grayscale conversion
#         if len(img_array.shape) == 2:
#             img_array = np.stack((img_array,)*3, axis=-1)
            
#         return np.expand_dims(img_array / 255.0, axis=0)
#     except Exception as e:
#         app.logger.error(f"Image processing error: {str(e)}")
#         raise

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Endpoint for health checks"""
#     return jsonify({"status": "healthy", "model_loaded": bool(model)})

# @app.route("/predict", methods=["POST"])
# def predict():
#     """Main prediction endpoint"""
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
        
#         file = request.files['file'].read()
        
#         # Validate file size (max 5MB)
#         if len(file) > 5 * 1024 * 1024:
#             return jsonify({"error": "File too large (max 5MB)"}), 400
            
#         img = preprocess_image(file)
#         prediction = float(model.predict(img, verbose=0)[0][0])  # Disable TF logging
        
#         return jsonify({
#             "result": "Pneumonia detected" if prediction > 0.5 else "Normal",
#             "confidence": round(prediction, 4),
#             "model": os.path.basename(MODEL_PATH)
#         })
        
#     except Exception as e:
#         app.logger.error(f"Prediction error: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
    
#     # Use Waitress for production (better than Flask's dev server)
#     serve(app, host='0.0.0.0', port=port, threads=4)

from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging
import os
import requests
from waitress import serve

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1BCV-dXoruc2roqbL_xSuxx5qQxHiWj5b"
MODEL_PATH = "pneumonia_classification_model.h5"

# Global variable for model
model = None

def download_model():
    """Download model from Google Drive if not present"""
    if not os.path.exists(MODEL_PATH):
        app.logger.info("Downloading model from Google Drive...")
        try:
            session = requests.Session()
            response = session.get(MODEL_URL, stream=True, timeout=60)
            
            # Handle large file downloads
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            app.logger.info("Model downloaded successfully")
        except Exception as e:
            app.logger.error(f"Model download failed: {str(e)}")
            os.remove(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
            raise

def load_model():
    """Load model with memory optimization"""
    global model
    try:
        # Download model first if needed
        download_model()
        
        # TensorFlow configuration
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce logging
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        
        # Load model with custom objects if needed
        model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        raise

# Load model at startup
load_model()

def preprocess_image(image_bytes):
    """Safe image preprocessing with validation"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.format not in ['JPEG', 'PNG']:
            raise ValueError("Only JPEG/PNG images supported")
            
        img = img.convert('L').resize((128, 128))
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
            
        return np.expand_dims(img_array / 255.0, axis=0)
    except Exception as e:
        app.logger.error(f"Image processing error: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": bool(model),
        "model_size": f"{os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB" if model else None
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file'].read()
        
        if len(file) > 5 * 1024 * 1024:
            return jsonify({"error": "File too large (max 5MB)"}), 400
            
        img = preprocess_image(file)
        prediction = float(model.predict(img, verbose=0)[0][0])
        
        return jsonify({
            "result": "Pneumonia detected" if prediction > 0.5 else "Normal",
            "confidence": round(prediction, 4),
            "model": os.path.basename(MODEL_PATH)
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host='0.0.0.0', port=port, threads=4)