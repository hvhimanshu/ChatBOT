from huggingface_hub import hf_hub_download
import os
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import requests

app = Flask(__name__)

MODEL_PATH = "pneumonia_classification_model.h5"
REPO_ID = "hvhimanshu/pneumonia_chatbot"
FILENAME = "pneumonia_classification_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Hugging Face Hub...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=".",
        local_dir_use_symlinks=False
    )
    print("Model download successfully.")
# Load your trained CNN model
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Match model input size
    img = np.array(img)
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    img = img / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file'].read()
    img = preprocess_image(file)
    prediction = model.predict(img)
    result = "Pneumonia detected" if prediction > 0.5 else "Normal"
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
