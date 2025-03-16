from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
import cv2
import pickle
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Google Drive File ID (Replace with your actual file ID)
file_id = "1tTOokKtqrzTiGx_7ehg2aBNpEvzSXy2S"
model_path = "model.pkl"

# Function to download model from Google Drive if not available
def download_model():
    if not os.path.exists(model_path):
        print("Downloading model.pkl from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    else:
        print("Model already exists.")

# Download model if not present
download_model()

# Load the trained ML model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None  # Handle case where image cannot be read
    img = cv2.resize(img, (256, 256))
    img = img.reshape((1, 256, 256, 3)) / 255.0  # Normalize the image
    return img

# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    # Secure and save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file.save(file_path)

    # Preprocess and predict
    img = preprocess_image(file_path)
    if img is None:
        return render_template('index.html', prediction="Error: Invalid Image", image_url=None)

    prediction = model.predict(img)[0][0]

    # Interpret the result
    result = "Reusable" if prediction > 0.5 else "Not Reusable"

    return render_template('index.html', prediction=result, image_url=file_path)

if __name__ == '__main__':
    app.run(debug=True)
