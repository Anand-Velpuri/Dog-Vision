import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io
import base64

# === CONFIG === #
MODEL_PATH = "./model/20250515-05211747286492-full-images-mobilenetv2-Adam.keras"
LABELS_PATH = "./data/labels.csv"
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
IMG_SIZE = 224

app = Flask(__name__)

# === LOAD LABELS === #
labels_csv = pd.read_csv(LABELS_PATH)
labels = labels_csv.sort_values("id")["breed"].to_numpy()
unique_breeds = np.unique(labels)

# === TF HUB WRAPPER === #
def hub_layer_fn(x):
    return hub.KerasLayer(MODEL_URL, trainable=False)(x)

# === LOAD MODEL === #
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'hub_layer_fn': hub_layer_fn,
        'KerasLayer': hub.KerasLayer
    }
)
print("Model loaded.")

# === UTILITY FUNCTIONS === #
def process_image_pil(img):
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_breed(image):
    image_array = process_image_pil(image)
    preds = model.predict(image_array)[0]
    top_idx = np.argmax(preds)
    confidence = float(preds[top_idx])
    return unique_breeds[top_idx], confidence

# === ROUTES === #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'})

        # Remove the data:image/jpeg;base64 prefix if present
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        breed, confidence = predict_breed(image)

        return jsonify({
            'breed': breed,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# === RUN APP === #
if __name__ == '__main__':
    app.run()
 
