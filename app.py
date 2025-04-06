from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import tempfile
import gdown

app = Flask(__name__)

# Google Drive file URL and destination
MODEL_URL = "https://drive.google.com/file/d/1-0LxgX0aQB83eBIbORZ8HM38Kxyfv_8E/view?usp=drive_link"
MODEL_PATH = "model.h5"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)

# Load the model
model = load_model(MODEL_PATH, compile=False)
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    return {
        "probabilities": predictions[0].tolist(),
        "predicted_class": predicted_class
    }

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            result = predict_image(temp.name)

        os.remove(temp.name)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return "✅ Flask Brain Tumor Prediction API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
