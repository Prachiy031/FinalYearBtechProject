from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image 
import os

app = Flask(__name__, template_folder="projectUI/templates",static_folder="projectUI/static")

# Load the trained model
MODEL_PATH = "Models/kidney_stone_detection_model_cnn.h5"  
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    print("Prediction Value: ", prediction)
    
    result = "Kidney Stone Detected" if prediction[0] > 0.5 else "No Kidney Stone"
    return render_template("results.html", prediction=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
