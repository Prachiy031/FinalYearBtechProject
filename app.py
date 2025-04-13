from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image 

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "C:/Users/Prachi Yadav/Documents/FinalYearProject/FinalYearProject/ImplementationFinalProject/Models/kidney_stone_detection_model_cnn.h5"  
model = tf.keras.models.load_model(MODEL_PATH)

# Route for uploading image
@app.route("/", methods=["GET", "POST"])
def upload_file():
    return render_template("index.html")

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    # Open and convert to RGB just to be safe
    img = Image.open(file.stream).convert("RGB")

    # Resize to match model input size (e.g., 224x224)
    img = img.resize((224, 224))

    # Convert to numpy array and normalize the image (if required)
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize to [0, 1] if the model was trained with normalized data

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Model handles resizing, scaling, etc.
    prediction = model.predict(img_array)[0]
    print("Prediction Value: ", prediction)  # Check prediction value here
    
    result = "Kidney Stone Detected" if prediction[0] > 0.5 else "No Kidney Stone"
    return render_template("results.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
