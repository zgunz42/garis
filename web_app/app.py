# web_app/app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image, ImageOps
import os
import csv
from datetime import datetime
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('../models/model.h5')

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the input image for model prediction.

    Args:
        image (PIL.Image.Image): The input image to be preprocessed.

    Returns:
        numpy.ndarray: The preprocessed image for model prediction.
    """
    # handle png images
    if image.mode == 'P':
        image = image.convert('RGB')
        

    # Convert to grayscale 
    image = ImageOps.grayscale(image)
    
    # Convert to 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    # and resize in one step
    image = np.array(image)
    
    # Invert the colors (make it black on white)
    image = 255 - image
    
    # Normalize the image pixel values
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)

    return image

def save_image_and_data(image, processed_image, expected_output, predicted_output):
    # Create the directory if it doesn't exist
    raw_data_dir = '../data/raw/'
    processed_data_dir = '../data/processed/'
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Create a unique filename based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    raw_image_filename = f'{raw_data_dir}{expected_output}_{timestamp}.png'
    proccessed_image_filename = f'{processed_data_dir}{expected_output}_{timestamp}.png'
    image.save(raw_image_filename)

    image_array = np.clip(processed_image, 0.0, 1.0)
    
    # Scale the pixel values to 0-255 (uint8)
    image_array = (image_array * 255).astype(np.uint8)
    
    # Reshape the image array to be 2-dimensional
    image_array = image_array.reshape((28, 28))
    
    # Create a PIL Image from the array
    prcImg = Image.fromarray(image_array)
    
    # Save the image to file
    prcImg.save(proccessed_image_filename)
    
    # Save data to CSV
    csv_filename = f'{raw_data_dir}data.csv'
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'expected_output', 'predicted_output', 'image_filename'])
        writer.writerow([timestamp, expected_output, predicted_output, raw_image_filename])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    expected_output = data['expected_output']
    image_data = image_data.split(',')[1]  # Remove the data URL part
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction[0])
    
    print(f"Model prediction: {prediction}")
    print(f"Predicted digit: {predicted_digit}")
    
    # Save the original image with the expected and predicted label
    save_image_and_data(image, processed_image, expected_output, predicted_digit)
    
    return jsonify({'prediction': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
