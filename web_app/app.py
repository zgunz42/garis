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
    # Convert image to RGBA (just in case it is not already in this mode)
    image = image.convert('RGBA')
    
    # Separate the alpha channel
    r, g, b, a = image.split()
    
    # Create a new image with a white background
    white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
    white_bg.paste(image, mask=a)
    
    # Convert the image to grayscale
    image = white_bg.convert('L')
    
    # Invert the colors (make it black on white)
    image = ImageOps.invert(image)
    
    # Trim empty space around the digit
    bbox = image.getbbox()
    image = image.crop(bbox)
    
    # Add padding (5% of the maximum dimension)
    max_dim = max(image.size)
    padding = int(max_dim * 0.15)
    image = ImageOps.expand(image, border=padding, fill=0)
    
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    
    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # invert the colors black to white and white to black
    # image = 1 - image
    
    # Save the preprocessed image for debugging
    debug_image = Image.fromarray((image.squeeze() * 255).astype(np.uint8))
    debug_image.save('../data/debug/preprocessed_debug.png')
    
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

    # check expected output not empty
    if not expected_output:
        return
    
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
