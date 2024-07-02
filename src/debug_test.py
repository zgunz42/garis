from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Preprocess the image
def preprocess_image(image):
    # Ensure the image is in RGBA mode
    # image = image.convert('RGBA')
    
    # Separate the alpha channel
    # r, g, b, a = image.split()
    
    # Create a new image with a black background
    # black_bg = Image.new('RGBA', image.size, (0, 0, 0, 255))
    # black_bg.paste(image, mask=a)
    
    # Convert the image to grayscale
    # image = black_bg.convert('L')

    # Invert the colors (make it black on white)
    # image = ImageOps.invert(image)
    
    # Trim empty space around the digit
    # bbox = image.getbbox()
    # image = image.crop(bbox)
    
    # Add padding (5% of the maximum dimension)
    # max_dim = max(image.size)
    # padding = int(max_dim * 0.05)
    # image = ImageOps.expand(image, border=padding, fill=0)  # fill with black
    
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    
    # Invert the colors (make it white on black)
    # image = 1 - image
    
    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    return image


if __name__ == "__main__":
    # Load the preprocessed image
    image_path = 'data/debug/preprocessed_debug.png'
    image = Image.open(image_path)
    preprocessed_image = preprocess_image(image)

    # Load the trained model
    model = load_model('models/model.h5')

    # Predict the digit
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction, axis=1)

    print(f"Predicted digit: {predicted_digit[0]}")
