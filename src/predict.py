import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data

def make_predictions():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = tf.keras.models.load_model('models/model.h5')

    predictions = model.predict(x_test)

    # Plot the first test image, true label, and predicted label
    plt.figure()
    plt.imshow(x_test[0], cmap=plt.cm.binary)
    plt.title(f"True label: {y_test[0]}, Predicted: {np.argmax(predictions[0])}")
    plt.show()

    print("Predictions made successfully.")
    print(f'True label: {y_test[0]}, Predicted: {np.argmax(predictions[0])}')

if __name__ == "__main__":
    make_predictions()
