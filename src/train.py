import tensorflow as tf
from data_preprocessing import load_and_preprocess_data
from model import build_model

def train_model():
    """
    Trains a model using the MNIST dataset, preprocessed data, and the model architecture defined in the `model.py` file.
    Saves the trained model to the `models/model.h5` file.
    """
    # Load and preprocess the data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build the model
    model = build_model()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Optimizer
        loss='sparse_categorical_crossentropy',  # Loss function
        metrics=['accuracy']  # Evaluation metric
    )

    # Train the model
    model.fit(
        x_train,  # Training data
        y_train,  # Training labels
        epochs=8,  # Number of epochs
        validation_data=(x_test, y_test)  # Validation data
    )

    # Save the trained model
    model.save('models/model.h5')
    
    # Print success message
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
