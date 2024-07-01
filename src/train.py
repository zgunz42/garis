import tensorflow as tf
from data_preprocessing import load_and_preprocess_data
from model import build_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save('models/model.h5')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
