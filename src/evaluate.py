import tensorflow as tf
from data_preprocessing import load_and_preprocess_data

def evaluate_model():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = tf.keras.models.load_model('models/model.h5')

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    evaluate_model()
