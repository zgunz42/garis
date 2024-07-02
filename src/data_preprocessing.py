import tensorflow as tf

def load_and_preprocess_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # ensure train and test sets are in the correct shape
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print("Data loaded and preprocessed successfully.")
