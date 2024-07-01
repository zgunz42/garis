from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
