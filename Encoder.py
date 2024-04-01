import tensorflow as tf
from tensorflow.keras import layers, Model


class Encoder:
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim
        self.encoder_model = None

    def build_model(self, input_shape):
        # Define the encoder architecture
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(x)

        # Create the encoder model
        self.encoder_model = Model(inputs, encoded, name='encoder')

        # Compile the model
        self.encoder_model.compile(optimizer='adam', loss='mse')

    def train(self, X_train, epochs=10, batch_size=32):
        # Train the model
        self.encoder_model.fit(X_train, X_train, epochs=epochs,
                               batch_size=batch_size)

    def save_model(self, filepath):
        # Save the encoder model
        self.encoder_model.save(filepath)
