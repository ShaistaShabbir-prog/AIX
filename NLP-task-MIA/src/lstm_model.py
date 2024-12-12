from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from loguru import logger
import numpy as np
from typing import Tuple
from settings import Settings


class LSTMModel:
    """
    A class to build, train, and evaluate an LSTM-based model for binary classification.

    This model uses an embedding layer followed by an LSTM layer and a dense output layer
    with a sigmoid activation for binary classification tasks.
    """

    def __init__(self):
        """
        Initializes the LSTMModel class. At the moment, the model is set to None.
        """
        self.model = None

    def build(self) -> Sequential:
        """
        Build the LSTM model architecture.

        The model consists of:
        - Embedding layer with `MAX_FEATURES` and a fixed embedding dimension of 128.
        - LSTM layer with 128 units, with dropout and recurrent dropout of 0.2.
        - Dense layer with a sigmoid activation for binary classification.

        Returns:
            Sequential: The compiled Keras model.
        """
        logger.info("Building the LSTM model...")
        settings = Settings()
        model = Sequential()
        model.add(Embedding(settings.MAX_FEATURES, 128, input_length=settings.MAXLEN))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation="sigmoid"))

        # Compile the model with binary cross-entropy loss and Adam optimizer
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        logger.info("LSTM model built successfully.")
        return model

    def train(
        self, x_train: np.ndarray, y_train: np.ndarray, x_val, y_val,return_history: bool = False
    ) -> Tuple[Sequential, dict]:
        """
        Train the LSTM model on the given training data.

        Args:
            x_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels.
            return_history (bool): Flag to indicate whether to return the training history.

        Returns:
            Tuple[Sequential, dict]: The trained model and the training history (if `return_history` is True).
        """
        logger.info("Training the model...")
        if self.model is None:
            self.model = self.build()

        history = self.model.fit(x_train, y_train,validation_data=(x_val, y_val), epochs=3, batch_size=64)

        logger.info(f"Training completed. Model history: {history.history}")
        return self.model, history
