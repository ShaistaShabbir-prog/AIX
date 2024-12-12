from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from typing import Tuple
import numpy as np
from settings import settings


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the IMDB dataset for training a model.

    The function loads the IMDB dataset, which contains movie reviews labeled as
    positive or negative. It then pads the sequences to ensure uniform length and
    returns the training and testing data.

    Returns:
        Tuple: A tuple containing:
            - x_train (np.ndarray): The padded training features.
            - y_train (np.ndarray): The training labels.
            - x_test (np.ndarray): The padded testing features.
            - y_test (np.ndarray): The testing labels.
    """
    # Load the IMDB dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=settings.MAX_FEATURES
    )

    # Pad sequences to ensure uniform length
    x_train = sequence.pad_sequences(x_train, maxlen=settings.MAXLEN)
    x_test = sequence.pad_sequences(x_test, maxlen=settings.MAXLEN)

    return x_train, y_train, x_test, y_test
