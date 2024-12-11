from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

MAX_FEATURES = 20000
MAXLEN = 100


# TODO: maybe i should move these two var to a util/settings. Can we use pydantic here?
def load_data():
    # Load the IMDB dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

    # Pad sequences to ensure uniform length
    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)

    return x_train, y_train, x_test, y_test
