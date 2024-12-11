from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

MAX_FEATURES = 20000
MAXLEN = 100
# have to add docstrings and type annotations
# also add logger from loguru


class LSTMModel:
    def __init__(self):
        self.model = None

    def build(self):
        model = Sequential()
        model.add(Embedding(MAX_FEATURES, 128, input_length=MAXLEN))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def train(self, x_train, y_train):
        self.model = self.build()
        self.model.fit(x_train, y_train, epochs=3, batch_size=64)
        return self.model
