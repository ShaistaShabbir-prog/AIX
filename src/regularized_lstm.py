"""
Fix: Proper L2 regularization + EarlyStopping for LSTM MIA model.
Closes #8
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam


def build_regularized_lstm(
    vocab_size: int = 20000,
    embed_dim: int = 128,
    lstm_units: int = 128,
    dropout_rate: float = 0.4,
    l2_weight: float = 0.001,
) -> Sequential:
    """
    LSTM with L2 regularization and dropout to reduce overfitting.
    Suitable for MIA experiments — lower overfitting = harder to attack.
    """
    model = Sequential([
        Embedding(vocab_size, embed_dim, mask_zero=True),
        LSTM(
            lstm_units,
            kernel_regularizer=l2(l2_weight),
            recurrent_regularizer=l2(l2_weight),
            return_sequences=True,
        ),
        Dropout(dropout_rate),
        LSTM(
            lstm_units // 2,
            kernel_regularizer=l2(l2_weight),
        ),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_path: str = "best_model.keras") -> list:
    """
    Callbacks: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


if __name__ == "__main__":
    # Quick smoke test
    model = build_regularized_lstm()
    model.summary()
    callbacks = get_callbacks()
    print(f"\n✅ {len(callbacks)} callbacks configured")
    print("   - EarlyStopping (patience=5, restore_best_weights=True)")
    print("   - ReduceLROnPlateau (factor=0.5, patience=3)")
    print("   - ModelCheckpoint (save best val_loss)")
