# import tensorflow_privacy as tfp
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Embedding


# def apply_differential_privacy(model, x_train, y_train):
#     # Apply Differential Privacy using DP-SGD optimizer
#     privacy_optimizer = tfp.optimizers.DPKerasSGD(
#         l2_norm_clip=1.0, noise_multiplier=1.1, num_microbatches=64, learning_rate=0.01
#     )

#     # Rebuild the model with differential privacy
#     model_dp = Sequential()
#     model_dp.add(Embedding(20000, 128, input_length=100))
#     model_dp.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#     model_dp.add(Dense(1, activation="sigmoid"))

#     model_dp.compile(
#         optimizer=privacy_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
#     )
#     model_dp.fit(x_train, y_train, epochs=3, batch_size=64)
#     return model_dp
# currently differential privacy is having some import problem
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model


def apply_regularization_and_early_stopping(
    model: Model, x_train: np.ndarray, y_train: np.ndarray
) -> Model:
    """
    Apply L2 regularization, dropout, and early stopping to the model.

    Args:
        model (Model): The untrained or pre-trained Keras model to apply techniques.
        x_train (np.ndarray): The training data (features).
        y_train (np.ndarray): The training data (labels).

    Returns:
        Model: The trained Keras model with applied regularization and early stopping.
    """
    # L2 regularization
    model.add(Embedding(20000, 128, input_length=100))
    model.add(
        LSTM(
            128,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(0.01),
        )
    )  # L2 regularization
    model.add(Dropout(0.5))  # Additional Dropout regularization
    model.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Train the model with early stopping
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    return model


def adversarial_training(
    x_train: np.ndarray, y_train: np.ndarray, model: Model, epsilon: float = 0.1
) -> Model:
    """
    Apply adversarial training to the model by creating adversarial examples.

    Args:
        x_train (np.ndarray): The training data (features).
        y_train (np.ndarray): The training data (labels).
        model (Model): The Keras model to train with adversarial examples.
        epsilon (float): The magnitude of the perturbations for adversarial examples.

    Returns:
        Model: The trained Keras model with adversarial training applied.
    """
    # Simple perturbations to create adversarial examples
    perturbations = np.random.normal(scale=epsilon, size=x_train.shape)
    x_train_adv = x_train + perturbations

    # Train the model on adversarial examples
    model.fit(x_train_adv, y_train, epochs=5, batch_size=64)
    return model


def cross_validate_model(
    model: Model, x_data: np.ndarray, y_data: np.ndarray, n_splits: int = 5
) -> None:
    """
    Perform k-fold cross-validation on the model.

    Args:
        model (Model): The Keras model to evaluate.
        x_data (np.ndarray): The feature data for cross-validation.
        y_data (np.ndarray): The label data for cross-validation.
        n_splits (int): The number of splits for k-fold cross-validation.

    Returns:
        None
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    results = []

    for train_index, val_index in kf.split(x_data):
        x_train, x_val = x_data[train_index], x_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        # Compile model (important before training)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train the model
        model.fit(
            x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val)
        )

        # Evaluate model on validation set
        val_loss, val_accuracy = model.evaluate(x_val, y_val)
        results.append(val_accuracy)

    print(f"Cross-Validation Accuracy: {np.mean(results)} ± {np.std(results)}")


def calibrate_model(model: Model, x_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Apply model calibration using temperature scaling.

    Args:
        model (Model): The trained Keras model to calibrate.
        x_val (np.ndarray): The validation data (features).
        y_val (np.ndarray): The validation data (labels).

    Returns:
        np.ndarray: The calibrated predictions (softmax probabilities).
    """
    # Calibrate model (using temperature scaling)
    logits = model.predict(x_val)
    temperature = 2.0  # tune this value
    logits = logits / temperature
    predictions = tf.nn.softmax(logits)
    return predictions
