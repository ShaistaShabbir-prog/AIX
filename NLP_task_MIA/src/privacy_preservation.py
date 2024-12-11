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
