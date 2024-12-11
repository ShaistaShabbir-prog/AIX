from lstm_model import LSTMModel
from attack_model import MembershipInferenceAttack
from privacy_preservation import apply_differential_privacy
from data_loader import load_data
import numpy as np


def main():
    # Load IMDB data
    x_train, y_train, x_test, y_test = load_data()

    # 1. Train LSTM model on IMDB data
    lstm_model = LSTMModel()
    model = lstm_model.train(x_train, y_train)

    # 2. Perform MIA before privacy preservation
    mia = MembershipInferenceAttack()
    train_conf, test_conf = mia.collect_confidences(model, x_train, x_test)
    mia_accuracy_before = mia.train_attack_model(train_conf, test_conf, y_train, y_test)
    print(f"MIA Accuracy before privacy preservation: {mia_accuracy_before}")

    # 3. Apply privacy-preserving techniques (e.g., Differential Privacy)
    model_with_privacy = apply_differential_privacy(model, x_train, y_train)

    # 4. Perform MIA after privacy preservation
    train_conf_dp, test_conf_dp = mia.collect_confidences(
        model_with_privacy, x_train, x_test
    )
    mia_accuracy_after = mia.train_attack_model(
        train_conf_dp, test_conf_dp, y_train, y_test
    )
    print(f"MIA Accuracy after privacy preservation: {mia_accuracy_after}")

    # 5. Compare results
    with open("logs/results_comparison.txt", "w") as log:
        log.write(f"MIA Accuracy before privacy preservation: {mia_accuracy_before}\n")
        log.write(f"MIA Accuracy after privacy preservation: {mia_accuracy_after}\n")


if __name__ == "__main__":
    main()
