from lstm_model import LSTMModel
from attack_model import MembershipInferenceAttack
from privacy_preservation import (
    apply_regularization_and_early_stopping,
    adversarial_training,
    calibrate_model,
)
from data_loader import load_data
from model_evaluator import ModelEvaluator


def main():
    # Initialize evaluator
    evaluator = ModelEvaluator(logs_dir="logs")

    # Load IMDB data
    x_train, y_train, x_test, y_test = load_data()

    # 1. Train LSTM model on IMDB data
    lstm_model = LSTMModel()
    base_model, base_history = lstm_model.train(x_train, y_train, return_history=True)

    # Save and plot metrics before privacy preservation
    evaluator.plot_training_history(base_history, "Base Model")

    # Evaluate base model
    evaluator.evaluate_performance(base_model, x_test, y_test, title="Base Model")

    # 2. Perform MIA before applying privacy-preserving techniques
    mia = MembershipInferenceAttack()
    train_conf, test_conf = mia.collect_confidences(base_model, x_train, x_test)
    mia_accuracy_before = mia.train_attack_model(train_conf, test_conf, y_train, y_test)
    print(f"MIA Accuracy before privacy preservation: {mia_accuracy_before}")

    # 3. Apply privacy-preserving techniques
    print("Applying privacy-preserving techniques...")
    regularized_model = apply_regularization_and_early_stopping(
        base_model, x_train, y_train
    )
    adversarial_model = adversarial_training(x_train, y_train, regularized_model)
    calibrated_model = calibrate_model(adversarial_model, x_test, y_test)

    # Save and plot metrics after privacy preservation
    evaluator.plot_training_history(
        adversarial_model.history, "Privacy-Preserved Model"
    )

    # Evaluate privacy-preserved model
    evaluator.evaluate_performance(
        adversarial_model, x_test, y_test, title="Privacy-Preserved Model"
    )

    # 4. Perform MIA after applying privacy-preserving techniques
    train_conf_dp, test_conf_dp = mia.collect_confidences(
        adversarial_model, x_train, x_test
    )
    mia_accuracy_after = mia.train_attack_model(
        train_conf_dp, test_conf_dp, y_train, y_test
    )
    print(f"MIA Accuracy after privacy preservation: {mia_accuracy_after}")

    # 5. Compare results and visualize
    evaluator.plot_mia_results(mia_accuracy_before, mia_accuracy_after)

    with open("logs/results_comparison.txt", "w") as log:
        log.write(f"MIA Accuracy before privacy preservation: {mia_accuracy_before}\n")
        log.write(f"MIA Accuracy after privacy preservation: {mia_accuracy_after}\n")


if __name__ == "__main__":
    main()
