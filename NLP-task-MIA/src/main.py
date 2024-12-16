from typing import Tuple
from lstm_model import LSTMModel
from attack_model import MembershipInferenceAttack
from privacy_preservation import apply_regularization_and_early_stopping
from data_loader import load_data
from model_evaluator import ModelEvaluator
from loguru import logger
import os


def main() -> None:
    """
    Main function to train and evaluate an LSTM model with regularization and early stopping.
    Tracks the impact of these techniques on model performance and membership inference attack (MIA) accuracy.

    Steps:
    1. Train a baseline LSTM model.
    2. Evaluate the baseline model and perform a membership inference attack (MIA).
    3. Apply regularization and early stopping techniques.
    4. Re-evaluate the model and compare MIA accuracy before and after applying the techniques.
    """
    logger.info("Initializing evaluator and loading data.")

    # Initialize evaluator
    evaluator = ModelEvaluator(logs_dir="logs")

    # Load IMDB data
    x_train, y_train, x_test, y_test = load_data()

    logger.info("Data loaded successfully. Starting baseline model training.")

    # 1. Train LSTM model on IMDB data
    lstm_model = LSTMModel()
    base_model, base_history = lstm_model.train(x_train, y_train,x_test,y_test, return_history=True)

    logger.info("Baseline model training complete. Saving and plotting metrics.")

    # Save and plot metrics before privacy preservation
    evaluator.plot_training_history(base_history, "Base Model")

    # Evaluate base model
    evaluator.evaluate_performance(base_model, x_test, y_test, title="Base Model")

    # 2. Perform MIA before applying privacy-preserving techniques
    logger.info("Performing membership inference attack on the baseline model.")
    mia = MembershipInferenceAttack()
    train_conf, test_conf = mia.collect_confidences(base_model, x_train, x_test)
    mia_accuracy_before = mia.train_attack_model(train_conf, test_conf, y_train, y_test)
    mia_accuracy_before, _ = mia_accuracy_before
    logger.info(f"MIA Accuracy before privacy preservation: {mia_accuracy_before:.4f}")

    # 3. Apply regularization and early stopping
    logger.info("Applying regularization and early stopping techniques.")
    regularized_model = apply_regularization_and_early_stopping(
        base_model, x_train, y_train
    )

    logger.info("Regularized model training complete. Saving and plotting metrics.")

    # Save and plot metrics after applying regularization and early stopping
    evaluator.plot_training_history(regularized_model.history, "Regularized Model")

    # Evaluate regularized model
    evaluator.evaluate_performance(
        regularized_model, x_test, y_test, title="Regularized Model"
    )

    # 4. Perform MIA after applying privacy-preserving techniques
    logger.info("Performing membership inference attack on the regularized model.")
    train_conf_dp, test_conf_dp = mia.collect_confidences(
        regularized_model, x_train, x_test
    )
    mia_accuracy_after = mia.train_attack_model(
        train_conf_dp, test_conf_dp, y_train, y_test
    )
    mia_accuracy_after, _ = mia_accuracy_after 
    logger.info(f"MIA Accuracy after applying regularization: {mia_accuracy_after:.4f}")

    # 5. Compare results and visualize
    logger.info("Comparing MIA accuracy results and visualizing.")
    evaluator.plot_mia_results(mia_accuracy_before, mia_accuracy_after)

    results_file_path = "logs/results_comparison.txt"
    try:
        # Attempt to open the file for writing
        with open(results_file_path, "w") as log:
            log.write(
                f"MIA Accuracy before regularization: {mia_accuracy_before:.4f}\n"
            )
            log.write(f"MIA Accuracy after regularization: {mia_accuracy_after:.4f}\n")
        logger.info("Results comparison saved to logs.")
    except FileNotFoundError:
        # Handle missing directory
        logger.warning("Logs directory not found. Creating logs directory.")
        os.makedirs("logs", exist_ok=True)
        with open(results_file_path, "w") as log:
            log.write(
                f"MIA Accuracy before regularization: {mia_accuracy_before:.4f}\n"
            )
            log.write(f"MIA Accuracy after regularization: {mia_accuracy_after:.4f}\n")
        logger.info("Results comparison saved to logs after creating the directory.")


if __name__ == "__main__":
    logger.info("Starting the main process.")
    main()
    logger.info("Main process completed.")
