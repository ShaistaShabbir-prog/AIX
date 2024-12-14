import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import History


class ModelEvaluator:
    """
    A class to evaluate and visualize the performance of machine learning models.
    It includes methods to generate classification reports, confusion matrices,
    plot training histories, and visualize Membership Inference Attack (MIA) results.
    """

    def __init__(self, logs_dir: str = "src/logs"):
        """
        Initialize the evaluator with a directory to store logs.

        Args:
            logs_dir (str): The directory where logs and plots will be saved.
        """
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)  # Ensure the logs directory exists

    def evaluate_performance(
        self, model, x_test: np.ndarray, y_test: np.ndarray, title: str = "Model"
    ) -> tuple[str, np.ndarray]:
        """
        Evaluate the classification performance of the model and log the results.

        Args:
            model: The trained model to evaluate.
            x_test (np.ndarray): The test input data.
            y_test (np.ndarray): The true labels for the test data.
            title (str): The title for the evaluation report and plot.

        Returns:
            tuple[str, np.ndarray]: The classification report (as a string) and confusion matrix (as a numpy array).
        """
        y_pred = model.predict(x_test)
        y_pred_rounded = np.round(y_pred)

        # Generate and print classification report
        report = classification_report(y_test, y_pred_rounded, digits=4)
        print(f"{title} Classification Report:\n{report}")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_rounded)
        print(f"{title} Confusion Matrix:\n{conf_matrix}")

        # Save report to file with try-except
        log_file_path = f"{self.logs_dir}/{title}_classification_report.txt"
        try:
            with open(log_file_path, "a") as f:  # Append if the file exists
                f.write(f"{title} Classification Report:\n{report}\n")
                f.write(f"{title} Confusion Matrix:\n{conf_matrix}\n")
        except FileNotFoundError:
            with open(log_file_path, "w") as f:  # Create a new file if it doesn’t exist
                f.write(f"{title} Classification Report:\n{report}\n")
                f.write(f"{title} Confusion Matrix:\n{conf_matrix}\n")

        return report, conf_matrix

    def plot_training_history(
        self, history: History, title: str = "Training History"
    ) -> None:
        """
        Plot and save the training and validation accuracy and loss over epochs.

        Args:
            history (History): The training history object containing accuracy and loss values.
            title (str): The title for the plot.
        """
        plt.figure(figsize=(12, 6))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
        plt.title(f"{title} - Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history.get("val_loss", []), label="Validation Loss")
        plt.title(f"{title} - Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()

        # Save the plot to logs directory
        plot_file_path = f"{self.logs_dir}/{title}_metrics.png"
        try:
            plt.savefig(plot_file_path)
            print(f"Plot saved to {plot_file_path}")
        except FileNotFoundError:
            print("Error: Logs directory not found. Creating logs directory.")
            os.makedirs(self.logs_dir, exist_ok=True)
            plt.savefig(plot_file_path)

        plt.show()

    def plot_mia_results(self, mia_before: float, mia_after: float) -> None:
        """
        Plot the Membership Inference Attack (MIA) accuracy before and after privacy preservation.

        Args:
            mia_before (float): The MIA accuracy before privacy preservation.
            mia_after (float): The MIA accuracy after privacy preservation.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(
            ["Before Privacy", "After Privacy"],
            [mia_before, mia_after],
            color=["red", "green"],
        )
        plt.title("MIA Accuracy Comparison")
        plt.ylabel("MIA Accuracy")

        # Save the plot to logs directory
        plot_file_path = f"{self.logs_dir}/mia_comparison.png"
        try:
            plt.savefig(plot_file_path)
            print(f"MIA comparison plot saved to {plot_file_path}")
        except FileNotFoundError:
            print("Error: Logs directory not found. Creating logs directory.")
            os.makedirs(self.logs_dir, exist_ok=True)
            plt.savefig(plot_file_path)

        plt.show()
