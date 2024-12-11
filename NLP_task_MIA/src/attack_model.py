import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MembershipInferenceAttack:
    """
    A class for performing Membership Inference Attacks (MIA) on a target model.

    The attack uses the predicted confidences from a target model to determine
    whether a given instance was part of the training set or not.
    """

    def __init__(self):
        """
        Initializes the MembershipInferenceAttack class with a Random Forest classifier
        for training the attack model.
        """
        self.attack_model = RandomForestClassifier()

    def collect_confidences(
        self, target_model, x_train: np.ndarray, x_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect predicted confidences (probabilities) from the target model.

        Args:
            target_model: The model to attack.
            x_train (np.ndarray): The training dataset features.
            x_test (np.ndarray): The testing dataset features.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The confidences for the training and testing data.
        """
        # Get the predicted probabilities for training and testing data
        train_confidences = np.max(target_model.predict_proba(x_train), axis=1)
        test_confidences = np.max(target_model.predict_proba(x_test), axis=1)

        logger.info(
            f"Collected confidences: train={train_confidences.shape[0]}, test={test_confidences.shape[0]}"
        )
        return train_confidences, test_confidences

    def train_attack_model(
        self,
        train_confidences: np.ndarray,
        test_confidences: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Train the Membership Inference Attack model using the collected confidences and true labels.

        Args:
            train_confidences (np.ndarray): The confidences for the training data.
            test_confidences (np.ndarray): The confidences for the testing data.
            y_train (np.ndarray): The true labels for the training data.
            y_test (np.ndarray): The true labels for the testing data.

        Returns:
            float: The accuracy of the attack model on the testing data.
        """
        # Combine confidences and true labels for training
        X_train = np.column_stack([train_confidences, y_train])
        X_test = np.column_stack([test_confidences, y_test])

        # Train the attack model
        logger.info("Training the attack model...")
        self.attack_model.fit(X_train, y_train)

        # Evaluate attack model on the test set
        attack_accuracy = self.attack_model.score(X_test, y_test)
        logger.info(f"MIA Accuracy: {attack_accuracy:.4f}")

        return attack_accuracy
