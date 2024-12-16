import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from loguru import logger
from typing import Tuple


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
        self.attack_model = RandomForestClassifier(n_estimators=100, random_state=42)

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
        train_confidences = np.max(target_model.predict(x_train), axis=1)
        test_confidences = np.max(target_model.predict(x_test), axis=1)

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
    ) -> Tuple[float, float]:
        """
        Train the Membership Inference Attack model using the collected confidences and true labels.

        Args:
            train_confidences (np.ndarray): The confidences for the training data.
            test_confidences (np.ndarray): The confidences for the testing data.
            y_train (np.ndarray): The true labels for the training data.
            y_test (np.ndarray): The true labels for the testing data.

        Returns:
            Tuple[float, float]: The accuracy and ROC-AUC score of the attack model on the testing data.
        """
        # Combine confidences and true labels for training
        X_train = np.column_stack([train_confidences, y_train])
        X_test = np.column_stack([test_confidences, y_test])

        logger.info("Training the attack model...")
        self.attack_model.fit(X_train, y_train)

        # Evaluate attack model on the test set
        attack_accuracy = self.attack_model.score(X_test, y_test)

        # Ensure predict_proba works without indexing error
        try:
            y_pred_proba = self.attack_model.predict_proba(X_test)[:, 1]  # Positive class probabilities
        except IndexError:
            logger.warning("Model returned only one class; cannot compute ROC-AUC.")
            y_pred_proba = np.zeros_like(y_test)  # Assign zero probabilities for uniformity

        # Calculate ROC-AUC only if probabilities are valid
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = float("nan")  
            logger.warning("ROC-AUC is undefined for a single-class test set.")

        logger.info(f"MIA Accuracy: {attack_accuracy:.4f}")
        logger.info(f"MIA ROC-AUC: {roc_auc:.4f}")

        return attack_accuracy, roc_auc
