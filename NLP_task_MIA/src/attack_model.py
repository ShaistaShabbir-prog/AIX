from sklearn.ensemble import RandomForestClassifier
import numpy as np

class MembershipInferenceAttack:
    def __init__(self):
        self.attack_model = RandomForestClassifier()

    def collect_confidences(self, target_model, x_train, x_test):
        # Get the predicted probabilities for training and testing data
        train_confidences = np.max(target_model.predict(x_train), axis=1)
        test_confidences = np.max(target_model.predict(x_test), axis=1)
        return train_confidences, test_confidences

    def train_attack_model(self, train_confidences, test_confidences, y_train, y_test):
        # Combine confidences and true labels for training
        X_train = np.column_stack([train_confidences, y_train])
        X_test = np.column_stack([test_confidences, y_test])

        # Train the attack model
        self.attack_model.fit(X_train, y_train)
        attack_accuracy = self.attack_model.score(X_test, y_test)
        print(f"MIA Accuracy: {attack_accuracy}")
        return attack_accuracy
