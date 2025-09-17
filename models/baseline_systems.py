import pandas as pd
import numpy as np

class MajorityBaseline:
    """
    A simple baseline classifier that always predicts the majority class
    from the training data.
    """
    def __init__(self):
        self.majority_label_ = None

    def fit(self, y_train):
        """
        Finds and stores the majority label from the training data.
        
        Args:
            y_train (pd.Series): A pandas Series containing the training labels.
        """
        self.majority_label_ = y_train.mode()[0] # use pandas 'mode' to find majority label
        return self

    def predict(self, X_test):
        """
        Predicts the stored majority label for all instances in X_test.
        
        Args:
            X_test (iterable): The test data (its content is ignored).
            
        Returns:
            np.ndarray: An array containing the majority label, repeated for
                        each instance in X_test.
        """
        return np.full(shape=len(X_test), fill_value=self.majority_label_)

class RuleBasedBaseline:
    """
    A simple baseline classifier that uses a set of hardcoded keyword rules
    to predict the dialogue act.
    """
    def __init__(self):
        # The majority label is a fallback if no rules match.
        self.fallback_label = "inform"
        self.rules = {
            "hello": ["hello", "welcome", "hi"],
            "thankyou": ["thank", "thanks", "noice",],
            "affirm": ["yes", "correct", "yea", "ye"],
            "deny": ["no","not","dont want", "wrong","something else"],
            "bye": ["bye", "goodbye", "good bye"],
            "repeat": ["again", "repeat", "say that again", "once more"],
            "reqmore": ["more"],
            "reqalts": ["different","there another" ,"other", "alternatives", "alternative","anything else","what about","how about","next"],
            "request": ["what", "which", "could","address","phone number"],
            "negate": ["not", "don't", "do not", "nothing", "no", "never"],
            "confirm": ["yes"],
            "restart": ["restart"],
            "ack": ["ok", "okay"],
            "inform": ["i want"]
        }

    def fit(self, X_train, y_train=None):
        # This model does not learn from data, so fit does nothing, it's a no-op.
        return self

    def predict(self, X_test):
        """
        Predicts the dialogue act for each utterance in X_test based on rules.
        
        Args:
            X_test (iterable): A list or Series of utterance strings.
            
        Returns:
            list: A list of predicted labels.
        """
        predictions = []

        # Loop through test data
        for utterance in X_test:
            predictions.append(self._predict_single(utterance)) # Store each prediction
        return predictions

    # Use manually defined rules to assign the intent by rule matching
    # (falling back to the majority label)
    def _predict_single(self, utterance):
        utterance_words = utterance.lower().split()
        for intent, keywords in self.rules.items():
            for keyword in keywords:
                keyword_words = keyword.split()
                if len(keyword_words) == 1 and keyword_words[0] in utterance_words:
                    return intent
                elif len(keyword_words) > 1:
                    for i in range(len(utterance_words) - len(keyword_words) + 1):
                        if utterance_words[i:i+len(keyword_words)] == keyword_words:
                            return intent
        return self.fallback_label
