from sklearn.linear_model import LogisticRegression
import numpy as np


class LRBaseline:
    def __init__(self, class_weight='balanced'):
        self.model = LogisticRegression(
            max_iter=1000, class_weight=class_weight, solver='lbfgs'
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs
