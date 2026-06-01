from xgboost import XGBClassifier
import numpy as np


class XGBoostBaseline:
    def __init__(self, scale_pos_weight=10):
        self.model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss',
            use_label_encoder=False
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs
