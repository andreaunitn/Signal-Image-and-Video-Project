from __future__ import absolute_import

from metric_learn.base_metric import BaseMetricLearner

class Cosine(BaseMetricLearner):
    def __init__(self):
        self.X_ = None

    def fit(self, X):
        self.X_ = X

    def transform(self, X=None):
        if X is None:
            return self.X_
        return X
