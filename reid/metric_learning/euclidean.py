from __future__ import absolute_import

from metric_learn.base_metric import BaseMetricLearner
import numpy as np

class Euclidean(BaseMetricLearner):
    def __init__(self):
        self.M_ = None

    def metric(self):
        return self.M_

    def fit(self, X):
        self.M_ = np.eye(X.shape[1])
        self.X_ = X

    def transform(self, X = None):
        if X is None:
            return self.X_
        
        return X
