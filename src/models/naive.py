"""Naive baselines"""
import numpy as np


def fit_predict_prior(X_train, y_train, X_test):
    """Predict the historical base rate of 'up' for every test row.

    This is the 'always guess the majority tendency' model. Its probability
    is just the fraction of up-days seen in training.
    """
    base_rate = y_train.mean()
    return np.full(len(X_test), base_rate)