"""Logistic regression on scaled scalar features. """
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# The 14 scalar features appropriate for a linear model
LOGIT_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "volatility_7", "volatility_30",
    "ma_7_minus_ma_30", "ma_30_minus_ma_90",
    "rsi", "macd_minus_signal", "bb_position", "daily_return",
]


def fit_predict_logit(X_train, y_train, X_test):
    """Fit a scaled logistic regression and return P(up) for test rows.

    The StandardScaler is INSIDE the Pipeline, so it is fit only on the
    training fold. This is the single most important leakage guard for
    scaled models.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=5000)),
    ])
    pipe.fit(X_train[LOGIT_FEATURES], y_train)
    return pipe.predict_proba(X_test[LOGIT_FEATURES])[:, 1]