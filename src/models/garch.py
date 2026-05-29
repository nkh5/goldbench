"""GJR-GARCH(1,1) baseline for volatility forecasting via the arch library.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HORIZON = 3
ANNUALIZE = np.sqrt(252)


def fit_predict_garch(X_train, y_train, X_test):
    """Fit GJR-GARCH on training returns, forecast 5-day vol for each test row.

    GARCH models returns, not the vol target directly, so we use the daily
    log returns embedded in the feature frame (column 'ret_1'). For each test
    row we refit on history up to that point and forecast HORIZON days ahead.
    """
    from arch import arch_model

    # Combine train + test chronologically so we can roll forward
    train = X_train.copy()
    train["_is_test"] = False
    test = X_test.copy()
    test["_is_test"] = True
    full = pd.concat([train, test], ignore_index=True)

    # Daily log returns, scaled to percent (arch fits better on percent returns)
    rets = full["ret_1"].values * 100.0

    preds = []
    # The first test row sits right after the training block
    n_train = len(train)
    for i in range(len(test)):
        # History = everything strictly before this test row
        hist_end = n_train + i
        hist_rets = rets[:hist_end]
        # Need a minimum history for a stable fit
        if len(hist_rets) < 100 or np.allclose(np.nanstd(hist_rets), 0):
            preds.append(np.nan)
            continue
        try:
            am = arch_model(hist_rets, vol="GARCH", p=1, o=1, q=1,
                            dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=HORIZON, reindex=False)
            # Variance forecast per day over the horizon; sum then annualize.
            # Divide by 100^2 to undo the percent scaling.
            daily_var = fc.variance.values[-1] / (100.0 ** 2)
            horizon_var = np.sum(daily_var)
            horizon_vol = np.sqrt(horizon_var / HORIZON) * ANNUALIZE
            preds.append(horizon_vol)
        except Exception:
            preds.append(np.nan)

    preds = np.array(preds)
    # Fill any failed fits with the trailing realized vol as a fallback
    if np.isnan(preds).any():
        fallback = np.nanmean(preds) if not np.all(np.isnan(preds)) else y_train.mean()
        preds = np.where(np.isnan(preds), fallback, preds)
    return preds