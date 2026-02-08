from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Spearman without scipy (safe fallback)
    try:
        import pandas as pd
        rho = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    except Exception:
        rho = None

    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": rho}
