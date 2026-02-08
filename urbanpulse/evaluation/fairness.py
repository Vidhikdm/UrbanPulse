from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def errors_by_income_quartile(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    df = pd.DataFrame({
        "true": y_true,
        "pred": y_pred,
        "err": y_pred - y_true,
        "abs_err": np.abs(y_pred - y_true),
    })

    df["quartile"] = pd.qcut(
        df["true"],
        q=4,
        labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
        duplicates="drop",
    )

    summary = (
        df.groupby("quartile", observed=True)
          .agg(
              mae=("abs_err", "mean"),
              rmse=("err", lambda x: float(np.sqrt(np.mean(np.square(x))))),
              bias=("err", "mean"),
              income_min=("true", "min"),
              income_max=("true", "max"),
              income_mean=("true", "mean"),
              n=("true", "count"),
          )
          .reset_index()
    )

    return summary

def save_fairness_report(y_true, y_pred, out_json: str | Path, metadata=None):
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary = errors_by_income_quartile(y_true, y_pred)

    report = {
        "metadata": metadata or {},
        "overall": {
            "mae": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
            "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))),
            "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
        },
        "by_quartile": summary.to_dict(orient="records"),
    }

    with out_json.open("w") as f:
        json.dump(report, f, indent=2)

    out_csv = out_json.with_suffix(".csv")
    summary.to_csv(out_csv, index=False)

    print(f"✅ Saved fairness report: {out_json}")
    print(f"✅ Saved fairness table:  {out_csv}")
    return report
