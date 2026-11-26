import argparse
import json
import os
import sys
import time

import pandas as pd

from src import config as C
from src import data_cleaning, database, explain, fairness, train


def main():
    ap = argparse.ArgumentParser(description="Clean data, train models, explain and audit fairness.")
    ap.add_argument("--raw", default=None, help="Path to the raw CSV (defaults to data/raw/Academic_Risk_Prediction_DW.csv)")
    ap.add_argument("--skip-fairness", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    print("[1/5] Validating and cleaning data")
    raw_path = a.raw or C.RAW_PATH
    if not os.path.exists(raw_path):
        sys.exit(f"Error: dataset not found: {raw_path}")
    v = data_cleaning.validate_schema(data_cleaning.load_raw(raw_path))
    if not v["ok"]:
        sys.exit(f"Error: schema problems - missing columns {v['missing_columns']}, unknown sources {v['unknown_sources']}")
    df, rep = data_cleaning.run(a.raw)
    print(json.dumps({k: v for k, v in rep.items() if k != "extreme_outliers_flagged_iqr3"}, indent=2))
    print("[2/5] Training and tuning models (5-fold CV)")
    metrics, fitted, (X, y, tr, te) = train.run(df)
    best = fitted[metrics["best_model"]]
    print(f"      Best model: {metrics['best_model']}")
    groups_tr = {g: df.loc[tr, g].values for g in C.SENSITIVE_ATTRIBUTES}
    groups_te = {g: df.loc[te, g].values for g in C.SENSITIVE_ATTRIBUTES}
    print("[3/5] SHAP explanations")
    sh, _, _ = explain.run(best, X.loc[te], groups_te, list(X.columns))
    print("      Top features: " + ", ".join(r["label"] for r in sh["importance"][:5]))
    if not a.skip_fairness:
        print("[4/5] Fairness audit and mitigation")
        fairness.run(best, X.loc[tr], y.loc[tr].values, X.loc[te], y.loc[te].values, groups_tr, groups_te)
    print("[5/5] Database")
    database.init_db()
    database.register_models(metrics)
    print(f"Done in {time.time() - t0:.0f}s. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
