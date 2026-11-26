import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import config as C
from .features import encode

LABELS = C.RISK_CLASSES


def split(df):
    X = encode(df)
    y = df[C.TARGET].map({c: i for i, c in enumerate(LABELS)}).astype(int)
    strat = df["source"].astype(str) + "_" + df[C.TARGET].astype(str)
    idx_tr, idx_te = train_test_split(df.index, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=strat)
    return X, y, idx_tr, idx_te


def make_models():
    lr = Pipeline([("scale", StandardScaler()), ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                   ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    rf = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1.0)),
                   ("clf", RandomForestClassifier(class_weight="balanced", random_state=C.RANDOM_STATE, n_jobs=-1))])
    hgb = Pipeline([("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=C.RANDOM_STATE))])
    return {
        "Logistic Regression": (lr, {"clf__C": [0.01, 0.1, 1.0, 10.0]}, "grid"),
        "Random Forest": (rf, {"clf__n_estimators": [200, 300, 400], "clf__max_depth": [8, 12, 16, None],
                               "clf__min_samples_leaf": [1, 3, 5, 10], "clf__max_features": ["sqrt", 0.5]}, "random"),
        "Gradient Boosting": (hgb, {"clf__learning_rate": [0.03, 0.06, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
                                    "clf__min_samples_leaf": [10, 20, 40], "clf__l2_regularization": [0.0, 0.1, 1.0],
                                    "clf__max_iter": [200, 300]}, "random"),
    }


def tune(name, pipe, grid, kind, X, y, cv):
    if kind == "grid":
        s = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    else:
        s = RandomizedSearchCV(pipe, grid, n_iter=12, scoring="f1_macro", cv=cv, n_jobs=-1, random_state=C.RANDOM_STATE)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_


def ci95(values):
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"mean": round(m, 4), "low": round(m - h, 4), "high": round(m + h, 4)}


def cv_scores(model, X, y, cv):
    r = cross_validate(model, X, y, cv=cv, n_jobs=-1,
                       scoring={"f1_macro": "f1_macro", "bal_acc": "balanced_accuracy", "accuracy": "accuracy",
                                "high_recall": _high_recall_scorer})
    return {k.replace("test_", ""): ci95(v) for k, v in r.items() if k.startswith("test_")}


def _high_recall_scorer(est, X, y):
    return recall_score(y, est.predict(X), labels=[2], average="macro", zero_division=0)


def multiclass_brier(y, proba):
    Y = np.eye(len(LABELS))[np.asarray(y)]
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def test_metrics(model, X, y):
    p = model.predict(X)
    pr = model.predict_proba(X)
    return {
        "accuracy": round(accuracy_score(y, p), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, p), 4),
        "f1_macro": round(f1_score(y, p, average="macro"), 4),
        "high_recall": round(recall_score(y, p, labels=[2], average="macro", zero_division=0), 4),
        "brier": round(multiclass_brier(y, pr), 4),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        "report": classification_report(y, p, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0),
    }


def per_source(model, X, y, src):
    rows = []
    p = model.predict(X)
    for s in sorted(src.unique()):
        m = (src == s).values
        maj = pd.Series(y[m]).mode().iloc[0]
        rows.append({"source": s, "n": int(m.sum()),
                     "model_f1_macro": round(f1_score(y[m], p[m], average="macro", labels=sorted(np.unique(y[m]))), 4),
                     "model_bal_acc": round(balanced_accuracy_score(y[m], p[m]), 4),
                     "majority_bal_acc": round(balanced_accuracy_score(y[m], np.full(m.sum(), maj)), 4),
                     "model_accuracy": round(accuracy_score(y[m], p[m]), 4),
                     "majority_accuracy": round(accuracy_score(y[m], np.full(m.sum(), maj)), 4)})
    return rows


def ablations(best, X_tr, y_tr, X_te, y_te):
    from sklearn.base import clone
    groups = {
        "All features": list(X_tr.columns),
        "Without record-type (source) indicator": [c for c in X_tr.columns if not c.startswith("source=")],
        "Record type + demographics only": [c for c in X_tr.columns if c.startswith("source=") or c in ("age", "gender")],
        "Without prior grades (G1, G2, previous scores)": [c for c in X_tr.columns if c not in ("G1", "G2", "previous_scores")],
        "Without attendance / absence features": [c for c in X_tr.columns if c not in ("attendance", "absences", "absence_days")],
    }
    out = []
    for name, cols in groups.items():
        m = clone(best)
        m.fit(X_tr[cols], y_tr)
        p = m.predict(X_te[cols])
        out.append({"setting": name, "n_features": len(cols), "f1_macro": round(f1_score(y_te, p, average="macro"), 4),
                    "high_recall": round(recall_score(y_te, p, labels=[2], average="macro", zero_division=0), 4)})
    return out


def calibration(best, X_tr, y_tr, X_te, y_te):
    cal = CalibratedClassifierCV(best, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)
    pr_raw = best.predict_proba(X_te)[:, 2]
    pr_cal = cal.predict_proba(X_te)[:, 2]
    yb = (np.asarray(y_te) == 2).astype(int)
    bins = np.linspace(0, 1, 11)
    curve = []
    for name, pr in (("uncalibrated", pr_raw), ("isotonic", pr_cal)):
        idx = np.clip(np.digitize(pr, bins) - 1, 0, 9)
        pts = [{"bin": int(b), "pred": round(float(pr[idx == b].mean()), 4), "obs": round(float(yb[idx == b].mean()), 4),
                "n": int((idx == b).sum())} for b in range(10) if (idx == b).sum() > 0]
        curve.append({"model": name, "points": pts})
    return cal, {"brier_high_uncalibrated": round(brier_score_loss(yb, pr_raw), 4),
                 "brier_high_isotonic": round(brier_score_loss(yb, pr_cal), 4),
                 "multiclass_brier_uncalibrated": round(multiclass_brier(y_te, best.predict_proba(X_te)), 4),
                 "multiclass_brier_isotonic": round(multiclass_brier(y_te, cal.predict_proba(X_te)), 4),
                 "f1_macro_isotonic": round(f1_score(y_te, cal.predict(X_te), average="macro"), 4),
                 "reliability": curve}


def run(df, log=print):
    X, y, idx_tr, idx_te = split(df)
    X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr].values, y.loc[idx_te].values
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    results, fitted = {}, {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    results["Majority-class baseline"] = {"cv": cv_scores(DummyClassifier(strategy="most_frequent"), X_tr, y_tr, cv),
                                          "test": test_metrics(dummy, X_te, y_te), "best_params": {}}
    for name, (pipe, grid, kind) in make_models().items():
        t = time.time()
        est, params = tune(name, pipe, grid, kind, X_tr, y_tr, cv)
        est.fit(X_tr, y_tr)
        fitted[name] = est
        results[name] = {"best_params": {k.replace("clf__", ""): (v if v is not None else "None") for k, v in params.items()},
                         "cv": cv_scores(est, X_tr, y_tr, cv), "test": test_metrics(est, X_te, y_te),
                         "train_seconds": round(time.time() - t, 1)}
        log(f"{name}: CV macro-F1 {results[name]['cv']['f1_macro']['mean']}  test macro-F1 {results[name]['test']['f1_macro']}")
    best_name = max(fitted, key=lambda n: results[n]["cv"]["f1_macro"]["mean"])
    best = fitted[best_name]
    src_te = df.loc[idx_te, "source"].astype(str)
    cal_model, cal = calibration(best, X_tr, y_tr, X_te, y_te)
    metrics = {
        "best_model": best_name, "labels": LABELS, "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
        "feature_columns": list(X.columns), "models": results,
        "per_source": per_source(best, X_te, y_te, src_te),
        "ablation": ablations(best, X_tr, y_tr, X_te, y_te),
        "calibration": cal,
        "label_definition": "High = grade band D (failing in the source grading system), Medium = band C, Low = bands A/B",
    }
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, m in fitted.items():
        if name != best_name:
            joblib.dump(m, C.MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"), compress=3)
    joblib.dump(best, C.MODEL_DIR / "best_model.joblib", compress=3)
    joblib.dump({"idx_train": list(map(int, idx_tr)), "idx_test": list(map(int, idx_te))}, C.MODEL_DIR / "split.joblib")
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics, fitted, (X, y, idx_tr, idx_te)
