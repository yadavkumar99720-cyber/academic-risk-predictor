import json
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C

HIGH = 2
warnings.filterwarnings("ignore")


class ScoreAsProba(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def group_table(y_true, y_pred, groups):
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        yt, yp = y_true[m] == HIGH, y_pred[m] == HIGH
        pos = yt.sum()
        tpr = (yt & yp).sum() / pos if pos else np.nan
        neg = (~yt).sum()
        fpr = ((~yt) & yp).sum() / neg if neg else np.nan
        prec = (yt & yp).sum() / yp.sum() if yp.sum() else np.nan
        rows.append({"group": str(g), "n": int(m.sum()), "actual_high": int(pos), "flagged_high": int(yp.sum()),
                     "selection_rate": round(float(yp.mean()), 4), "high_recall": _r(tpr), "fnr": _r(1 - tpr if pos else np.nan),
                     "fpr": _r(fpr), "precision_high": _r(prec),
                     "f1_macro": round(float(f1_score(y_true[m], y_pred[m], average="macro", labels=[0, 1, 2], zero_division=0)), 4),
                     "pred_low": round(float((y_pred[m] == 0).mean()), 4), "pred_medium": round(float((y_pred[m] == 1).mean()), 4),
                     "pred_high": round(float((y_pred[m] == 2).mean()), 4)})
    return rows


def _r(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


def gaps(y_true, y_pred, groups, n_boot=500, seed=C.RANDOM_STATE):
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    keep = groups != "Unknown"
    y_true, y_pred, groups = y_true[keep], y_pred[keep], groups[keep]

    def calc(yt, yp, gg):
        tprs, srs = [], []
        for g in np.unique(gg):
            m = gg == g
            pos = (yt[m] == HIGH)
            if pos.sum() == 0:
                continue
            tprs.append(((yp[m] == HIGH) & pos).sum() / pos.sum())
            srs.append((yp[m] == HIGH).mean())
        if len(tprs) < 2:
            return np.nan, np.nan
        return max(tprs) - min(tprs), max(srs) - min(srs)

    tg, sg = calc(y_true, y_pred, groups)
    rng = np.random.default_rng(seed)
    bt, bs = [], []
    n = len(y_true)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, b = calc(y_true[i], y_pred[i], groups[i])
        bt.append(a)
        bs.append(b)
    bt, bs = np.array(bt), np.array(bs)
    return {"recall_gap": _r(tg), "recall_gap_ci": [_r(np.nanpercentile(bt, 2.5)), _r(np.nanpercentile(bt, 97.5))],
            "selection_rate_gap": _r(sg), "selection_rate_gap_ci": [_r(np.nanpercentile(bs, 2.5)), _r(np.nanpercentile(bs, 97.5))]}


def overall(y_true, y_pred):
    return {"f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "high_recall": round(float(recall_score(y_true, y_pred, labels=[HIGH], average="macro", zero_division=0)), 4)}


def reweigh_weights(y, groups):
    df = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(groups)})
    pg = df["g"].value_counts(normalize=True)
    py = df["y"].value_counts(normalize=True)
    pgy = df.groupby(["g", "y"]).size() / len(df)
    return df.apply(lambda r: pg[r.g] * py[r.y] / pgy[(r.g, r.y)], axis=1).values


def merge_high_flag(proba, flag):
    rest = np.where(proba[:, 0] >= proba[:, 1], 0, 1)
    return np.where(np.asarray(flag) == 1, HIGH, rest)


def method_reweighing(best, X_tr, y_tr, g_tr, X_te):
    m = clone(best)
    w = reweigh_weights(y_tr, g_tr)
    m.fit(X_tr, y_tr, clf__sample_weight=w)
    return m.predict(X_te), m


def method_threshold(best, X_tr, y_tr, g_tr, X_te, g_te):
    from fairlearn.postprocessing import ThresholdOptimizer
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    oof = cross_val_predict(clone(best), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, HIGH]
    to = ThresholdOptimizer(estimator=ScoreAsProba().fit(None), constraints="true_positive_rate_parity",
                            objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
    to.fit(pd.DataFrame({"s": oof}), (np.asarray(y_tr) == HIGH).astype(int), sensitive_features=np.asarray(g_tr))
    pr = best.predict_proba(X_te)
    flag = to.predict(pd.DataFrame({"s": pr[:, HIGH]}), sensitive_features=np.asarray(g_te), random_state=C.RANDOM_STATE)
    return merge_high_flag(pr, flag), to


def method_expgrad(best, X_tr, y_tr, g_tr, X_te):
    from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
    base = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, class_weight="balanced", random_state=C.RANDOM_STATE)
    eg = ExponentiatedGradient(base, constraints=TruePositiveRateParity(difference_bound=0.02), max_iter=30)
    eg.fit(X_tr, (np.asarray(y_tr) == HIGH).astype(int), sensitive_features=np.asarray(g_tr))
    flag = eg.predict(X_te, random_state=C.RANDOM_STATE)
    return merge_high_flag(best.predict_proba(X_te), flag), eg


def run(best, X_tr, y_tr, X_te, y_te, groups_tr, groups_te, mitigate_on=("gender", "ses_group"), log=print):
    base_pred = best.predict(X_te)
    out = {"decision": "A student is flagged when the model predicts the High-risk class. Fairness is measured on this flag.",
           "baseline": {}, "mitigation": {}}
    for attr in C.SENSITIVE_ATTRIBUTES:
        out["baseline"][attr] = {"groups": group_table(y_te, base_pred, groups_te[attr]),
                                 "gaps": gaps(y_te, base_pred, groups_te[attr])}
    models = {}
    for attr in mitigate_on:
        res = [{"method": "Baseline (no mitigation)", "stage": "-", **overall(y_te, base_pred),
                **gaps(y_te, base_pred, groups_te[attr]), "groups": group_table(y_te, base_pred, groups_te[attr])}]
        for label, stage, fn in (("Reweighing", "Pre-processing", "rw"), ("Exponentiated Gradient (TPR parity)", "In-processing", "eg"),
                                 ("Threshold Optimizer (equal opportunity)", "Post-processing", "to")):
            try:
                if fn == "rw":
                    p, m = method_reweighing(best, X_tr, y_tr, groups_tr[attr], X_te)
                elif fn == "eg":
                    p, m = method_expgrad(best, X_tr, y_tr, groups_tr[attr], X_te)
                else:
                    p, m = method_threshold(best, X_tr, y_tr, groups_tr[attr], X_te, groups_te[attr])
                models[(attr, fn)] = m
                res.append({"method": label, "stage": stage, **overall(y_te, p), **gaps(y_te, p, groups_te[attr]),
                            "groups": group_table(y_te, p, groups_te[attr])})
                log(f"  fairness [{attr}] {label}: F1 {res[-1]['f1_macro']} High-recall {res[-1]['high_recall']} gap {res[-1]['recall_gap']}")
            except Exception as e:
                res.append({"method": label, "stage": stage, "error": str(e)})
                log(f"  fairness [{attr}] {label} failed: {e}")
        out["mitigation"][attr] = res
    out["ethics_statement"] = ETHICS
    C.FAIRNESS_PATH.write_text(json.dumps(out, indent=2, default=float))
    return out, models


ETHICS = [
    "Predictions support, and never replace, the judgement of teachers and counsellors.",
    "A High-risk flag should trigger an offer of help, not a penalty, label on a transcript, or change to admission or grading.",
    "Sensitive attributes (gender, socio-economic status) are used only to audit fairness and, for post-processing, to equalise recall; they are never shown to staff as a reason for risk.",
    "The training data combine five public datasets from different countries and grading systems; results may not transfer to a new institution without local validation.",
    "Region was planned as a fairness attribute but is not present in the dataset; socio-economic status is audited instead.",
    "Student records are stored with role-based access; predictions are logged for audit and can be deleted on request.",
]
