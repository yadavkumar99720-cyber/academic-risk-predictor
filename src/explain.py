import json
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from . import config as C
from .features import pretty_name

HIGH = 2


def _parts(model):
    steps = model.named_steps
    pre = None
    if "impute" in steps or "scale" in steps:
        from sklearn.pipeline import Pipeline
        pre = Pipeline(list(model.steps[:-1]))
    return pre, steps["clf"]


def _transform(model, X):
    pre, _ = _parts(model)
    return pre.transform(X) if pre is not None else X.values


def make_explainer(model, background=None):
    _, clf = _parts(model)
    name = type(clf).__name__
    if name in ("RandomForestClassifier", "HistGradientBoostingClassifier"):
        try:
            return shap.TreeExplainer(clf)
        except Exception:
            pass
    if name == "LogisticRegression":
        bg = _transform(model, background) if background is not None else None
        return shap.LinearExplainer(clf, bg)
    bg = shap.sample(_transform(model, background), 100, random_state=C.RANDOM_STATE)
    return shap.Explainer(clf.predict_proba, bg)


def shap_values(explainer, model, X):
    Z = _transform(model, X)
    sv = explainer.shap_values(Z) if hasattr(explainer, "shap_values") else explainer(Z).values
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 2:
        sv = sv[:, :, None]
    return sv


def base_values(explainer):
    ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    return ev


def global_importance(sv, cols):
    imp = np.abs(sv).mean(axis=(0, 2))
    high = sv[:, :, HIGH].mean(axis=0) if sv.shape[2] > HIGH else sv[:, :, 0].mean(axis=0)
    df = pd.DataFrame({"feature": cols, "label": [pretty_name(c) for c in cols], "mean_abs_shap": imp,
                       "mean_shap_high": high, "mean_abs_shap_high": np.abs(sv[:, :, min(HIGH, sv.shape[2] - 1)]).mean(axis=0)})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_consistency(sv, cols, groups):
    out = {}
    for attr, g in groups.items():
        vals = [v for v in sorted(pd.Series(g).dropna().unique()) if v != "Unknown"]
        if len(vals) < 2:
            continue
        vecs = {v: np.abs(sv[np.asarray(g) == v, :, HIGH]).mean(axis=0) for v in vals}
        a, b = vals[0], vals[1]
        rho = spearmanr(vecs[a], vecs[b]).statistic
        top_a = [cols[i] for i in np.argsort(-vecs[a])[:5]]
        top_b = [cols[i] for i in np.argsort(-vecs[b])[:5]]
        out[attr] = {"groups": [a, b], "spearman_rho": round(float(rho), 4),
                     "top5_overlap": len(set(top_a) & set(top_b)),
                     "top5": {a: [pretty_name(c) for c in top_a], b: [pretty_name(c) for c in top_b]}}
    return out


def explain_one(explainer, model, x_row, cols, top_k=6, display=None):
    sv = shap_values(explainer, model, x_row)[0]
    proba = model.predict_proba(x_row)[0]
    k = int(np.argmax(proba))
    contrib = sv[:, k] if sv.shape[1] > k else sv[:, 0]
    high = sv[:, HIGH] if sv.shape[1] > HIGH else sv[:, 0]
    raw = x_row.iloc[0]
    rows = []
    for i, c in enumerate(cols):
        if pd.isna(raw[c]):
            continue
        if "=" in c and raw[c] == 0:
            continue
        val = display.get(c, raw[c]) if display else raw[c]
        rows.append({"feature": c, "label": pretty_name(c), "value": _fmt(val), "shap_pred": float(contrib[i]),
                     "shap_high": float(high[i])})
    df = pd.DataFrame(rows)
    df = df.reindex(df["shap_high"].abs().sort_values(ascending=False).index)
    up = df[df["shap_high"] > 0].head(top_k)
    down = df[df["shap_high"] < 0].head(top_k)
    return {"pred_class": C.RISK_CLASSES[k], "proba": proba, "table": df.reset_index(drop=True),
            "risk_increasing": up, "risk_reducing": down, "text": narrative(C.RISK_CLASSES[k], proba, up, down)}


def _fmt(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:.2f}"
    except Exception:
        return str(v)


def narrative(cls, proba, up, down):
    conf = proba.max() * 100
    s = [f"The model places this student in the {cls} risk group with {conf:.0f}% confidence."]
    if len(up):
        items = ", ".join(f"{r.label} ({_fmt(r.value)})" for r in up.head(3).itertuples())
        s.append(f"The factors pushing the risk up most are {items}.")
    if len(down):
        items = ", ".join(f"{r.label} ({_fmt(r.value)})" for r in down.head(3).itertuples())
        s.append(f"The factors pulling the risk down most are {items}.")
    if cls == "High":
        s.append("A counsellor review is recommended; the explanation shows where support is likely to help.")
    elif cls == "Medium":
        s.append("Monitoring and light-touch support are suggested.")
    else:
        s.append("No intervention is indicated, but the prediction should be rechecked as new data arrives.")
    return " ".join(s)


def run(model, X_te, groups_te, cols, n=1500):
    Xs = X_te.sample(min(n, len(X_te)), random_state=C.RANDOM_STATE)
    g = {k: pd.Series(v, index=X_te.index).loc[Xs.index].values for k, v in groups_te.items()}
    ex = make_explainer(model, X_te)
    sv = shap_values(ex, model, Xs)
    imp = global_importance(sv, cols)
    cons = group_consistency(sv, cols, g)
    out = {"n_explained": int(len(Xs)), "importance": imp.to_dict(orient="records"), "group_consistency": cons}
    C.SHAP_PATH.write_text(json.dumps(out, indent=2, default=float))
    return out, sv, Xs
