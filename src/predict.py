import io
import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import explain
from .data_cleaning import _norm_text
from .features import encode


class RiskService:
    def __init__(self, model_path=None, background=None):
        self.model = joblib.load(model_path or (C.MODEL_DIR / "best_model.joblib"))
        self.background = background
        self.explainer = explain.make_explainer(self.model, background)
        self.columns = None

    def prepare(self, df):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        if "source" not in df:
            raise ValueError("The file needs a 'source' column (one of: " + ", ".join(C.SOURCE_FEATURES) + ").")
        df["source"] = df["source"].astype(str).str.strip()
        bad = sorted(set(df["source"]) - set(C.SOURCE_FEATURES))
        if bad:
            raise ValueError("Unknown source value(s): " + ", ".join(bad))
        for f in C.FEATURES:
            if f not in df:
                df[f] = np.nan
        for f in set(C.ORDINAL_FEATURES) | {"topic"}:
            df[f] = df[f].map(_norm_text)
        for f in C.NUMERIC_FEATURES:
            df[f] = pd.to_numeric(df[f], errors="coerce")
            if f in C.VALID_RANGES:
                lo, hi = C.VALID_RANGES[f]
                df[f] = df[f].clip(lo, hi)
        for src, own in C.SOURCE_FEATURES.items():
            rows = df["source"] == src
            others = [f for s, cols in C.SOURCE_FEATURES.items() if s != src for f in cols]
            df.loc[rows, others] = np.nan
        return df

    def predict_frame(self, df):
        prep = self.prepare(df)
        X = encode(prep)
        proba = self.model.predict_proba(X)
        out = df.copy()
        out["predicted_risk"] = [C.RISK_CLASSES[i] for i in proba.argmax(1)]
        out["confidence"] = proba.max(1).round(4)
        for i, c in enumerate(C.RISK_CLASSES):
            out[f"prob_{c.lower()}"] = proba[:, i].round(4)
        sv = explain.shap_values(self.explainer, self.model, X)
        high = sv[:, :, min(2, sv.shape[2] - 1)]
        tops = []
        for r in range(len(X)):
            order = np.argsort(-high[r])
            valid = [j for j in order if high[r, j] > 0 and not pd.isna(X.iloc[r, j]) and not ("=" in X.columns[j] and X.iloc[r, j] == 0)]
            items = [explain.pretty_name(X.columns[j]) for j in valid[:3]]
            tops.append("; ".join(items))
        out["top_risk_factors"] = tops
        return out, X

    def explain_record(self, record):
        df = self.prepare(pd.DataFrame([record]))
        X = encode(df)
        r = df.iloc[0]
        display = {c: ("yes" if "=" in c else r[c]) for c in X.columns if "=" in c or c in r.index}
        return explain.explain_one(self.explainer, self.model, X, list(X.columns), display=display), X


def template_csv():
    rows = []
    ex = {
        "xAPI": {"gender": "male", "age": 13, "raisedhands": 45, "visited_resources": 60, "announcements_view": 5, "discussion": 20,
                 "absence_days": "under-7", "stage": "middleschool", "topic": "math"},
        "StudentPerf": {"gender": "female", "age": 16, "absences": 18, "study_time": 4.5, "parental_education": 1, "parental_support": 1, "tutoring": 0},
        "StudentGrades": {"gender": "male", "age": 17, "G1": 15, "G2": 16, "failures": 0, "Medu": 4, "Fedu": 3},
        "PerfScores": {"gender": "female", "age": 16, "lunch": "free/reduced", "test_prep": "none"},
        "StudentFactors": {"gender": "male", "age": 16, "attendance": 65, "hours_studied": 10, "motivation_level": "low", "internet_access": "yes",
                           "tutoring_sessions": 0, "parental_involvement": "low", "family_income": "low", "previous_scores": 58, "sleep_hours": 6},
    }
    for i, (src, vals) in enumerate(ex.items(), 1):
        r = {"student_ref": f"S{i:03d}", "source": src}
        r.update(vals)
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["student_ref", "source", "gender", "age"] + [f for v in C.SOURCE_FEATURES.values() for f in v]
    buf = io.StringIO()
    df.reindex(columns=cols).to_csv(buf, index=False)
    return buf.getvalue()
