import io
import json

import pandas as pd
import pytest

from src import config as C
from src.features import encode
from src.predict import RiskService, template_csv

pytestmark = pytest.mark.skipif(not (C.MODEL_DIR / "best_model.joblib").exists(), reason="run run_pipeline.py first")


@pytest.fixture(scope="module")
def svc():
    df = pd.read_csv(C.CLEAN_PATH)
    return RiskService(background=encode(df.sample(200, random_state=0)))


def test_metrics_beat_baseline():
    m = json.loads(C.METRICS_PATH.read_text())
    best = m["models"][m["best_model"]]
    assert best["test"]["f1_macro"] > m["models"]["Majority-class baseline"]["test"]["f1_macro"] + 0.3
    assert best["cv"]["f1_macro"]["low"] <= best["cv"]["f1_macro"]["mean"] <= best["cv"]["f1_macro"]["high"]


def test_no_leakage_columns_in_features():
    m = json.loads(C.METRICS_PATH.read_text())
    assert not set(m["feature_columns"]) & set(C.LEAKAGE_COLUMNS)


def test_batch_prediction(svc):
    out, X = svc.predict_frame(pd.read_csv(io.StringIO(template_csv())))
    assert len(out) == 5
    assert set(out["predicted_risk"]) <= set(C.RISK_CLASSES)
    assert ((out[["prob_low", "prob_medium", "prob_high"]].sum(axis=1) - 1).abs() < 1e-3).all()


def test_single_explanation(svc):
    rec = {"source": "StudentGrades", "gender": "male", "age": 17, "G1": 5, "G2": 4, "failures": 2, "Medu": 1, "Fedu": 1}
    res, _ = svc.explain_record(rec)
    assert res["pred_class"] == "High"
    assert "Period-2 grade" in set(res["risk_increasing"]["label"])
    assert res["text"].startswith("The model places")


def test_bad_source_rejected(svc):
    with pytest.raises(ValueError):
        svc.predict_frame(pd.DataFrame([{"source": "Unknown", "gender": "male"}]))


def test_monotonic_sanity(svc):
    good = {"source": "StudentPerf", "gender": "female", "age": 16, "absences": 1, "study_time": 15, "parental_education": 3, "parental_support": 3, "tutoring": 1}
    bad = dict(good, absences=28, study_time=1, parental_support=0, tutoring=0)
    pg = svc.explain_record(good)[0]["proba"][2]
    pb = svc.explain_record(bad)[0]["proba"][2]
    assert pb > pg
