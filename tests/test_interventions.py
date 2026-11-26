import pandas as pd

from src import interventions


def test_suggestions_follow_risk_factors():
    up = pd.DataFrame([{"label": "Absences", "value": "24", "shap_high": 0.3},
                       {"label": "Weekly study time", "value": "3", "shap_high": 0.2}])
    s = interventions.suggest(up, "High")
    assert list(s["Factor"]) == ["Absences", "Weekly study time", "Overall"]
    assert s.iloc[-1]["Suggested action"] == interventions.GENERAL["High"]


def test_unknown_factor_gives_only_general_advice():
    up = pd.DataFrame([{"label": "Gender", "value": "male", "shap_high": 0.1}])
    s = interventions.suggest(up, "Low")
    assert list(s["Factor"]) == ["Overall"]
