import numpy as np
import pandas as pd

from src import config as C
from src import data_cleaning as dc


def raw_sample():
    return dc.load_raw().groupby("source").head(40).reset_index(drop=True)


def test_duplicates_removed():
    df = raw_sample()
    df2 = pd.concat([df, df.iloc[:5].assign(fact_id=df.fact_id.max() + np.arange(1, 6))], ignore_index=True)
    out, rep = dc.clean(df2)
    assert rep["duplicates_removed"] >= 5


def test_non_applicable_fields_are_empty():
    out, _ = dc.clean(raw_sample())
    x = out[out.source == "xAPI"]
    assert x["G1"].isna().all()
    assert x["attendance"].isna().all()
    assert x["raisedhands"].notna().all()


def test_risk_label_mapping():
    out, _ = dc.clean(raw_sample())
    for g, r in C.GRADE_TO_RISK.items():
        assert (out.loc[out.grade_letter == g, C.TARGET] == r).all()


def test_category_standardisation_and_ranges():
    df = raw_sample()
    df.loc[0, "gender"] = " Male "
    df.loc[1, "gender"] = "F"
    df.loc[df.source == "StudentFactors", "exam_score"] = 150
    out, rep = dc.clean(df)
    assert set(out["gender"]) <= {"male", "female"}
    assert out["exam_score"].max() <= 100
    assert rep["out_of_range_values_clipped"]["exam_score"] > 0


def test_missing_values_imputed_within_source():
    df = raw_sample()
    idx = df.index[df.source == "StudentPerf"][:3]
    df.loc[idx, "absences"] = np.nan
    out, rep = dc.clean(df)
    assert out.loc[out.source == "StudentPerf", "absences"].notna().all()
    assert rep["missing_values_imputed"] >= 3


def test_schema_validation():
    assert dc.validate_schema(raw_sample())["ok"]
    assert not dc.validate_schema(raw_sample().drop(columns=["source"]).assign(source="Other"))["ok"]
