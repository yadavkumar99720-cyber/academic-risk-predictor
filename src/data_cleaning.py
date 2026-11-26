import json
import numpy as np
import pandas as pd

from . import config as C


def _norm_text(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower().replace("_", " ")
    s = " ".join(s.split())
    s = C.CATEGORY_SYNONYMS.get(s, s)
    return s


def load_raw(path=None):
    return pd.read_csv(path or C.RAW_PATH)


def standardize_categories(df, report):
    cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) in ("str", "string", "category")]
    changed = 0
    for c in cat_cols:
        if c in ("source", "grade_letter"):
            df[c] = df[c].astype(str).str.strip()
            continue
        before = df[c].astype(str).copy()
        df[c] = df[c].map(_norm_text)
        changed += int((before != df[c].astype(str)).sum())
    if "grade_letter" in df:
        df["grade_letter"] = df["grade_letter"].str.upper()
    report["category_values_standardized"] = changed
    return df


def remove_duplicates(df, report):
    key = [c for c in df.columns if c != "fact_id"]
    n = int(df.duplicated(subset=key).sum())
    report["duplicates_removed"] = n
    return df.drop_duplicates(subset=key).reset_index(drop=True)


def mask_non_applicable(df, report):
    total = 0
    for src, own in C.SOURCE_FEATURES.items():
        rows = df["source"] == src
        others = [f for s, cols in C.SOURCE_FEATURES.items() if s != src for f in cols]
        for f in others:
            if f in df:
                total += int(df.loc[rows, f].notna().sum())
                df.loc[rows, f] = np.nan
    report["placeholder_cells_marked_not_applicable"] = total
    return df


def range_checks(df, report):
    out = {}
    for c, (lo, hi) in C.VALID_RANGES.items():
        if c not in df:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        bad = (col < lo) | (col > hi)
        if bad.any():
            out[c] = int(bad.sum())
            df.loc[bad, c] = col[bad].clip(lo, hi)
    report["out_of_range_values_clipped"] = out
    return df


def outlier_report(df, report):
    out = {}
    for c in C.NUMERIC_FEATURES:
        if c not in df:
            continue
        for src, g in df.groupby("source"):
            s = g[c].dropna()
            if s.nunique() < 5:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            n = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
            if n:
                out[f"{c} [{src}]"] = n
    report["extreme_outliers_flagged_iqr3"] = out
    return df


def impute_missing(df, report):
    filled = 0
    for src, own in C.SOURCE_FEATURES.items():
        rows = df["source"] == src
        for f in own + ["gender", "age"]:
            if f not in df:
                continue
            miss = rows & df[f].isna()
            if not miss.any():
                continue
            if f in C.NUMERIC_FEATURES:
                val = df.loc[rows, f].median()
            else:
                val = df.loc[rows, f].mode().iloc[0]
            df.loc[miss, f] = val
            filled += int(miss.sum())
    report["missing_values_imputed"] = filled
    before = len(df)
    df = df[df["grade_letter"].isin(C.GRADE_TO_RISK)].reset_index(drop=True)
    report["rows_dropped_missing_label"] = before - len(df)
    return df


def add_derived(df):
    df[C.TARGET] = df["grade_letter"].map(C.GRADE_TO_RISK)
    ses = pd.Series("Unknown", index=df.index)
    if "family_income" in df:
        fi = df["family_income"]
        ses = ses.mask(fi.eq("low"), "Low SES").mask(fi.isin(["medium", "high"]), "Not low SES")
    if "lunch" in df:
        lu = df["lunch"]
        ses = ses.mask(lu.eq("free/reduced"), "Low SES").mask(lu.eq("standard"), "Not low SES")
    df["ses_group"] = ses
    df["age_group"] = np.where(pd.to_numeric(df["age"], errors="coerce") < 15, "Under 15", "15 and above")
    return df


def clean(df):
    report = {"rows_in": int(len(df)), "columns_in": int(df.shape[1])}
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = standardize_categories(df, report)
    df = remove_duplicates(df, report)
    df = mask_non_applicable(df, report)
    df = range_checks(df, report)
    df = outlier_report(df, report)
    df = impute_missing(df, report)
    df = add_derived(df)
    report["rows_out"] = int(len(df))
    report["risk_distribution"] = df[C.TARGET].value_counts().to_dict()
    report["rows_by_source"] = df["source"].value_counts().to_dict()
    return df, report


def validate_schema(df):
    required = ["source", "gender", "age", "grade_letter"]
    missing = [c for c in required if c not in df.columns]
    unknown_sources = sorted(set(df["source"].astype(str).str.strip()) - set(C.SOURCE_FEATURES)) if "source" in df else []
    return {"missing_columns": missing, "unknown_sources": unknown_sources, "ok": not missing and not unknown_sources}


def run(path=None, save=True):
    raw = load_raw(path)
    df, report = clean(raw)
    if save:
        C.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(C.CLEAN_PATH, index=False)
        C.CLEAN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        C.CLEAN_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return df, report
