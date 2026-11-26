import numpy as np
import pandas as pd
import plotly.express as px

from . import config as C

RISK_COLORS = {"Low": "#2e7d5b", "Medium": "#e0a526", "High": "#c0392b"}
ORDER = {C.TARGET: C.RISK_CLASSES}


def _s(df, src):
    return df[df["source"] == src]


def class_distribution(df):
    t = df.groupby(["source", C.TARGET]).size().reset_index(name="students")
    return px.bar(t, x="source", y="students", color=C.TARGET, color_discrete_map=RISK_COLORS, category_orders=ORDER,
                  title="Risk-category distribution by record type", barmode="stack")


def overall_distribution(df):
    t = df[C.TARGET].value_counts().reindex(C.RISK_CLASSES).reset_index()
    t.columns = ["risk", "students"]
    return px.bar(t, x="risk", y="students", color="risk", color_discrete_map=RISK_COLORS, title="Overall risk-category distribution")


def attendance_vs_risk(df):
    d = _s(df, "StudentPerf")
    return px.box(d, x=C.TARGET, y="absences", color=C.TARGET, color_discrete_map=RISK_COLORS, category_orders=ORDER,
                  title="Absences vs risk category (GPA and study-habit records)")


def attendance_pct_vs_risk(df):
    d = _s(df, "StudentFactors")
    return px.box(d, x=C.TARGET, y="attendance", color=C.TARGET, color_discrete_map=RISK_COLORS, category_orders=ORDER,
                  title="Attendance % vs risk category (student-factors survey)")


def prev_grade_vs_risk(df):
    d = _s(df, "StudentGrades")
    return px.box(d, x=C.TARGET, y="G2", color=C.TARGET, color_discrete_map=RISK_COLORS, category_orders=ORDER,
                  title="Previous-period grade (G2, out of 20) vs risk category")


def study_hours_vs_score(df):
    d = _s(df, "StudentFactors")
    return px.scatter(d, x="hours_studied", y="exam_score", color=C.TARGET, color_discrete_map=RISK_COLORS, opacity=0.5,
                      category_orders=ORDER, title="Study hours per week vs final exam score")


def completion_vs_score(df):
    d = _s(df, "PerfScores")
    return px.box(d, x="test_prep", y="avg_score", color="test_prep", title="Test-preparation completion vs final average score")


def engagement_vs_risk(df):
    d = _s(df, "xAPI")
    return px.box(d, x=C.TARGET, y="visited_resources", color=C.TARGET, color_discrete_map=RISK_COLORS, category_orders=ORDER,
                  title="LMS resources visited vs risk category")


def failures_vs_risk(df):
    d = _s(df, "StudentGrades")
    t = d.groupby("failures")[C.TARGET].value_counts(normalize=True).rename("share").reset_index()
    return px.bar(t, x="failures", y="share", color=C.TARGET, color_discrete_map=RISK_COLORS, category_orders=ORDER,
                  title="Past failures vs risk category (share of students)")


def internet_income(df):
    d = _s(df, "StudentFactors").copy()
    t = pd.concat([
        d.groupby("internet_access")[C.TARGET].apply(lambda s: (s == "High").mean()).rename("high_risk_rate").reset_index().assign(factor="Internet access").rename(columns={"internet_access": "level"}),
        d.groupby("family_income")[C.TARGET].apply(lambda s: (s == "High").mean()).rename("high_risk_rate").reset_index().assign(factor="Family income").rename(columns={"family_income": "level"}),
    ])
    t2 = pd.concat([
        d.groupby("internet_access")["exam_score"].mean().rename("mean_exam").reset_index().assign(factor="Internet access").rename(columns={"internet_access": "level"}),
        d.groupby("family_income")["exam_score"].mean().rename("mean_exam").reset_index().assign(factor="Family income").rename(columns={"family_income": "level"}),
    ])
    t = t.merge(t2, on=["factor", "level"])
    return px.bar(t, x="level", y="mean_exam", facet_col="factor", color="factor", text=t["high_risk_rate"].map(lambda v: f"High-risk {v:.1%}"),
                  title="Internet access and family income: mean exam score (label = High-risk rate)",
                  labels={"level": "", "mean_exam": "mean exam score"}).update_xaxes(matches=None).for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])).update_traces(textposition="outside", textfont_size=11).update_layout(showlegend=False)


def correlation_heatmap(df, source="StudentFactors"):
    d = _s(df, source)
    cols = [c for c in C.SOURCE_FEATURES[source] + ["age", "G3", "GPA", "exam_score", "avg_score", "math_score", "science_score", "english_score"]
            if c in d and pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 1]
    corr = d[cols].corr().round(2)
    return px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title=f"Correlation heatmap - {C.SOURCE_LABELS.get(source, source)}")


ALL = [
    ("Overall class distribution", overall_distribution),
    ("Class distribution by record type", class_distribution),
    ("Attendance % vs risk", attendance_pct_vs_risk),
    ("Absences vs risk", attendance_vs_risk),
    ("Previous grade vs risk", prev_grade_vs_risk),
    ("Study hours vs final score", study_hours_vs_score),
    ("Coursework / test-prep completion vs final score", completion_vs_score),
    ("LMS engagement vs risk", engagement_vs_risk),
    ("Past failures vs risk", failures_vs_risk),
    ("Internet access and family income", internet_income),
]
