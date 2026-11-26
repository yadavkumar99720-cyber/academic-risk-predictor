import numpy as np
import pandas as pd

from . import config as C

TOPICS = ["arabic", "biology", "chemistry", "english", "french", "geology", "history", "it", "math", "quran", "science", "spanish"]
SOURCES = list(C.SOURCE_FEATURES)

PRETTY = {
    "age": "Age", "gender": "Gender", "raisedhands": "Times raised hand", "visited_resources": "Resources visited",
    "announcements_view": "Announcements viewed", "discussion": "Discussion posts", "absence_days": "Absence days",
    "stage": "School stage", "absences": "Absences", "study_time": "Weekly study time", "parental_education": "Parental education",
    "parental_support": "Parental support", "tutoring": "Tutoring (0 = no, 1 = yes)", "G1": "Period-1 grade", "G2": "Period-2 grade",
    "failures": "Past failures", "Medu": "Mother's education", "Fedu": "Father's education", "lunch": "Lunch type",
    "test_prep": "Test preparation", "attendance": "Attendance %", "hours_studied": "Hours studied per week",
    "motivation_level": "Motivation level", "internet_access": "Internet access", "tutoring_sessions": "Tutoring sessions",
    "parental_involvement": "Parental involvement", "family_income": "Family income", "previous_scores": "Previous scores",
    "sleep_hours": "Sleep hours", "topic": "Course topic",
}


def encode(df):
    out = pd.DataFrame(index=df.index)
    for f in C.NUMERIC_FEATURES:
        out[f] = pd.to_numeric(df[f], errors="coerce") if f in df else np.nan
    for f, levels in C.ORDINAL_FEATURES.items():
        m = {v: i for i, v in enumerate(levels)}
        out[f] = df[f].map(m).astype(float) if f in df else np.nan
    src = df["source"].astype(str)
    for s in SOURCES:
        out[f"source={s}"] = (src == s).astype(float)
    topic = df["topic"] if "topic" in df else pd.Series(np.nan, index=df.index)
    is_x = src == "xAPI"
    for t in TOPICS:
        col = (topic == t).astype(float)
        out[f"topic={t}"] = col.where(is_x, np.nan)
    return out


def feature_names():
    return list(encode(_dummy()).columns)


def pretty_name(col):
    if col.startswith("source="):
        return "Record type: " + C.SOURCE_LABELS.get(col.split("=", 1)[1], col.split("=", 1)[1])
    if col.startswith("topic="):
        return "Course topic: " + col.split("=", 1)[1]
    return PRETTY.get(col, col)


def _dummy():
    row = {f: np.nan for f in C.FEATURES}
    row.update({"source": "xAPI", "gender": "male", "age": 12, "topic": "it"})
    return pd.DataFrame([row])


def blank_record(source):
    row = {f: np.nan for f in C.FEATURES}
    row["source"] = source
    return row
