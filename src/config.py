import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "Academic_Risk_Prediction_DW.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
FIG_DIR = ARTIFACTS / "figures"
METRICS_PATH = ARTIFACTS / "metrics.json"
FAIRNESS_PATH = ARTIFACTS / "fairness.json"
SHAP_PATH = ARTIFACTS / "shap_global.json"
CLEAN_REPORT_PATH = ARTIFACTS / "cleaning_report.json"
DB_PATH = Path(os.environ.get("ACADEMIC_RISK_DB", ROOT / "data" / "academic_risk.db"))


def setting(name, default=None):
    """Read a setting from an environment variable, else from Streamlit secrets (.streamlit/secrets.toml or the
    Secrets box on Streamlit Community Cloud), else return the default."""
    if name in os.environ:
        return os.environ[name]
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


# Database: SQLite file by default. Set DATABASE_URL to use MySQL instead, for example
#   mysql://user:password@localhost:3306/academic_risk
#   mysql://user.root:password@gateway01.ap-southeast-1.prod.aws.tidbcloud.com:4000/academic_risk?ssl=true
DATABASE_URL = (setting("DATABASE_URL", "") or "").strip()
# Password of the first 'admin' account, used only when the database is created. Set it in secrets for a public deployment.
ADMIN_PASSWORD = setting("ADMIN_PASSWORD", "Admin@123") or "Admin@123"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "risk_level"
RISK_CLASSES = ["Low", "Medium", "High"]
GRADE_TO_RISK = {"A": "Low", "B": "Low", "C": "Medium", "D": "High"}

LEAKAGE_COLUMNS = ["grade_letter", "G3", "GPA", "exam_score", "avg_score", "math_score", "science_score", "english_score"]

SOURCE_LABELS = {
    "xAPI": "LMS engagement records",
    "StudentPerf": "GPA and study-habit records",
    "StudentGrades": "Secondary-school period grades",
    "PerfScores": "Standardized-test background",
    "StudentFactors": "Student-factors survey",
}

SOURCE_FEATURES = {
    "xAPI": ["raisedhands", "visited_resources", "announcements_view", "discussion", "absence_days", "stage", "topic"],
    "StudentPerf": ["absences", "study_time", "parental_education", "parental_support", "tutoring"],
    "StudentGrades": ["G1", "G2", "failures", "Medu", "Fedu"],
    "PerfScores": ["lunch", "test_prep"],
    "StudentFactors": ["attendance", "hours_studied", "motivation_level", "internet_access", "tutoring_sessions",
                       "parental_involvement", "family_income", "previous_scores", "sleep_hours"],
}

COMMON_FEATURES = ["source", "gender", "age"]

NUMERIC_FEATURES = ["age", "raisedhands", "visited_resources", "announcements_view", "discussion",
                    "absences", "study_time", "parental_education", "parental_support", "tutoring",
                    "G1", "G2", "failures", "Medu", "Fedu",
                    "attendance", "hours_studied", "tutoring_sessions", "previous_scores", "sleep_hours"]

ORDINAL_FEATURES = {
    "absence_days": ["under-7", "above-7"],
    "stage": ["lowerlevel", "middleschool", "highschool"],
    "motivation_level": ["low", "medium", "high"],
    "parental_involvement": ["low", "medium", "high"],
    "family_income": ["low", "medium", "high"],
    "internet_access": ["no", "yes"],
    "lunch": ["free/reduced", "standard"],
    "test_prep": ["none", "completed"],
    "gender": ["female", "male"],
}

NOMINAL_FEATURES = ["source", "topic"]

FEATURES = COMMON_FEATURES + [f for cols in SOURCE_FEATURES.values() for f in cols]

VALID_RANGES = {
    "age": (5, 30), "raisedhands": (0, 100), "visited_resources": (0, 100), "announcements_view": (0, 100),
    "discussion": (0, 100), "absences": (0, 60), "study_time": (0, 40), "parental_education": (0, 4),
    "parental_support": (0, 4), "tutoring": (0, 1), "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
    "failures": (0, 4), "Medu": (0, 4), "Fedu": (0, 4), "attendance": (0, 100), "hours_studied": (0, 60),
    "tutoring_sessions": (0, 10), "previous_scores": (0, 100), "sleep_hours": (0, 14), "exam_score": (0, 100),
    "math_score": (0, 100), "science_score": (0, 100), "english_score": (0, 100), "avg_score": (0, 100), "GPA": (0, 4),
}

CATEGORY_SYNONYMS = {
    "m": "male", "f": "female", "boy": "male", "girl": "female",
    "y": "yes", "n": "no", "true": "yes", "false": "no",
    "med": "medium", "mid": "medium", "lo": "low", "hi": "high",
    "free reduced": "free/reduced", "free-reduced": "free/reduced",
    "under 7": "under-7", "above 7": "above-7", "lower level": "lowerlevel", "middle school": "middleschool",
    "high school": "highschool", "complete": "completed",
}

SENSITIVE_ATTRIBUTES = ["gender", "ses_group", "age_group"]
