import pytest
from streamlit.testing.v1 import AppTest

from src import config as C

pytestmark = pytest.mark.skipif(not (C.MODEL_DIR / "best_model.joblib").exists(), reason="run run_pipeline.py first")
PAGES = ["Dashboard", "Risk Prediction", "Batch Prediction", "Prediction History", "Data Management",
         "Exploratory Data Analysis", "Model Evaluation", "Explainability", "Fairness Audit", "User Management",
         "Audit Log", "My Account"]


def test_login_rejects_bad_password():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    at.text_input[0].input("admin")
    at.text_input[1].input("nope")
    at.button[0].click().run()
    assert at.error and "Invalid" in at.error[0].value


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.session_state["user"] = {"user_id": 1, "username": "admin", "full_name": "System Administrator", "role": "admin"}
    at.session_state["start_page"] = page
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


def test_single_prediction_flow():
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.session_state["user"] = {"user_id": 1, "username": "admin", "full_name": "System Administrator", "role": "admin"}
    at.session_state["start_page"] = "Risk Prediction"
    at.run()
    next(b for b in at.button if b.label == "Predict risk").click().run()
    assert not at.exception
    assert any("Saved as prediction" in c.value for c in at.caption)
    assert any(h.value == "Suggested interventions" for h in at.subheader)


def test_teacher_cannot_see_admin_pages():
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.session_state["user"] = {"user_id": 1, "username": "t", "full_name": "Teacher", "role": "teacher"}
    at.session_state["start_page"] = "User Management"
    at.run()
    assert not at.exception
    assert not any(t.value == "User Management" for t in at.title)
