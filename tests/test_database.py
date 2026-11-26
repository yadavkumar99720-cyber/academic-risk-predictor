import numpy as np
import pytest

from src import database as db


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "t.db"
    db.init_db(p)
    return p


def test_default_admin_and_login(path):
    assert db.authenticate("admin", "Admin@123", path)["role"] == "admin"
    assert db.authenticate("admin", "wrong", path) is None


def test_create_user_and_password_rules(path):
    uid = db.create_user("Teacher1", "T One", "teacher", "secret123", path=path)
    assert db.authenticate("teacher1", "secret123", path)["user_id"] == uid
    with pytest.raises(ValueError):
        db.create_user("x", "X", "teacher", "short", path=path)
    db.change_password(uid, "newsecret1", path)
    assert db.authenticate("teacher1", "newsecret1", path)
    db.set_active(uid, False, path)
    assert db.authenticate("teacher1", "newsecret1", path) is None


def test_role_constraint(path):
    with pytest.raises(Exception):
        db.create_user("bad", "Bad", "student", "password1", path=path)


def test_save_history_and_delete(path):
    rec = {"source": "StudentPerf", "gender": "female", "age": 16, "absences": 20}
    res = {"pred_class": "High", "proba": np.array([0.1, 0.2, 0.7]), "text": "x"}
    fac = [{"label": "Absences", "value": 20, "shap_high": 0.3}]
    pid = db.save_prediction(1, rec, res, fac, path=path)
    h = db.history(path=path)
    assert h[0]["prediction_id"] == pid and h[0]["predicted_risk"] == "High"
    assert db.factors_for(pid, path)[0]["direction"] == "increases risk"
    db.delete_prediction(pid, path)
    assert db.history(path=path) == []
    assert db.factors_for(pid, path) == []


def test_dashboard_counts_and_alerts(path):
    tid = db.create_user("teach2", "T Two", "teacher", "secret123", path=path)
    for uid, risk, ph in [(1, "High", 0.7), (1, "Low", 0.1), (tid, "High", 0.9)]:
        res = {"pred_class": risk, "proba": np.array([1 - ph - 0.1, 0.1, ph]), "text": "x"}
        db.save_prediction(uid, {"source": "StudentPerf", "gender": "male", "age": 16}, res, path=path)
    s = db.stats(path=path)
    assert s["predictions"] == 3 and s["high"] == 2 and s["by_risk"]["Low"] == 1
    assert db.stats(tid, path=path)["predictions"] == 1
    al = db.early_alerts(path=path)
    assert [a["prob_high"] for a in al] == [0.9, 0.7]
    assert len(db.early_alerts(tid, path=path)) == 1
    assert sum(r["n"] for r in db.daily_counts(path=path)) == 3


def test_audit_filter(path):
    db.log(1, "login", path=path)
    db.log(None, "failed_login", {"username": "x"}, path=path)
    assert [r["action"] for r in db.audit(action="failed_login", path=path)] == ["failed_login"]
    assert all(r["username"] == "admin" for r in db.audit(username="admin", path=path))
    assert set(db.audit_actions(path=path)) == {"login", "failed_login"}
