"""MySQL backend tests.

The first two tests always run; they check how a DATABASE_URL is read and how MySQL rows behave.
The database tests from test_database.py run a second time against a real MySQL server when the
environment variable TEST_MYSQL_URL is set, for example (the database must exist and will be emptied):

    set TEST_MYSQL_URL=mysql://root:password@localhost:3306/academic_risk_test      (Windows)
    export TEST_MYSQL_URL=mysql://root:password@localhost:3306/academic_risk_test   (macOS / Linux)
    python -m pytest -q tests/test_mysql.py
"""
import os

import pytest

from src import database as db

URL = os.environ.get("TEST_MYSQL_URL")


def test_parse_mysql_url():
    k = db.parse_mysql_url("mysql://bharath:p%40ss@db.example.com:4000/academic_risk?ssl=true&ssl_ca=/tmp/ca.pem")
    assert (k["host"], k["port"], k["user"], k["password"], k["database"]) == ("db.example.com", 4000, "bharath", "p@ss", "academic_risk")
    assert k["ssl_ca"] == "/tmp/ca.pem" and k["ssl_verify_cert"]
    k = db.parse_mysql_url("mysql://root@localhost/x")
    assert k["port"] == 3306 and k["password"] == "" and "ssl_ca" not in k
    assert db.is_mysql("mysql://root@localhost/x") and not db.is_mysql("/tmp/a.db")
    assert db.describe("mysql://root@localhost/x") == "MySQL (localhost:3306/x)"


def test_mysql_row_works_like_sqlite_row():
    r = db._Row(["user_id", "role"], (7, "admin"))
    assert r[0] == 7 and r["role"] == "admin" and dict(r) == {"user_id": 7, "role": "admin"}


needs_mysql = pytest.mark.skipif(not URL, reason="set TEST_MYSQL_URL to run the database tests on MySQL")


@pytest.fixture
def path():
    with db.connect(URL) as con:
        con.execute("SET FOREIGN_KEY_CHECKS = 0")
        for t in db.TABLES:
            con.execute(f"DROP TABLE IF EXISTS {t}")
        con.execute("SET FOREIGN_KEY_CHECKS = 1")
    db.init_db(URL)
    return URL


# The same six database tests as test_database.py, run on MySQL with the fixture above.
from tests import test_database as td  # noqa: E402


@needs_mysql
def test_mysql_default_admin_and_login(path):
    td.test_default_admin_and_login(path)


@needs_mysql
def test_mysql_create_user_and_password_rules(path):
    td.test_create_user_and_password_rules(path)


@needs_mysql
def test_mysql_role_constraint(path):
    td.test_role_constraint(path)


@needs_mysql
def test_mysql_save_history_and_delete(path):
    td.test_save_history_and_delete(path)


@needs_mysql
def test_mysql_dashboard_counts_and_alerts(path):
    td.test_dashboard_counts_and_alerts(path)


@needs_mysql
def test_mysql_audit_filter(path):
    td.test_audit_filter(path)
