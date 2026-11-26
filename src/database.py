import hashlib
import hmac
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from urllib.parse import parse_qs, unquote, urlparse

from . import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    email TEXT,
    role TEXT NOT NULL CHECK (role IN ('admin', 'counsellor', 'teacher')),
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_ref TEXT,
    source TEXT NOT NULL,
    gender TEXT,
    age INTEGER,
    created_by INTEGER REFERENCES users(user_id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS student_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_value TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    hyperparameters TEXT,
    cv_f1_macro REAL,
    test_f1_macro REAL,
    file_path TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    trained_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    file_name TEXT,
    n_rows INTEGER,
    n_high INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    model_id INTEGER REFERENCES model_versions(model_id),
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    batch_id INTEGER REFERENCES batch_jobs(batch_id),
    predicted_risk TEXT NOT NULL CHECK (predicted_risk IN ('Low', 'Medium', 'High')),
    prob_low REAL, prob_medium REAL, prob_high REAL,
    explanation TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_factors (
    factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_value TEXT,
    shap_value REAL,
    direction TEXT CHECK (direction IN ('increases risk', 'reduces risk'))
);
CREATE TABLE IF NOT EXISTS audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(user_id),
    action TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_student ON predictions(student_id);
CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_feat_student ON student_features(student_id);
"""

# The same 8 tables for MySQL 8 (or a MySQL-compatible server such as TiDB Cloud). MySQL ignores foreign keys written
# next to a column, so they are written as table constraints here; text keys need VARCHAR instead of TEXT.
SCHEMA_MYSQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    full_name VARCHAR(120) NOT NULL,
    email VARCHAR(120),
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'counsellor', 'teacher')),
    password_hash VARCHAR(128) NOT NULL,
    salt VARCHAR(64) NOT NULL,
    is_active TINYINT NOT NULL DEFAULT 1,
    created_at VARCHAR(19) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    external_ref VARCHAR(120),
    source VARCHAR(40) NOT NULL,
    gender VARCHAR(20),
    age INT,
    created_by INT,
    created_at VARCHAR(19) NOT NULL,
    FOREIGN KEY (created_by) REFERENCES users(user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS student_features (
    feature_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    feature_name VARCHAR(64) NOT NULL,
    feature_value VARCHAR(255),
    KEY idx_feat_student (student_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS model_versions (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    algorithm VARCHAR(120) NOT NULL,
    hyperparameters TEXT,
    cv_f1_macro DOUBLE,
    test_f1_macro DOUBLE,
    file_path VARCHAR(255) NOT NULL,
    is_active TINYINT NOT NULL DEFAULT 0,
    trained_at VARCHAR(19) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    file_name VARCHAR(255),
    n_rows INT,
    n_high INT,
    created_at VARCHAR(19) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    model_id INT,
    user_id INT NOT NULL,
    batch_id INT,
    predicted_risk VARCHAR(10) NOT NULL CHECK (predicted_risk IN ('Low', 'Medium', 'High')),
    prob_low DOUBLE, prob_medium DOUBLE, prob_high DOUBLE,
    explanation TEXT,
    created_at VARCHAR(19) NOT NULL,
    KEY idx_pred_student (student_id),
    KEY idx_pred_user (user_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES model_versions(model_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (batch_id) REFERENCES batch_jobs(batch_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS prediction_factors (
    factor_id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    feature_name VARCHAR(120) NOT NULL,
    feature_value VARCHAR(255),
    shap_value DOUBLE,
    direction VARCHAR(20) CHECK (direction IN ('increases risk', 'reduces risk')),
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS audit_log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action VARCHAR(64) NOT NULL,
    details TEXT,
    created_at VARCHAR(19) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

TABLES = ["users", "model_versions", "students", "batch_jobs", "student_features", "predictions", "prediction_factors", "audit_log"]


def now():
    return datetime.now().isoformat(timespec="seconds")


def _target(path=None):
    """Where to connect: an explicit SQLite path or mysql:// URL, else DATABASE_URL, else the SQLite file."""
    return str(path or C.DATABASE_URL or C.DB_PATH)


def is_mysql(path=None):
    return _target(path).startswith("mysql")


def parse_mysql_url(url):
    """mysql://user:password@host:port/database?ssl=true&ssl_ca=/path -> keyword arguments for pymysql.connect."""
    u = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    kw = {"host": u.hostname or "localhost", "port": u.port or 3306, "user": unquote(u.username or "root"),
          "password": unquote(u.password or ""), "database": u.path.lstrip("/") or "academic_risk", "charset": "utf8mb4"}
    if q.get("ssl", "").lower() in ("1", "true", "yes") or "ssl_ca" in q:
        ca = q.get("ssl_ca") or next((p for p in ("/etc/ssl/certs/ca-certificates.crt", "/etc/ssl/cert.pem") if os.path.exists(p)), None)
        if ca:
            kw.update(ssl_ca=ca, ssl_verify_cert=True, ssl_verify_identity=True)
        else:
            kw["ssl"] = {"check_hostname": False}
    return kw


def describe(path=None):
    """Short text for the screens, e.g. 'MySQL (localhost:3306/academic_risk)' or 'SQLite (data/academic_risk.db)'."""
    t = _target(path)
    if t.startswith("mysql"):
        k = parse_mysql_url(t)
        return f"MySQL ({k['host']}:{k['port']}/{k['database']})"
    return f"SQLite ({os.path.relpath(t, C.ROOT) if t.startswith(str(C.ROOT)) else t})"


class _Row(dict):
    """A MySQL result row that, like sqlite3.Row, works with both row['name'] and row[0]."""

    def __init__(self, cols, values):
        super().__init__(zip(cols, values))
        self._values = tuple(values)

    def __getitem__(self, key):
        return self._values[key] if isinstance(key, int) else dict.__getitem__(self, key)


class _Result:
    def __init__(self, cur):
        self._cur = cur
        self.lastrowid = cur.lastrowid
        self._cols = [d[0] for d in cur.description] if cur.description else []

    def fetchone(self):
        r = self._cur.fetchone()
        return None if r is None else _Row(self._cols, r)

    def fetchall(self):
        return [_Row(self._cols, r) for r in self._cur.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


def _py(v):
    """numpy numbers (np.int64, np.float64) -> plain Python numbers, which PyMySQL can send."""
    return v.item() if hasattr(v, "item") and not isinstance(v, (str, bytes)) else v


class _MySQL:
    """Gives a pymysql connection the small part of the sqlite3 interface this module uses, so every query below
    is written once ('?' placeholders are turned into '%s')."""

    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql, args=()):
        cur = self.raw.cursor()
        cur.execute(sql.replace("?", "%s"), tuple(_py(a) for a in args) if args else None)
        return _Result(cur)

    def executemany(self, sql, rows):
        rows = list(rows)
        if rows:
            self.raw.cursor().executemany(sql.replace("?", "%s"), [tuple(_py(a) for a in r) for r in rows])

    def executescript(self, script):
        cur = self.raw.cursor()
        for stmt in script.split(";"):
            if stmt.strip():
                cur.execute(stmt)


@contextmanager
def connect(path=None):
    t = _target(path)
    if t.startswith("mysql"):
        try:
            import pymysql
        except ImportError as e:
            raise RuntimeError("DATABASE_URL points to MySQL, so the PyMySQL package is needed: pip install pymysql cryptography") from e
        raw = pymysql.connect(**parse_mysql_url(t))
        con = _MySQL(raw)
        try:
            yield con
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()
        return
    con = sqlite3.connect(t)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def hash_password(password, salt=None):
    salt = salt or os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()
    return h, salt


def init_db(path=None):
    if not is_mysql(path):
        C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as con:
        con.executescript(SCHEMA_MYSQL if is_mysql(path) else SCHEMA)
        if con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            h, s = hash_password(C.ADMIN_PASSWORD)
            con.execute("INSERT INTO users (username, full_name, email, role, password_hash, salt, created_at) VALUES (?,?,?,?,?,?,?)",
                        ("admin", "System Administrator", None, "admin", h, s, now()))


def create_user(username, full_name, role, password, email=None, path=None):
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    h, s = hash_password(password)
    with connect(path) as con:
        cur = con.execute("INSERT INTO users (username, full_name, email, role, password_hash, salt, created_at) VALUES (?,?,?,?,?,?,?)",
                          (username.strip().lower(), full_name, email, role, h, s, now()))
        return cur.lastrowid


def authenticate(username, password, path=None):
    with connect(path) as con:
        r = con.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username.strip().lower(),)).fetchone()
    if not r:
        return None
    h, _ = hash_password(password, r["salt"])
    return dict(r) if hmac.compare_digest(h, r["password_hash"]) else None


def change_password(user_id, new_password, path=None):
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    h, s = hash_password(new_password)
    with connect(path) as con:
        con.execute("UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?", (h, s, user_id))


def set_active(user_id, active, path=None):
    with connect(path) as con:
        con.execute("UPDATE users SET is_active = ? WHERE user_id = ?", (1 if active else 0, user_id))


def list_users(path=None):
    with connect(path) as con:
        return [dict(r) for r in con.execute("SELECT user_id, username, full_name, email, role, is_active, created_at FROM users ORDER BY user_id")]


def log(user_id, action, details=None, path=None):
    with connect(path) as con:
        con.execute("INSERT INTO audit_log (user_id, action, details, created_at) VALUES (?,?,?,?)",
                    (user_id, action, json.dumps(details, default=str) if details is not None else None, now()))


ALGO = {"Logistic Regression": "sklearn.linear_model.LogisticRegression", "Random Forest": "sklearn.ensemble.RandomForestClassifier",
        "Gradient Boosting": "sklearn.ensemble.HistGradientBoostingClassifier"}


def register_models(metrics, path=None):
    with connect(path) as con:
        con.execute("UPDATE model_versions SET is_active = 0")
        for name, r in metrics["models"].items():
            if name.startswith("Majority"):
                continue
            con.execute("INSERT INTO model_versions (name, algorithm, hyperparameters, cv_f1_macro, test_f1_macro, file_path, is_active, trained_at) VALUES (?,?,?,?,?,?,?,?)",
                        (name, ALGO.get(name, name), json.dumps(r.get("best_params", {})), r["cv"]["f1_macro"]["mean"], r["test"]["f1_macro"],
                         "artifacts/models/" + ("best_model" if name == metrics["best_model"] else name.lower().replace(" ", "_")) + ".joblib",
                         1 if name == metrics["best_model"] else 0, now()))


def active_model_id(path=None):
    with connect(path) as con:
        r = con.execute("SELECT model_id FROM model_versions WHERE is_active = 1").fetchone()
        return r[0] if r else None


def create_batch(user_id, file_name, n_rows, n_high, path=None):
    with connect(path) as con:
        return con.execute("INSERT INTO batch_jobs (user_id, file_name, n_rows, n_high, created_at) VALUES (?,?,?,?,?)",
                           (user_id, file_name, n_rows, n_high, now())).lastrowid


def save_prediction(user_id, record, result, factors=None, batch_id=None, external_ref=None, path=None):
    with connect(path) as con:
        sid = con.execute("INSERT INTO students (external_ref, source, gender, age, created_by, created_at) VALUES (?,?,?,?,?,?)",
                          (external_ref, record.get("source"), record.get("gender"),
                           None if record.get("age") is None else int(float(record.get("age"))), user_id, now())).lastrowid
        con.executemany("INSERT INTO student_features (student_id, feature_name, feature_value) VALUES (?,?,?)",
                        [(sid, k, str(v)) for k, v in record.items() if k not in ("source", "gender", "age") and v is not None and str(v) != "nan"])
        mid = con.execute("SELECT model_id FROM model_versions WHERE is_active = 1").fetchone()
        p = result["proba"]
        pid = con.execute("INSERT INTO predictions (student_id, model_id, user_id, batch_id, predicted_risk, prob_low, prob_medium, prob_high, explanation, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (sid, mid[0] if mid else None, user_id, batch_id, result["pred_class"], float(p[0]), float(p[1]), float(p[2]),
                           result.get("text"), now())).lastrowid
        if factors is not None:
            con.executemany("INSERT INTO prediction_factors (prediction_id, feature_name, feature_value, shap_value, direction) VALUES (?,?,?,?,?)",
                            [(pid, f["label"], str(f["value"]), float(f["shap_high"]), "increases risk" if f["shap_high"] > 0 else "reduces risk")
                             for f in factors])
        return pid


def history(user_id=None, limit=500, path=None):
    q = """SELECT p.prediction_id, p.created_at, u.username, s.external_ref, s.source, s.gender, s.age, p.predicted_risk,
                  ROUND(p.prob_low, 3) AS prob_low, ROUND(p.prob_medium, 3) AS prob_medium, ROUND(p.prob_high, 3) AS prob_high, p.batch_id
           FROM predictions p JOIN students s ON s.student_id = p.student_id JOIN users u ON u.user_id = p.user_id"""
    args = []
    if user_id is not None:
        q += " WHERE p.user_id = ?"
        args.append(user_id)
    q += " ORDER BY p.prediction_id DESC LIMIT ?"
    args.append(limit)
    with connect(path) as con:
        return [dict(r) for r in con.execute(q, args)]


def factors_for(prediction_id, path=None):
    with connect(path) as con:
        return [dict(r) for r in con.execute("SELECT feature_name, feature_value, ROUND(shap_value, 4) AS shap_value, direction FROM prediction_factors WHERE prediction_id = ? ORDER BY ABS(shap_value) DESC", (prediction_id,))]


def delete_prediction(prediction_id, path=None):
    with connect(path) as con:
        r = con.execute("SELECT student_id FROM predictions WHERE prediction_id = ?", (prediction_id,)).fetchone()
        if r:
            con.execute("DELETE FROM students WHERE student_id = ?", (r[0],))


def audit(limit=300, action=None, username=None, path=None):
    q = """SELECT a.log_id, a.created_at, u.username, a.action, a.details FROM audit_log a
           LEFT JOIN users u ON u.user_id = a.user_id WHERE 1 = 1"""
    args = []
    if action:
        q += " AND a.action = ?"
        args.append(action)
    if username:
        q += " AND u.username = ?"
        args.append(username)
    q += " ORDER BY a.log_id DESC LIMIT ?"
    args.append(limit)
    with connect(path) as con:
        return [dict(r) for r in con.execute(q, args)]


def audit_actions(path=None):
    with connect(path) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT action FROM audit_log ORDER BY action")]


def stats(user_id=None, path=None):
    """Counts for the Dashboard (F02). With user_id, only that user's predictions and batches are counted."""
    w, a = ("WHERE user_id = ?", [user_id]) if user_id is not None else ("", [])
    with connect(path) as con:
        by_risk = {r: 0 for r in ("Low", "Medium", "High")}
        for r in con.execute(f"SELECT predicted_risk, COUNT(*) FROM predictions {w} GROUP BY predicted_risk", a):
            by_risk[r[0]] = r[1]
        return {
            "users": con.execute("SELECT COUNT(*) FROM users WHERE is_active = 1").fetchone()[0],
            "predictions": sum(by_risk.values()),
            "high": by_risk["High"],
            "by_risk": by_risk,
            "batches": con.execute(f"SELECT COUNT(*) FROM batch_jobs {w}", a).fetchone()[0],
        }


def daily_counts(user_id=None, days=30, path=None):
    """Predictions per day and risk level for the last `days` days (Dashboard trend chart)."""
    q = """SELECT substr(created_at, 1, 10) AS day, predicted_risk, COUNT(*) AS n FROM predictions
           WHERE created_at >= ?"""
    args = [(datetime.now() - timedelta(days=int(days))).date().isoformat()]
    if user_id is not None:
        q += " AND user_id = ?"
        args.append(user_id)
    q += " GROUP BY day, predicted_risk ORDER BY day"
    with connect(path) as con:
        return [dict(r) for r in con.execute(q, args)]


def early_alerts(user_id=None, limit=10, path=None):
    """Most recent High-risk predictions, highest probability first (Dashboard early-alert list)."""
    q = """SELECT p.prediction_id, p.created_at, s.external_ref, s.source, ROUND(p.prob_high, 3) AS prob_high, u.username
           FROM predictions p JOIN students s ON s.student_id = p.student_id JOIN users u ON u.user_id = p.user_id
           WHERE p.predicted_risk = 'High'"""
    args = []
    if user_id is not None:
        q += " AND p.user_id = ?"
        args.append(user_id)
    q += " ORDER BY p.prediction_id DESC LIMIT ?"
    args.append(limit)
    with connect(path) as con:
        rows = [dict(r) for r in con.execute(q, args)]
    return sorted(rows, key=lambda r: -(r["prob_high"] or 0))


def active_model(path=None):
    with connect(path) as con:
        r = con.execute("SELECT model_id, name, cv_f1_macro, test_f1_macro, trained_at FROM model_versions WHERE is_active = 1").fetchone()
        return dict(r) if r else None


def model_registry(path=None):
    with connect(path) as con:
        return [dict(r) for r in con.execute("SELECT model_id, name, algorithm, ROUND(cv_f1_macro, 4) AS cv_f1_macro, "
                                             "ROUND(test_f1_macro, 4) AS test_f1_macro, is_active, trained_at, file_path "
                                             "FROM model_versions ORDER BY model_id DESC")]
