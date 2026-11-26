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
