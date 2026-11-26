-- MySQL 8 version of the 8-table schema (created automatically by src/database.py when DATABASE_URL is set).
CREATE DATABASE IF NOT EXISTS academic_risk CHARACTER SET utf8mb4;
USE academic_risk;

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
