# Explainable and Fairness-Aware Prediction of Student Academic Risk Using Machine Learning

Capstone Project, Phase-1. G BharathKumar (PES1PG25CA065). Guide: Shambavi BN, Assistant Professor.

The project predicts whether a student is at **Low, Medium or High** academic risk. It explains every prediction with SHAP and checks fairness across gender, socio-economic status and age group. Staff use it through a Streamlit web app with login, roles and a SQLite database locally or a MySQL database in the cloud. **Live deployment:** Streamlit Community Cloud with MySQL; see [DEPLOYMENT.md](DEPLOYMENT.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
python run_pipeline.py            # about 4 minutes: clean, train, explain, fairness, database
streamlit run app.py
```

Open http://localhost:8501 and sign in as **admin / Admin@123**. Change this password under *My Account*, then create teacher and counsellor accounts under *User Management*.

Use Python 3.10 to 3.12. If loading the bundled models gives a scikit-learn version error, run `python run_pipeline.py` again to retrain them.

To run the tests: `python -m pytest -q` (43 tests: 37 pass and 6 MySQL tests are skipped unless `TEST_MYSQL_URL` is set; see DEPLOYMENT.md).

## Dataset and risk label

`data/raw/Academic_Risk_Prediction_DW.csv` has 11,523 records combined from five public datasets. Each one measures different things:

| Source | Records | What it measures |
|---|---|---|
| StudentFactors | 6,607 | attendance, study hours, motivation, internet, tutoring, parental involvement, income, previous scores, sleep |
| StudentPerf | 2,392 | absences, weekly study time, parental education and support, tutoring |
| StudentGrades | 1,044 | period-1 and period-2 grades, past failures, parents' education |
| PerfScores | 1,000 | lunch type, test preparation |
| xAPI | 480 | LMS activity (hands raised, resources, announcements, discussion), absence days, stage, topic |

**Risk label.** Each source gives a grade band (A to D) computed from its own final result. The label maps these bands to risk:

- **D → High**: a failing outcome in that source's system, such as G3 below 10/20, GPA below 2.0 or a score below 60.
- **C → Medium**: passing but borderline.
- **A/B → Low**: good standing.

**Leakage control.** The grade band is computed from the final-outcome columns (G3, GPA, exam_score, avg_score and the subject scores), so these columns are never given to the model. The model only sees information available before the final result.

**Placeholder handling.** In the warehouse file, fields a source does not measure are filled with constant values. Cleaning resets these 239,460 cells to "not applicable" so the model cannot treat them as real measurements.

## Results (held-out test set, 2,295 students)

| Model | CV macro-F1 (95% CI) | Test macro-F1 | Test accuracy | High-risk recall |
|---|---|---|---|---|
| Majority-class baseline | 0.273 | 0.273 | 0.694 | 0.000 |
| Logistic Regression | 0.840 [0.827, 0.852] | 0.826 | 0.872 | 0.860 |
| **Random Forest (selected)** | **0.843 [0.833, 0.853]** | **0.834** | **0.880** | **0.871** |
| Gradient Boosting | 0.841 [0.828, 0.855] | 0.820 | 0.871 | 0.858 |

The *Model Evaluation* page also shows:

- A per-source check against a majority-class baseline. The model beats it clearly in every source except PerfScores, which has only two weak fields.
- An ablation study. With the record type and demographics alone, macro-F1 is 0.679. With all features it is 0.834, so the model learns more than which source a record came from.
- A calibration check (Brier score and reliability diagram).

## Fairness

A student is **flagged** when the model predicts High risk. Fairness is measured on this flag, using High-risk recall (equal opportunity) and selection rate. Gaps come with 95% bootstrap confidence intervals. Three mitigation methods are compared with the baseline:

| Mitigation | Stage | Gender: recall gap | Macro-F1 | SES: recall gap | Macro-F1 |
|---|---|---|---|---|---|
| Baseline | n/a | 0.005 | 0.834 | 0.178 | 0.834 |
| Reweighing | pre-processing | 0.061 | 0.833 | 0.371 | 0.826 |
| Exponentiated Gradient (Fairlearn) | in-processing | 0.034 | 0.822 | 0.214 | 0.820 |
| Threshold Optimizer (Fairlearn) | post-processing | 0.027 | 0.821 | 0.026 | 0.800 |

**Gender.** The model is already fair by gender. The baseline recall gap is 0.005, and mitigation brings no benefit.

**Socio-economic status (SES).** There is a real gap: low-SES students who fail are caught less often. The Threshold Optimizer closes the gap to 0.026 and raises High-risk recall to 0.98, but it costs precision and macro-F1. Only 71 High-risk test students have a known SES, so the confidence intervals are wide.

**Explanation consistency.** SHAP feature rankings are almost identical across gender (Spearman ρ = 0.99) and SES (ρ = 0.998). They differ by age group (ρ = 0.77) because younger students come from the LMS source, which measures different behaviours.

## Functionalities (F01-F13)

The app has 13 functionalities. Each one is a screen with exactly this name; the same names and IDs are used in the
report, the presentation, the requirements, the pseudocode and the test cases. After sign-in, pages are listed in the
left sidebar in five groups; Sign out is at the top right.

| ID | Functionality (page) | Menu group | Roles | What it does |
|---|---|---|---|---|
| F01 | Sign In | (login screen) | All | Username and password check (PBKDF2-SHA256), role loaded into the session, failed attempts logged; Sign out button |
| F02 | Dashboard | Overview | All (teacher: own data) | KPI cards, saved predictions by risk, predictions per day, Early-alert list of High-risk students, model in use |
| F03 | Risk Prediction | Prediction | All | Form for one student, risk + confidence, probabilities, SHAP chart, plain-English explanation, suggested interventions, HTML report, saved to history |
| F04 | Batch Prediction | Prediction | All | CSV template, upload, risk + top-3 factors per student, CSV download, save batch to history |
| F05 | Prediction History | Prediction | All (teacher: own only) | Saved predictions, risk filter, stored SHAP factors, CSV export, delete (admin) |
| F06 | Data Management | Data and Models | Admin, Counsellor | Upload, schema validation, cleaning report, cleaned CSV export, save as working dataset (admin) |
| F07 | Exploratory Data Analysis | Data and Models | All | 10 charts plus a correlation heatmap for each record type |
| F08 | Model Evaluation | Data and Models | All | Model comparison with 95% CIs, confusion matrix, per-source check, ablation, calibration, model registry |
| F09 | Explainability | Responsible AI | All | Global SHAP importance, explanation consistency across groups, explanation for any test student |
| F10 | Fairness Audit | Responsible AI | All | Recall gap by gender / income group / age, three bias-mitigation methods compared, ethical-use statement |
| F11 | User Management | Administration | Admin | User list, create user, activate / deactivate |
| F12 | Audit Log | Administration | Admin | All logged actions with action and user filters, CSV export |
| F13 | My Account | Account | All | Profile and change password |

Model training is done offline by the administrator with `python run_pipeline.py`; its results appear in F08 Model
Evaluation, F09 Explainability and F10 Fairness Audit.

## Presentation screenshots (F01-F13)

`tools/make_screenshots.py` starts the app with a separate demo database, performs the action from each slide caption,
saves `screenshots/F01.png` ... `F13.png` and inserts them into the presentation:

```bash
pip install playwright python-pptx
python -m playwright install chromium
python tools/make_screenshots.py --pptx "G_BharathKumar_PES1PG25CA065_AcademicRisk_Presentation3_v5.pptx"
```

## Database (SQLite or MySQL)

By default the app uses the SQLite file `data/academic_risk.db`. Set `DATABASE_URL` (environment variable or `.streamlit/secrets.toml`) to a `mysql://user:password@host:port/database` URL to use MySQL 8 or TiDB Cloud instead; the tables are created automatically. `tools/migrate_sqlite_to_mysql.py` copies existing SQLite data into MySQL. Tables: `users`, `students`, `student_features`, `model_versions`, `batch_jobs`, `predictions`, `prediction_factors`, `audit_log`. The full schema is in `src/database.py` and `docs/schema.sql` (SQLite) and `docs/schema_mysql.sql` (MySQL). Passwords are stored as PBKDF2-SHA256 hashes with a per-user salt.

## Project layout

```
app.py                  Streamlit app
run_pipeline.py         end-to-end training pipeline
src/config.py           paths, feature lists, label mapping, valid ranges
src/data_cleaning.py    validation, deduplication, placeholder masking, range checks, imputation
src/features.py         encoding and readable feature names
src/eda.py              EDA charts
src/train.py            tuning, CV with CIs, per-source check, ablation, calibration
src/explain.py          SHAP global, local, narrative, group consistency
src/fairness.py         group metrics, bootstrap gaps, three mitigation methods, ethics statement
src/predict.py          prediction service for single and batch predictions
src/database.py         SQLite and MySQL schema and data access
src/report.py           downloadable HTML report
src/interventions.py    suggested interventions for each risk factor (F03)
tests/                  pytest suite
docs/                   pseudocode and schema
tools/                  screenshots, SQLite-to-MySQL migration, pinned cloud requirements
.streamlit/             Streamlit settings and a secrets example
DEPLOYMENT.md           MySQL and Streamlit Community Cloud steps
```

## Limitations

- The five sources come from different countries and grading systems. Results need local validation before use at a new institution.
- Region is not in the dataset, so socio-economic status is audited instead.
- Assignment completion is not in the dataset. Test-preparation completion is used as the nearest proxy.
- The prediction is decision support. A High flag should lead to an offer of help, never a penalty.
