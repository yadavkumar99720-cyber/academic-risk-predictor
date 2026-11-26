# Deployment: Streamlit Community Cloud + MySQL

The app runs in two ways:

| Where | Database | How it is chosen |
|---|---|---|
| Your laptop | SQLite file `data/academic_risk.db` | default, nothing to set |
| Your laptop or the cloud | MySQL 8 (or TiDB Cloud, which is MySQL-compatible) | set `DATABASE_URL` |

On Streamlit Community Cloud the disk is wiped every time the app restarts, so a SQLite file would lose all users and predictions. That is why the cloud version stores its data in MySQL.

---

## Step 1: Test MySQL on your laptop (about 15 minutes)

1. Install MySQL 8 (MySQL Community Server, or XAMPP, which includes MySQL/MariaDB).
2. Create an empty database:
   ```sql
   CREATE DATABASE academic_risk CHARACTER SET utf8mb4;
   CREATE DATABASE academic_risk_test CHARACTER SET utf8mb4;
   ```
3. Install the driver in your virtual environment: `pip install -r requirements.txt` (this now includes `pymysql` and `cryptography`).
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set:
   ```toml
   ADMIN_PASSWORD = "Choose-A-Strong-Password-1"
   DATABASE_URL = "mysql://root:YOUR_MYSQL_PASSWORD@localhost:3306/academic_risk"
   ```
   If the password contains `@`, `:` or `/`, write it URL-encoded (`@` becomes `%40`).
5. Run the MySQL tests (they empty `academic_risk_test`):
   ```bash
   set TEST_MYSQL_URL=mysql://root:YOUR_MYSQL_PASSWORD@localhost:3306/academic_risk_test      # Windows
   export TEST_MYSQL_URL=mysql://root:YOUR_MYSQL_PASSWORD@localhost:3306/academic_risk_test   # macOS / Linux
   python -m pytest -q
   ```
   All 43 tests should pass. Without `TEST_MYSQL_URL`, 37 pass and 6 are skipped.
6. Optional: copy your existing SQLite data (users, history, audit log) into MySQL:
   ```bash
   python tools/migrate_sqlite_to_mysql.py mysql://root:YOUR_MYSQL_PASSWORD@localhost:3306/academic_risk
   ```
7. `python check_setup.py` should print `OK database MySQL (localhost:3306/academic_risk)`. Then run `streamlit run app.py`, sign in as admin. Under **My Account** the admin sees *Database in use: MySQL (...)*.

## Step 2: Create a free cloud MySQL database (TiDB Cloud Starter)

Aiven no longer offers new free MySQL databases. TiDB Cloud Starter is free within its monthly quota, needs no credit card, and speaks the MySQL protocol.

1. Sign up at https://tidbcloud.com and create a **Starter** cluster (choose a region near India, e.g. Singapore / Mumbai if offered).
2. Open the cluster, click **Connect**, choose **Public** connection, and click **Generate Password**. Note the host, port (4000), user (looks like `xxxxxxxx.root`) and password.
3. In the cluster's SQL editor run `CREATE DATABASE academic_risk;`
4. Your URL is:
   ```
   mysql://xxxxxxxx.root:PASSWORD@gateway01.<region>.prod.aws.tidbcloud.com:4000/academic_risk?ssl=true
   ```
   `ssl=true` is required by TiDB Cloud. The app finds the system certificates itself.
5. Test it from your laptop first: put this URL in `.streamlit/secrets.toml`, run `python check_setup.py`, then `streamlit run app.py`.

Note: TiDB ignores CHECK constraints unless they are switched on, so `test_mysql_role_constraint` can fail on TiDB. The app still checks roles itself. Run the MySQL tests against local MySQL 8.

## Step 3: Put the project on GitHub

1. In the virtual environment where you ran `python run_pipeline.py`, pin the exact versions so the cloud loads the saved models without version errors:
   ```bash
   python tools/make_cloud_requirements.py
   ```
2. Create a new repository on GitHub (it can be private), e.g. `academic-risk-predictor`.
3. From the project folder:
   ```bash
   git init
   git add .
   git commit -m "Academic Risk Predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/academic-risk-predictor.git
   git push -u origin main
   ```
   `.gitignore` keeps out `.venv`, the local SQLite file and `.streamlit/secrets.toml` (your passwords). Check that `artifacts/models/best_model.joblib` and `data/` were pushed: the app needs them.

## Step 4: Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, then **Yup, I have an app**.
3. Repository: `<your-username>/academic-risk-predictor`, branch `main`, main file `app.py`.
4. App URL: choose something like `academic-risk-bharath` → `https://academic-risk-bharath.streamlit.app`.
5. Click **Advanced settings**:
   - Python version: the same as on your laptop (3.11 or 3.12).
   - Secrets: paste
     ```toml
     ADMIN_PASSWORD = "Choose-A-Strong-Password-1"
     DATABASE_URL = "mysql://xxxxxxxx.root:PASSWORD@gateway01.<region>.prod.aws.tidbcloud.com:4000/academic_risk?ssl=true"
     ```
6. Click **Deploy**. The first build takes a few minutes (SHAP and scikit-learn are large).
7. Sign in as `admin` with your `ADMIN_PASSWORD`. Create one teacher and one counsellor, make one prediction, then restart the app from the Cloud menu (⋮ → Reboot) and check that the user and the prediction are still in **Prediction History**. That proves the data lives in MySQL.

## Evidence for the review

- Live URL of the app.
- Screenshot of **My Account** showing *Database in use: MySQL (...)*.
- Screenshot of the TiDB Cloud SQL editor running `SELECT COUNT(*) FROM predictions;`.
- Output of `python -m pytest -q` with `TEST_MYSQL_URL` set (43 passed).

## Troubleshooting

| Message | Fix |
|---|---|
| `PyMySQL package is needed` | `pip install pymysql cryptography` (already in requirements.txt) |
| `Access denied for user` | Wrong user or password in `DATABASE_URL`; URL-encode special characters |
| `Unknown database 'academic_risk'` | Run `CREATE DATABASE academic_risk;` first |
| `Connections using insecure transport are prohibited` | Add `?ssl=true` to the URL |
| scikit-learn / joblib version error on the cloud | Run `python tools/make_cloud_requirements.py`, commit, push |
| App asleep | Free apps sleep when unused; open the URL and click to wake it before the review |
