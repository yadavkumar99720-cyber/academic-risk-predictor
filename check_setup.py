import importlib
import platform
import sys
import time
from pathlib import Path

print("Python", sys.version.split()[0], "on", platform.system())
ok = True
for name in ["pandas", "numpy", "scipy", "sklearn", "plotly", "streamlit", "joblib", "fairlearn", "shap"]:
    t = time.time()
    try:
        m = importlib.import_module(name)
        print(f"  OK   {name:10s} {getattr(m, '__version__', '?'):10s} ({time.time() - t:.1f}s)")
    except Exception as e:
        ok = False
        print(f"  FAIL {name:10s} {e}")
root = Path(__file__).resolve().parent
for f in ["data/raw/Academic_Risk_Prediction_DW.csv", "data/processed/cleaned_dataset.csv", "artifacts/models/best_model.joblib", "artifacts/metrics.json"]:
    exists = (root / f).exists()
    ok &= exists
    print(("  OK   " if exists else "  MISSING ") + f)
if (root / "artifacts/models/best_model.joblib").exists():
    try:
        import joblib
        joblib.load(root / "artifacts/models/best_model.joblib")
        print("  OK   model loads")
    except Exception as e:
        ok = False
        print("  FAIL model load:", e, "-> run: python run_pipeline.py")
try:
    from src import database
    database.init_db()
    with database.connect() as con:
        n = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"  OK   database {database.describe()} ({n} users)")
except Exception as e:
    ok = False
    print("  FAIL database:", e)
print("\nAll good. Start with: streamlit run app.py" if ok else "\nFix the items marked FAIL / MISSING above.")
