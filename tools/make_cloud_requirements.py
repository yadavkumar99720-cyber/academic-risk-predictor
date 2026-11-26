"""Write requirements.txt with the exact package versions installed in this environment.

Run it in the virtual environment where `python run_pipeline.py` trained the models, then commit the file.
Streamlit Community Cloud then installs the same versions, so the saved model files load without
scikit-learn version errors.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "shap", "fairlearn", "streamlit", "plotly", "joblib",
            "pyarrow", "pymysql", "cryptography", "pytest"]

lines = []
for p in PACKAGES:
    try:
        lines.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print(f"not installed, left out: {p}  (pip install {p})")
out = Path(__file__).resolve().parent.parent / "requirements.txt"
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}:\n" + "\n".join(lines))
