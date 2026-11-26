import os
import sys
import tempfile
from pathlib import Path

os.environ["ACADEMIC_RISK_DB"] = str(Path(tempfile.mkdtemp()) / "test_app.db")
os.environ["DATABASE_URL"] = ""  # tests use temporary SQLite files; MySQL tests use TEST_MYSQL_URL
os.environ["ADMIN_PASSWORD"] = "Admin@123"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
