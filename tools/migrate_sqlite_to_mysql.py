"""Copy every row from the local SQLite database into MySQL (users, predictions, history, audit log ...).

Usage:
    python tools/migrate_sqlite_to_mysql.py mysql://user:password@host:3306/academic_risk [path/to/academic_risk.db]

The MySQL tables are created if needed. Rows keep their ids, so links between tables stay correct.
The MySQL database should be empty; rows whose id already exists there are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src import database as db  # noqa: E402


def main():
    if len(sys.argv) < 2 or not sys.argv[1].startswith("mysql"):
        sys.exit(__doc__)
    url, src = sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else C.DB_PATH
    if not src.exists():
        sys.exit(f"SQLite file not found: {src}")
    db.init_db(url)
    with db.connect(src) as lite, db.connect(url) as my:
        my.execute("SET FOREIGN_KEY_CHECKS = 0")
        for t in db.TABLES:
            rows = [dict(r) for r in lite.execute(f"SELECT * FROM {t}")]
            if not rows:
                print(f"  {t:20s} 0 rows")
                continue
            cols = list(rows[0])
            my.executemany(f"INSERT IGNORE INTO {t} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                           [tuple(r[c] for c in cols) for r in rows])
            print(f"  {t:20s} {len(rows)} rows")
        my.execute("SET FOREIGN_KEY_CHECKS = 1")
    print("Done. Set DATABASE_URL to this MySQL URL and start the app.")


if __name__ == "__main__":
    main()
