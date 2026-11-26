"""Capture the 13 functionality screenshots (F01-F13) from the running app and insert them into the presentation.

Usage (from the project folder, after `python run_pipeline.py` has been run once):

    pip install playwright python-pptx
    python -m playwright install chromium
    python tools/make_screenshots.py --pptx "G_BharathKumar_PES1PG25CA065_AcademicRisk_Presentation3_v5.pptx"

What it does:
  1. Starts the Streamlit app on port 8599 with a separate demo database (your own database is not touched).
  2. Opens Chromium, signs in, performs the action written in each slide caption and saves screenshots/F01.png ... F13.png.
  3. Replaces each "Paste screenshot here" box in the deck with the matching screenshot and saves *_with_screenshots.pptx.

Options: --only-capture (just the PNGs), --only-insert (use PNGs already in screenshots/), --show (watch the browser).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"
PORT = 8599
URL = f"http://localhost:{PORT}"
W, H = 1600, 900  # 16:9, same shape as the slide picture area

PAGES = {"F01": "Sign In", "F02": "Dashboard", "F03": "Risk Prediction", "F04": "Batch Prediction",
         "F05": "Prediction History", "F06": "Data Management", "F07": "Exploratory Data Analysis",
         "F08": "Model Evaluation", "F09": "Explainability", "F10": "Fairness Audit",
         "F11": "User Management", "F12": "Audit Log", "F13": "My Account"}


# ---------------------------------------------------------------- app
def start_app():
    db = Path(tempfile.mkdtemp()) / "screenshots_demo.db"
    env = dict(os.environ, ACADEMIC_RISK_DB=str(db))
    sys.path.insert(0, str(ROOT))
    os.environ["ACADEMIC_RISK_DB"] = str(db)
    import json
    from src import config as C, database
    C.DB_PATH = db
    database.init_db(db)
    database.register_models(json.loads(C.METRICS_PATH.read_text()), path=db)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen(URL, timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    sys.exit("The app did not start. Run `streamlit run app.py` yourself to see the error.")


# ---------------------------------------------------------------- browser helpers
def settle(page, extra=1200):
    page.wait_for_timeout(600)
    try:
        page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=90000)
    except Exception:
        pass
    page.wait_for_timeout(extra)


def shot(page, fid, scroll_to=None):
    if scroll_to:
        try:
            page.get_by_text(scroll_to, exact=False).first.scroll_into_view_if_needed(timeout=10000)
            page.wait_for_timeout(500)
        except Exception:
            pass
    OUT.mkdir(exist_ok=True)
    page.screenshot(path=str(OUT / f"{fid}.png"))
    print(f"  saved {fid}.png")


def go(page, title):
    page.get_by_role("link", name=title, exact=True).first.click()
    settle(page, 2500)


def choose(page, label, option):
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first
    box.click()
    page.get_by_role("option", name=option, exact=True).first.click()
    settle(page)


def number(page, label, value):
    f = page.get_by_label(label, exact=True).first
    f.fill(str(value))
    f.press("Tab")


def text(page, label, value):
    f = page.get_by_label(label, exact=True).first
    f.fill(value)
    f.press("Tab")


def click(page, name):
    page.get_by_role("button", name=name).first.click()
    settle(page, 2000)


# ---------------------------------------------------------------- the 13 captions
def capture(show=False):
    from playwright.sync_api import sync_playwright
    sys.path.insert(0, str(ROOT))
    from src.predict import template_csv
    tpl = Path(tempfile.mkdtemp()) / "batch_template.csv"
    tpl.write_text(template_csv())

    proc = start_app()
    steps = []

    def step(fid, fn):
        steps.append((fid, fn))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not show)
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(URL)
        settle(page, 3000)

        def f01():  # wrong password, then admin / Admin@123
            text(page, "Username", "admin")
            text(page, "Password", "wrong-password")
            click(page, "Sign in")
            page.get_by_text("Invalid username or password.").wait_for(timeout=30000)
            shot(page, "F01")
            text(page, "Password", "Admin@123")
            click(page, "Sign in")
            page.get_by_role("link", name="Dashboard").first.wait_for(timeout=120000)
            settle(page, 2000)

        def f06():  # run schema check and cleaning
            go(page, "Data Management")
            click(page, "Run cleaning and validation")
            shot(page, "F06", "Rows out")

        def f07():
            go(page, "Exploratory Data Analysis")
            settle(page, 3000)
            shot(page, "F07")

        def f03():  # StudentPerf record, 24 absences, 3 study hours
            go(page, "Risk Prediction")
            choose(page, "Record type", "GPA and study-habit records")
            text(page, "Student reference (optional)", "S017")
            number(page, "Absences", 24)
            number(page, "Weekly study time", 3)
            click(page, "Predict risk")
            page.get_by_text("Suggested interventions").first.wait_for(timeout=180000)
            settle(page, 2500)
            shot(page, "F03", "Predicted risk")

        def f04():  # upload the CSV template, save the batch
            go(page, "Batch Prediction")
            page.locator('input[type="file"]').first.set_input_files(str(tpl))
            settle(page, 4000)
            click(page, "Save this batch to prediction history")
            shot(page, "F04", "Students")

        def f05():
            go(page, "Prediction History")
            shot(page, "F05", "View stored explanation")

        def f02():
            go(page, "Dashboard")
            settle(page, 2500)
            shot(page, "F02")

        def f08():
            go(page, "Model Evaluation")
            shot(page, "F08")

        def f09():
            go(page, "Explainability")
            shot(page, "F09", "Explain one student")

        def f10():  # compare by Income group
            go(page, "Fairness Audit")
            try:
                page.get_by_role("button", name="Income group").first.click()
            except Exception:
                page.get_by_text("Income group", exact=True).first.click()
            settle(page, 2500)
            shot(page, "F10")

        def f11():  # user with a 5-character password -> error
            go(page, "User Management")
            text(page, "Username", "teacher1")
            text(page, "Full name", "Demo Teacher")
            text(page, "Temporary password (min 8 characters)", "abc12")
            click(page, "Create user")
            shot(page, "F11", "Create user")

        def f12():  # filter by failed_login
            go(page, "Audit Log")
            try:
                choose(page, "Action", "failed_login")
            except Exception:
                pass
            shot(page, "F12")

        def f13():  # mismatched new passwords
            go(page, "My Account")
            text(page, "Current password", "Admin@123")
            text(page, "New password (min 8 characters)", "NewPass@123")
            text(page, "Repeat new password", "Different@123")
            click(page, "Change password")
            shot(page, "F13")

        for fid, fn in [("F01", f01), ("F06", f06), ("F07", f07), ("F03", f03), ("F04", f04), ("F05", f05), ("F02", f02),
                        ("F08", f08), ("F09", f09), ("F10", f10), ("F11", f11), ("F12", f12), ("F13", f13)]:
            print(f"{fid} {PAGES[fid]}")
            try:
                fn()
            except Exception as e:
                print(f"  could not finish the caption action ({type(e).__name__}); saving the page as it is")
                try:
                    shot(page, fid)
                except Exception:
                    pass
        browser.close()
    proc.terminate()


# ---------------------------------------------------------------- insert into the deck
def insert(pptx_path):
    from pptx import Presentation
    from PIL import Image
    prs = Presentation(pptx_path)
    done = []
    for slide in prs.slides:
        for sh in list(slide.shapes):
            if not sh.name.startswith("Screenshot placeholder "):
                continue
            fid = sh.name.split()[-1]
            img = OUT / f"{fid}.png"
            if not img.exists():
                print(f"  {fid}.png missing - placeholder kept")
                continue
            x, y, w, h = sh.left, sh.top, sh.width, sh.height
            iw, ih = Image.open(img).size
            scale = min(w / iw, h / ih)
            pw, ph = int(iw * scale), int(ih * scale)
            pic = slide.shapes.add_picture(str(img), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)
            pic.name = f"Screenshot {fid}"
            pic.line.color.rgb = __import__("pptx.dml.color", fromlist=["RGBColor"]).RGBColor(0xD9, 0xDE, 0xE5)
            pic.line.width = 12700
            sh._element.getparent().remove(sh._element)
            done.append(fid)
    out = Path(pptx_path).with_name(Path(pptx_path).stem + "_with_screenshots.pptx")
    prs.save(out)
    print(f"Inserted {len(done)} screenshots: {', '.join(done)}\nSaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pptx", help="presentation file with the screenshot placeholders")
    ap.add_argument("--only-capture", action="store_true")
    ap.add_argument("--only-insert", action="store_true")
    ap.add_argument("--show", action="store_true", help="show the browser while capturing")
    a = ap.parse_args()
    if not a.only_insert:
        capture(a.show)
    if not a.only_capture:
        if not a.pptx:
            sys.exit("Give --pptx path to insert the screenshots.")
        insert(a.pptx)
