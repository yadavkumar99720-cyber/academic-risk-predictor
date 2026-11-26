import html
from datetime import datetime

from . import config as C


def single_report(record, result, user, suggestions=None):
    p = result["proba"]
    rows_up = "".join(f"<tr><td>{html.escape(r.label)}</td><td>{html.escape(str(r.value))}</td><td>+{r.shap_high:.3f}</td></tr>" for r in result["risk_increasing"].itertuples())
    rows_dn = "".join(f"<tr><td>{html.escape(r.label)}</td><td>{html.escape(str(r.value))}</td><td>{r.shap_high:.3f}</td></tr>" for r in result["risk_reducing"].itertuples())
    inputs = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in record.items() if v is not None and str(v) != "nan")
    sug = ""
    if suggestions is not None and len(suggestions):
        sug = "<h3>Suggested interventions</h3><table><tr><th>Factor</th><th>Support area</th><th>Suggested action</th></tr>" + "".join(
            f"<tr><td>{html.escape(str(r['Factor']))}</td><td>{html.escape(str(r['Support area']))}</td><td>{html.escape(str(r['Suggested action']))}</td></tr>"
            for _, r in suggestions.iterrows()) + "</table>"
    color = {"High": "#c0392b", "Medium": "#b7791f", "Low": "#2e7d5b"}[result["pred_class"]]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Academic risk report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:820px;margin:32px auto;color:#1f2933;padding:0 16px}}
h1{{font-size:22px}}table{{border-collapse:collapse;width:100%;margin:8px 0 20px}}td,th{{border:1px solid #d9dee5;padding:6px 10px;text-align:left;font-size:14px}}
th{{background:#f1f4f8}}.badge{{display:inline-block;padding:6px 14px;border-radius:6px;color:#fff;background:{color};font-weight:600}}
.note{{font-size:12px;color:#5b6675;border-top:1px solid #d9dee5;padding-top:10px}}</style></head><body>
<h1>Student academic-risk report</h1>
<p>Generated {datetime.now():%d %b %Y, %H:%M} by {html.escape(user)} &middot; Record type: {html.escape(C.SOURCE_LABELS.get(record.get('source'), str(record.get('source'))))}</p>
<p><span class="badge">{result['pred_class']} risk</span> &nbsp; Confidence {p.max()*100:.1f}%</p>
<table><tr><th>Low</th><th>Medium</th><th>High</th></tr><tr><td>{p[0]*100:.1f}%</td><td>{p[1]*100:.1f}%</td><td>{p[2]*100:.1f}%</td></tr></table>
<h2>Explanation</h2><p>{html.escape(result['text'])}</p>
<h3>Factors increasing High-risk probability</h3><table><tr><th>Factor</th><th>Value</th><th>SHAP</th></tr>{rows_up or '<tr><td colspan=3>None</td></tr>'}</table>
<h3>Factors reducing High-risk probability</h3><table><tr><th>Factor</th><th>Value</th><th>SHAP</th></tr>{rows_dn or '<tr><td colspan=3>None</td></tr>'}</table>
{sug}<h3>Inputs</h3><table><tr><th>Field</th><th>Value</th></tr>{inputs}</table>
<p class="note">This prediction is decision support only. It should be reviewed by a teacher or counsellor and used to offer help, never to penalise a student.</p>
</body></html>"""
