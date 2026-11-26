import streamlit as st

st.set_page_config(page_title="Academic Risk Predictor", page_icon="🎓", layout="wide")
_loading = st.empty()
_loading.info("Loading the Academic Risk Predictor... the first start can take up to a minute while libraries load.")

import json
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import config as C
from src import data_cleaning, database, eda, interventions, report
from src.features import encode

RISK_COLORS = eda.RISK_COLORS


def template_csv():
    from src.predict import template_csv as t
    return t()

# The 13 functionalities of the system. Each page title below is the official functionality name used in the
# report, the presentation, the requirements (F01-F13), the pseudocode and the test cases. F01 Sign In is the
# login screen shown before these pages.
ALL_ROLES = ["admin", "counsellor", "teacher"]
PAGES = [
    # (functionality id, page title, menu group, icon, roles allowed)
    ("F02", "Dashboard", "Overview", ":material/dashboard:", ALL_ROLES),
    ("F03", "Risk Prediction", "Prediction", ":material/person_search:", ALL_ROLES),
    ("F04", "Batch Prediction", "Prediction", ":material/group:", ALL_ROLES),
    ("F05", "Prediction History", "Prediction", ":material/history:", ALL_ROLES),
    ("F06", "Data Management", "Data and Models", ":material/database:", ["admin", "counsellor"]),
    ("F07", "Exploratory Data Analysis", "Data and Models", ":material/bar_chart:", ALL_ROLES),
    ("F08", "Model Evaluation", "Data and Models", ":material/model_training:", ALL_ROLES),
    ("F09", "Explainability", "Responsible AI", ":material/lightbulb:", ALL_ROLES),
    ("F10", "Fairness Audit", "Responsible AI", ":material/balance:", ALL_ROLES),
    ("F11", "User Management", "Administration", ":material/manage_accounts:", ["admin"]),
    ("F12", "Audit Log", "Administration", ":material/receipt_long:", ["admin"]),
    ("F13", "My Account", "Account", ":material/account_circle:", ALL_ROLES),
]


@st.cache_resource
def boot():
    database.init_db()
    # A fresh database has no model registered until run_pipeline.py is run; register the saved models so the
    # Dashboard and Model Evaluation show the model in use.
    if database.active_model() is None and C.METRICS_PATH.exists():
        database.register_models(json.loads(C.METRICS_PATH.read_text()))
    return True


@st.cache_data
def load_clean():
    if C.CLEAN_PATH.exists():
        return pd.read_csv(C.CLEAN_PATH)
    df, _ = data_cleaning.run()
    return df


@st.cache_data
def load_json(path):
    p = C.ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def service():
    df = load_clean()
    from src.predict import RiskService
    bg = encode(df.sample(min(500, len(df)), random_state=C.RANDOM_STATE))
    with st.spinner("Loading the model and SHAP explainer (first time only)..."):
        return RiskService(background=bg)


def require_artifacts():
    if not (C.MODEL_DIR / "best_model.joblib").exists() or not C.METRICS_PATH.exists():
        st.error("Trained models were not found. Run `python run_pipeline.py` once, then reload this page.")
        st.stop()


def login_view():
    c1, c2, c3 = st.columns([1, 1.3, 1])
    with c2:
        st.markdown("## 🎓 Academic Risk Predictor")
        st.caption("Explainable and Fairness-Aware Prediction of Student Academic Risk Using Machine Learning")
        st.subheader("Sign In")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", width="stretch")
        if ok:
            user = database.authenticate(u, p)
            if user:
                st.session_state.user = {k: user[k] for k in ("user_id", "username", "full_name", "role")}
                database.log(user["user_id"], "login")
                st.rerun()
            else:
                database.log(None, "failed_login", {"username": u})
                st.error("Invalid username or password.")
        st.info("First run: sign in as **admin** / **Admin@123**, then change the password under My Account and create staff users under User Management.")


def risk_badge(cls, conf):
    col = RISK_COLORS[cls]
    st.markdown(f"<div style='padding:14px 18px;border-radius:10px;background:{col}1a;border-left:6px solid {col}'>"
                f"<div style='font-size:13px;color:#555'>Predicted risk</div>"
                f"<div style='font-size:30px;font-weight:700;color:{col}'>{cls}</div>"
                f"<div style='font-size:14px'>Confidence {conf*100:.1f}%</div></div>", unsafe_allow_html=True)


def proba_chart(p):
    d = pd.DataFrame({"Risk": C.RISK_CLASSES, "Probability": p})
    f = px.bar(d, x="Probability", y="Risk", orientation="h", color="Risk", color_discrete_map=RISK_COLORS, range_x=[0, 1], text=d["Probability"].map(lambda v: f"{v:.1%}"))
    f.update_layout(showlegend=False, height=220, margin=dict(l=10, r=10, t=10, b=10))
    return f


def shap_chart(table, k=10):
    t = table.head(k).iloc[::-1]
    f = go.Figure(go.Bar(x=t["shap_high"], y=[f"{a} = {b}" for a, b in zip(t["label"], t["value"].map(fmt))], orientation="h",
                         marker_color=["#c0392b" if v > 0 else "#2e7d5b" for v in t["shap_high"]]))
    f.update_layout(title="SHAP contribution to High-risk probability (red raises risk, green lowers it)", height=420,
                    margin=dict(l=10, r=10, t=50, b=10), xaxis_title="SHAP value")
    return f


def fmt(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:.2f}"
    except Exception:
        return str(v)


def page_data():
    st.title("Data Management")
    st.write("Import a raw file, validate it, clean it and export the cleaned dataset.")
    up = st.file_uploader("Upload a raw CSV in the same format as the Kaggle file (leave empty to use the bundled dataset)", type="csv")
    raw = pd.read_csv(up) if up else data_cleaning.load_raw()
    st.write(f"**Raw data:** {raw.shape[0]:,} rows × {raw.shape[1]} columns")
    st.dataframe(raw.head(20), width="stretch")
    v = data_cleaning.validate_schema(raw)
    if not v["ok"]:
        st.error(f"Schema problems: missing columns {v['missing_columns']}, unknown sources {v['unknown_sources']}")
        return
    st.success("Schema check passed.")
    if st.button("Run cleaning and validation", type="primary"):
        df, rep = data_cleaning.clean(raw)
        st.session_state.cleaned = (df, rep)
        database.log(st.session_state.user["user_id"], "clean_dataset", {"rows_in": rep["rows_in"], "rows_out": rep["rows_out"]})
    if "cleaned" in st.session_state:
        df, rep = st.session_state.cleaned
        c = st.columns(5)
        c[0].metric("Rows in", f"{rep['rows_in']:,}")
        c[1].metric("Duplicates removed", rep["duplicates_removed"])
        c[2].metric("Missing values imputed", rep["missing_values_imputed"])
        c[3].metric("Out-of-range values clipped", sum(rep["out_of_range_values_clipped"].values()))
        c[4].metric("Rows out", f"{rep['rows_out']:,}")
        st.write("**Not-applicable placeholder cells set to empty:**", f"{rep['placeholder_cells_marked_not_applicable']:,}",
                 "— each source only measures some fields; the warehouse had filled the rest with constant placeholder values, which would mislead a model.")
        st.write("**Category values standardised:**", rep["category_values_standardized"])
        st.write("**Range check (clipped):**", rep["out_of_range_values_clipped"] or "none")
        st.write("**Extreme outliers flagged (3×IQR, kept):**", rep["extreme_outliers_flagged_iqr3"] or "none")
        st.dataframe(df.head(50), width="stretch")
        st.download_button("Download cleaned dataset (CSV)", df.to_csv(index=False).encode(), "cleaned_dataset.csv", "text/csv")
        if st.session_state.user["role"] == "admin" and st.button("Save as the working dataset"):
            df.to_csv(C.CLEAN_PATH, index=False)
            load_clean.clear()
            st.success("Saved. Re-run `python run_pipeline.py` to retrain models on it.")


def page_eda():
    st.title("Exploratory Data Analysis")
    df = load_clean()
    names = [n for n, _ in eda.ALL]
    choice = st.multiselect("Charts", names, default=names)
    cols = st.columns(2)
    for i, (n, fn) in enumerate([x for x in eda.ALL if x[0] in choice]):
        with cols[i % 2]:
            st.plotly_chart(fn(df), width="stretch")
    st.subheader("Correlation heatmap")
    src = st.selectbox("Record type", list(C.SOURCE_FEATURES), index=4, format_func=lambda s: C.SOURCE_LABELS[s])
    st.plotly_chart(eda.correlation_heatmap(df, src), width="stretch")
    st.caption("Assignment-completion data is not in the dataset; test-preparation completion is shown as the closest available proxy.")


def page_models():
    st.title("Model Evaluation")
    m = load_json("artifacts/metrics.json")
    st.write(f"Training rows: {m['n_train']:,} · Test rows: {m['n_test']:,} · 5-fold stratified cross-validation with 95% confidence intervals.")
    rows = []
    for n, r in m["models"].items():
        cv = r["cv"]
        rows.append({"Model": n, "CV macro-F1": f"{cv['f1_macro']['mean']:.3f} [{cv['f1_macro']['low']:.3f}, {cv['f1_macro']['high']:.3f}]",
                     "CV balanced acc.": f"{cv['bal_acc']['mean']:.3f} [{cv['bal_acc']['low']:.3f}, {cv['bal_acc']['high']:.3f}]",
                     "CV High recall": f"{cv['high_recall']['mean']:.3f} [{cv['high_recall']['low']:.3f}, {cv['high_recall']['high']:.3f}]",
                     "Test macro-F1": r["test"]["f1_macro"], "Test accuracy": r["test"]["accuracy"], "Test High recall": r["test"]["high_recall"],
                     "Brier": r["test"]["brier"], "Best hyperparameters": json.dumps(r.get("best_params", {}))})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.success(f"Selected model: **{m['best_model']}** (highest mean CV macro-F1). The three tuned models overlap within their confidence intervals.")
    sel = st.selectbox("Confusion matrix for", [k for k in m["models"] if not k.startswith("Majority")], index=1)
    cm = np.array(m["models"][sel]["test"]["confusion_matrix"])
    c1, c2 = st.columns(2)
    with c1:
        f = px.imshow(cm, x=[f"Pred {l}" for l in C.RISK_CLASSES], y=[f"True {l}" for l in C.RISK_CLASSES], text_auto=True, color_continuous_scale="Blues")
        f.update_layout(title=f"Confusion matrix - {sel}", height=380)
        st.plotly_chart(f, width="stretch")
    with c2:
        rep = pd.DataFrame(m["models"][sel]["test"]["report"]).T.loc[C.RISK_CLASSES, ["precision", "recall", "f1-score", "support"]].round(3)
        st.dataframe(rep, width="stretch")
    st.subheader("Per-source validation")
    ps = pd.DataFrame(m["per_source"])
    ps["source"] = ps["source"].map(C.SOURCE_LABELS)
    st.dataframe(ps, width="stretch", hide_index=True)
    st.caption("Balanced accuracy is compared with a 'guess the most common class in this source' baseline (0.333). The model clearly beats it everywhere except the standardized-test records, which carry only two weak background fields.")
    st.subheader("Ablation study")
    st.dataframe(pd.DataFrame(m["ablation"]), width="stretch", hide_index=True)
    st.subheader("Calibration of the High-risk probability")
    cal = m["calibration"]
    c = st.columns(3)
    c[0].metric("Brier (High) uncalibrated", cal["brier_high_uncalibrated"])
    c[1].metric("Brier (High) isotonic", cal["brier_high_isotonic"])
    c[2].metric("Macro-F1 after isotonic", cal["f1_macro_isotonic"])
    pts = pd.concat([pd.DataFrame(c_["points"]).assign(model=c_["model"]) for c_ in cal["reliability"]])
    f = px.line(pts, x="pred", y="obs", color="model", markers=True, title="Reliability diagram (High risk)")
    f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    st.plotly_chart(f, width="stretch")
    st.subheader("Model registry")
    st.caption("Every model trained by the pipeline (run_pipeline.py) is registered here; the active model is used for all predictions.")
    reg = pd.DataFrame(database.model_registry())
    if reg.empty:
        st.info("No models registered yet. Run python run_pipeline.py.")
    else:
        reg["is_active"] = reg["is_active"].map({1: "Active", 0: ""})
        st.dataframe(reg, width="stretch", hide_index=True)


def page_explain():
    st.title("Explainability")
    sh = load_json("artifacts/shap_global.json")
    imp = pd.DataFrame(sh["importance"]).head(15).iloc[::-1]
    f = px.bar(imp, x="mean_abs_shap", y="label", orientation="h", title=f"Global feature importance (mean |SHAP|, {sh['n_explained']} test students)")
    f.update_layout(height=520)
    st.plotly_chart(f, width="stretch")
    st.subheader("Are explanations consistent across groups?")
    rows = [{"Attribute": a, "Groups compared": " vs ".join(v["groups"]), "Spearman ρ of feature importance": v["spearman_rho"],
             "Top-5 overlap": f"{v['top5_overlap']}/5"} for a, v in sh["group_consistency"].items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("ρ close to 1 means the model relies on the same factors for both groups. Gender and socio-economic groups get near-identical explanations; age groups differ because younger students come from the LMS records, which measure different behaviours.")
    st.subheader("Explain one student from the test set")
    df = load_clean()
    split = __import__("joblib").load(C.MODEL_DIR / "split.joblib")
    te = df.loc[split["idx_test"]]
    src = st.selectbox("Record type", list(C.SOURCE_FEATURES), format_func=lambda s: C.SOURCE_LABELS[s])
    pool = te[te["source"] == src]
    i = st.selectbox("Student (row id · true risk)", pool.index[:300], format_func=lambda k: f"{k} · true {pool.loc[k, C.TARGET]}")
    rec = {k: pool.loc[i, k] for k in C.FEATURES}
    res, _ = service().explain_record(rec)
    c1, c2 = st.columns([1, 2])
    with c1:
        risk_badge(res["pred_class"], res["proba"].max())
        st.plotly_chart(proba_chart(res["proba"]), width="stretch")
        st.write(f"True risk: **{pool.loc[i, C.TARGET]}**")
    with c2:
        st.plotly_chart(shap_chart(res["table"]), width="stretch")
    st.info(res["text"])
    c1, c2 = st.columns(2)
    c1.write("**Risk-increasing factors**")
    c1.dataframe(res["risk_increasing"][["label", "value", "shap_high"]], hide_index=True, width="stretch")
    c2.write("**Risk-reducing factors**")
    c2.dataframe(res["risk_reducing"][["label", "value", "shap_high"]], hide_index=True, width="stretch")


ATTR_LABELS = {"gender": "Gender", "ses_group": "Income group", "age_group": "Age group"}


GROUP_LABELS = {"female": "Female", "male": "Male", "Low SES": "Low income", "Not low SES": "Other income",
                "Unknown": "Not recorded", "Under 15": "Under 15", "15 and above": "15 and above"}


def glabel(g):
    return GROUP_LABELS.get(str(g), str(g))


def pct(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def page_fairness():
    st.title("Fairness Audit")
    st.write("A student is flagged when the model predicts **High risk**. We check whether High-risk students "
             "from every group are found equally well. The **recall gap** is the difference between the best "
             "and worst group; a smaller gap is fairer.")
    fa = load_json("artifacts/fairness.json")
    attr = st.segmented_control("Compare students by", C.SENSITIVE_ATTRIBUTES, default="gender",
                                format_func=lambda a: ATTR_LABELS[a]) or "gender"
    base = fa["baseline"][attr]
    groups = pd.DataFrame(base["groups"])
    known = groups[groups["group"] != "Unknown"].copy()
    known["group"] = known["group"].map(glabel)
    mit = fa["mitigation"].get(attr)

    c = st.columns([1, 1, 1.5, 1])
    c[0].metric("Recall gap (current model)", f"{base['gaps']['recall_gap']:.3f}")
    worst = known.loc[known["high_recall"].idxmin()] if known["high_recall"].notna().any() else None
    if worst is None:
        c[1].metric("Lowest recall group", "-")
    else:
        c[1].metric(f"Lowest recall: {glabel(worst['group'])}", pct(worst["high_recall"]))
    if mit:
        ok = [r for r in mit if "error" not in r and r.get("recall_gap") is not None]
        best = min(ok, key=lambda r: r["recall_gap"])
        if best["method"].startswith("Baseline"):
            c[2].metric("Best mitigation method", "None needed")
            c[3].metric("Recall gap after mitigation", f"{best['recall_gap']:.3f}")
        else:
            c[2].metric("Best mitigation method", best["method"].split(" (")[0])
            c[3].metric("Recall gap after mitigation", f"{best['recall_gap']:.3f}",
                        f"{best['recall_gap'] - base['gaps']['recall_gap']:+.3f}", delta_color="inverse")
    else:
        c[2].metric("Best mitigation method", "Not run")
        c[3].metric("Selection-rate gap", f"{base['gaps']['selection_rate_gap']:.3f}")

    t1, t2, t3 = st.tabs(["Results by group", "Bias mitigation", "Ethical use"])
    with t1:
        left, right = st.columns(2)
        with left:
            f = px.bar(known, x="group", y="high_recall", text=known["high_recall"].map(pct), color_discrete_sequence=["#1f4e8c"],
                       title="High-risk students correctly found, by group")
            f.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05], xaxis_title="", yaxis_title="Recall", height=360)
            st.plotly_chart(f, width="stretch")
        with right:
            d = known.melt(id_vars="group", value_vars=["pred_low", "pred_medium", "pred_high"], var_name="risk", value_name="share")
            d["risk"] = d["risk"].str.replace("pred_", "").str.title()
            f = px.bar(d, y="group", x="share", color="risk", orientation="h", color_discrete_map=RISK_COLORS,
                       category_orders={"risk": C.RISK_CLASSES}, title="Predicted risk levels in each group")
            f.update_layout(xaxis_tickformat=".0%", yaxis_title="", xaxis_title="Share of students", height=360, legend_title="")
            st.plotly_chart(f, width="stretch")
        table = pd.DataFrame({
            "Group": groups["group"].map(glabel),
            "Students": groups["n"],
            "Actually High risk": groups["actual_high"],
            "High risk found (recall)": groups["high_recall"].map(pct),
            "High risk missed": groups["fnr"].map(pct),
            "Flagged as High": groups["selection_rate"].map(pct),
            "Macro-F1": groups["f1_macro"].map(lambda v: f"{v:.3f}"),
        })
        st.dataframe(table, width="stretch", hide_index=True)
        ci = base["gaps"]["recall_gap_ci"]
        st.caption(f"Recall gap {base['gaps']['recall_gap']:.3f} (95% confidence interval {ci[0]:.3f} to {ci[1]:.3f}). "
                   "Students whose group is not recorded are shown as Not recorded and left out of the gap.")
    with t2:
        if not mit:
            st.info(f"Bias mitigation was run for gender and income group. For {ATTR_LABELS[attr].lower()} only the audit is shown.")
        else:
            rows = [r for r in mit if "error" not in r]
            mt = pd.DataFrame({
                "Method": [r["method"].split(" (")[0] for r in rows],
                "Stage": [r["stage"] if r["stage"] != "-" else "None" for r in rows],
                "Macro-F1": [f"{r['f1_macro']:.3f}" for r in rows],
                "High-risk recall": [pct(r["high_recall"]) for r in rows],
                "Recall gap": [round(r["recall_gap"], 3) for r in rows],
                "95% CI of gap": [f"{r['recall_gap_ci'][0]:.3f} to {r['recall_gap_ci'][1]:.3f}" for r in rows],
            })
            left, right = st.columns([1.1, 1])
            with left:
                f = px.bar(mt, x="Method", y="Recall gap", text=mt["Recall gap"].map(lambda v: f"{v:.3f}"),
                           color=mt["Method"].eq(mt.loc[mt["Recall gap"].idxmin(), "Method"]).map({True: "Lowest gap", False: "Other"}),
                           color_discrete_map={"Lowest gap": "#2e7d5b", "Other": "#9aa5b1"}, title="Recall gap by method (lower is fairer)")
                f.update_layout(showlegend=False, xaxis_title="", height=360)
                st.plotly_chart(f, width="stretch")
            with right:
                f = px.bar(mt.assign(F1=mt["Macro-F1"].astype(float)), x="Method", y="F1", text="Macro-F1", color_discrete_sequence=["#1f4e8c"],
                           title="Overall accuracy by method (higher is better)")
                f.update_layout(xaxis_title="", yaxis_title="Macro-F1", yaxis_range=[0.7, 0.86], height=360)
                st.plotly_chart(f, width="stretch")
            st.dataframe(mt, width="stretch", hide_index=True)
            st.caption("Reweighing changes training weights, Exponentiated Gradient adds a fairness rule during training, "
                       "and Threshold Optimizer sets a separate decision threshold for each group after training.")
            with st.expander("Results by group for one method"):
                pick = st.selectbox("Method", [r["method"] for r in rows], format_func=lambda m: m.split(" (")[0])
                g2 = pd.DataFrame(next(r for r in rows if r["method"] == pick)["groups"])
                st.dataframe(pd.DataFrame({"Group": g2["group"].map(glabel), "Students": g2["n"], "High risk found (recall)": g2["high_recall"].map(pct),
                                           "Flagged as High": g2["selection_rate"].map(pct), "Precision (High)": g2["precision_high"].map(pct)}),
                             width="stretch", hide_index=True)
    with t3:
        for line in fa["ethics_statement"]:
            st.markdown(f"- {line}")


FIELD_SPECS = {
    "raisedhands": ("number", 0, 100, 30), "visited_resources": ("number", 0, 100, 40), "announcements_view": ("number", 0, 100, 20),
    "discussion": ("number", 0, 100, 30), "absence_days": ("select", ["under-7", "above-7"]), "stage": ("select", ["lowerlevel", "middleschool", "highschool"]),
    "topic": ("select", ["arabic", "biology", "chemistry", "english", "french", "geology", "history", "it", "math", "quran", "science", "spanish"]),
    "absences": ("number", 0, 30, 10), "study_time": ("float", 0.0, 20.0, 8.0), "parental_education": ("number", 0, 4, 2),
    "parental_support": ("number", 0, 4, 2), "tutoring": ("select", [0, 1]), "G1": ("number", 0, 20, 11), "G2": ("number", 0, 20, 11),
    "failures": ("number", 0, 4, 0), "Medu": ("number", 0, 4, 2), "Fedu": ("number", 0, 4, 2), "lunch": ("select", ["standard", "free/reduced"]),
    "test_prep": ("select", ["none", "completed"]), "attendance": ("number", 60, 100, 80), "hours_studied": ("number", 1, 44, 20),
    "motivation_level": ("select", ["low", "medium", "high"]), "internet_access": ("select", ["yes", "no"]), "tutoring_sessions": ("number", 0, 8, 1),
    "parental_involvement": ("select", ["low", "medium", "high"]), "family_income": ("select", ["low", "medium", "high"]),
    "previous_scores": ("number", 50, 100, 70), "sleep_hours": ("number", 4, 10, 7),
}


def page_predict():
    st.title("Risk Prediction")
    from src.features import PRETTY
    src = st.selectbox("Record type", list(C.SOURCE_FEATURES), format_func=lambda s: C.SOURCE_LABELS[s])
    with st.form("predict"):
        c = st.columns(3)
        ref = c[0].text_input("Student reference (optional)")
        gender = c[1].selectbox("Gender", ["female", "male"])
        age = c[2].number_input("Age", 8, 30, 13 if src == "xAPI" else 16)
        rec = {"source": src, "gender": gender, "age": age}
        cols = st.columns(3)
        for i, f in enumerate(C.SOURCE_FEATURES[src]):
            spec = FIELD_SPECS[f]
            with cols[i % 3]:
                lab = PRETTY.get(f, f.replace("_", " ").capitalize())
                if spec[0] == "select":
                    rec[f] = st.selectbox(lab, spec[1])
                elif spec[0] == "float":
                    rec[f] = st.number_input(lab, spec[1], spec[2], spec[3], step=0.5)
                else:
                    rec[f] = st.number_input(lab, spec[1], spec[2], spec[3], step=1)
        go_ = st.form_submit_button("Predict risk", type="primary")
    if go_:
        res, _ = service().explain_record(rec)
        pid = database.save_prediction(st.session_state.user["user_id"], rec, res, res["table"].head(10).to_dict("records"), external_ref=ref or None)
        database.log(st.session_state.user["user_id"], "predict_single", {"prediction_id": pid, "risk": res["pred_class"]})
        st.session_state.last = (rec, res, pid)
    if "last" in st.session_state and st.session_state.last[0]["source"] == src:
        rec, res, pid = st.session_state.last
        c1, c2 = st.columns([1, 2])
        with c1:
            risk_badge(res["pred_class"], res["proba"].max())
            st.plotly_chart(proba_chart(res["proba"]), width="stretch")
            st.caption(f"Saved as prediction #{pid}")
        with c2:
            st.plotly_chart(shap_chart(res["table"]), width="stretch")
        st.info(res["text"])
        a, b = st.columns(2)
        a.write("**Risk-increasing factors**")
        a.dataframe(res["risk_increasing"][["label", "value", "shap_high"]], hide_index=True, width="stretch")
        b.write("**Risk-reducing factors**")
        b.dataframe(res["risk_reducing"][["label", "value", "shap_high"]], hide_index=True, width="stretch")
        st.subheader("Suggested interventions")
        sug = interventions.suggest(res["risk_increasing"], res["pred_class"])
        st.dataframe(sug, width="stretch", hide_index=True)
        st.caption("Suggestions come from the student's top risk-increasing factors. They support the teacher's decision; they do not replace it.")
        st.download_button("Download prediction report (HTML)", report.single_report(rec, res, st.session_state.user["full_name"], sug).encode(),
                           f"risk_report_{pid}.html", "text/html")


def page_batch():
    st.title("Batch Prediction")
    st.download_button("Download CSV template", template_csv().encode(), "batch_template.csv", "text/csv")
    st.caption("Required column: source. Fill only the fields that belong to each record type; others can stay empty.")
    up = st.file_uploader("Upload student CSV", type="csv")
    if up is None:
        return
    raw = pd.read_csv(up)
    try:
        out, _ = service().predict_frame(raw)
    except ValueError as e:
        st.error(str(e))
        return
    c = st.columns(4)
    c[0].metric("Students", len(out))
    for i, k in enumerate(C.RISK_CLASSES):
        c[i + 1].metric(k, int((out["predicted_risk"] == k).sum()))
    st.dataframe(out, width="stretch")
    st.plotly_chart(px.histogram(out, x="predicted_risk", color="predicted_risk", color_discrete_map=RISK_COLORS, category_orders={"predicted_risk": C.RISK_CLASSES}), width="stretch")
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(), "batch_predictions.csv", "text/csv")
    if st.button("Save this batch to prediction history"):
        uid = st.session_state.user["user_id"]
        bid = database.create_batch(uid, up.name, len(out), int((out["predicted_risk"] == "High").sum()))
        prep = service().prepare(raw)
        for i in range(len(out)):
            r = {k: (None if pd.isna(prep.iloc[i][k]) else prep.iloc[i][k]) for k in C.FEATURES}
            res = {"pred_class": out.iloc[i]["predicted_risk"], "proba": out.iloc[i][["prob_low", "prob_medium", "prob_high"]].values.astype(float),
                   "text": "Top risk factors: " + str(out.iloc[i]["top_risk_factors"])}
            ref = str(raw.iloc[i]["student_ref"]) if "student_ref" in raw else None
            database.save_prediction(uid, r, res, batch_id=bid, external_ref=ref)
        database.log(uid, "predict_batch", {"batch_id": bid, "rows": len(out)})
        st.success(f"Saved batch #{bid} with {len(out)} predictions.")


def page_history():
    st.title("Prediction History")
    u = st.session_state.user
    mine = st.toggle("Only my predictions", value=u["role"] == "teacher", disabled=u["role"] == "teacher")
    h = pd.DataFrame(database.history(u["user_id"] if mine else None))
    if h.empty:
        st.info("No predictions saved yet.")
        return
    f = st.multiselect("Filter by risk", C.RISK_CLASSES, default=C.RISK_CLASSES)
    h = h[h["predicted_risk"].isin(f)]
    st.dataframe(h, width="stretch", hide_index=True)
    st.download_button("Export history (CSV)", h.to_csv(index=False).encode(), "prediction_history.csv", "text/csv")
    pid = st.selectbox("View stored explanation for prediction", h["prediction_id"].tolist())
    fac = database.factors_for(pid)
    if fac:
        st.dataframe(pd.DataFrame(fac), width="stretch", hide_index=True)
    else:
        st.caption("No factor detail stored for this prediction (batch predictions keep the top factors in the summary only).")
    if u["role"] == "admin" and st.button(f"Delete prediction #{pid} and its student record"):
        database.delete_prediction(pid)
        database.log(u["user_id"], "delete_prediction", {"prediction_id": pid})
        st.rerun()


def page_dashboard():
    st.title("Dashboard")
    u = st.session_state.user
    own = u["role"] == "teacher"
    uid = u["user_id"] if own else None
    st.caption("Your own predictions only." if own else "Predictions saved by all users.")
    s = database.stats(uid)
    am = database.active_model()
    c = st.columns(4)
    c[0].metric("Predictions saved", s["predictions"])
    c[1].metric("High-risk students flagged", s["high"],
                f"{s['high'] / s['predictions']:.0%} of saved" if s["predictions"] else None, delta_color="off")
    c[2].metric("Batch uploads", s["batches"])
    c[3].metric("Model in use", am["name"] if am else "Not registered",
                f"test macro-F1 {am['test_f1_macro']:.3f}" if am else None, delta_color="off")
    if s["predictions"] == 0:
        st.info("No predictions saved yet. Use Risk Prediction or Batch Prediction; results will appear here.")
        return
    left, right = st.columns([1, 1.6])
    with left:
        d = pd.DataFrame({"Risk": C.RISK_CLASSES, "Students": [s["by_risk"][k] for k in C.RISK_CLASSES]})
        f = px.pie(d, names="Risk", values="Students", hole=0.55, color="Risk", color_discrete_map=RISK_COLORS,
                   category_orders={"Risk": C.RISK_CLASSES}, title="Saved predictions by risk level")
        f.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(f, width="stretch")
    with right:
        t = pd.DataFrame(database.daily_counts(uid))
        if t.empty:
            st.caption("No predictions in the last 30 days.")
        else:
            f = px.bar(t, x="day", y="n", color="predicted_risk", color_discrete_map=RISK_COLORS,
                       category_orders={"predicted_risk": C.RISK_CLASSES}, title="Predictions per day (last 30 days)")
            f.update_layout(height=340, xaxis_title="", yaxis_title="Predictions", legend_title="", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(f, width="stretch")
    st.subheader("Early-alert list")
    st.caption("Most recent students predicted High risk, highest probability first. Open Prediction History to see each explanation.")
    al = pd.DataFrame(database.early_alerts(uid))
    if al.empty:
        st.success("No High-risk students in the saved predictions.")
    else:
        al["source"] = al["source"].map(C.SOURCE_LABELS).fillna(al["source"])
        st.dataframe(al.rename(columns={"prediction_id": "Prediction #", "created_at": "Date", "external_ref": "Student ref",
                                        "source": "Record type", "prob_high": "P(High)", "username": "Saved by"}),
                     width="stretch", hide_index=True)


def page_users():
    st.title("User Management")
    users = pd.DataFrame(database.list_users())
    c = st.columns(4)
    c[0].metric("Users", len(users))
    for i, r in enumerate(["admin", "counsellor", "teacher"]):
        c[i + 1].metric(r.capitalize() + "s", int((users["role"] == r).sum()))
    st.dataframe(users, width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        with st.form("newuser", clear_on_submit=True):
            st.write("**Create user**")
            un = st.text_input("Username")
            fn = st.text_input("Full name")
            em = st.text_input("Email")
            ro = st.selectbox("Role", ["teacher", "counsellor", "admin"])
            pw = st.text_input("Temporary password (min 8 characters)", type="password")
            if st.form_submit_button("Create user", type="primary"):
                if not un.strip() or not fn.strip():
                    st.error("Username and full name are required.")
                else:
                    try:
                        database.create_user(un, fn, ro, pw, em or None)
                        database.log(st.session_state.user["user_id"], "create_user", {"username": un, "role": ro})
                        st.success(f"User '{un.strip().lower()}' created.")
                    except Exception as e:
                        st.error("Username already exists." if ("UNIQUE" in str(e) or "Duplicate" in str(e)) else str(e))
    with right:
        st.write("**Activate or deactivate a user**")
        ulist = database.list_users()
        sel = st.selectbox("User", [x["user_id"] for x in ulist], format_func=lambda i: next(x["username"] for x in ulist if x["user_id"] == i))
        b = st.columns(2)
        if b[0].button("Deactivate", width="stretch"):
            if sel == st.session_state.user["user_id"]:
                st.error("You cannot deactivate your own account.")
            else:
                database.set_active(sel, False)
                database.log(st.session_state.user["user_id"], "deactivate_user", {"user_id": sel})
                st.rerun()
        if b[1].button("Activate", width="stretch"):
            database.set_active(sel, True)
            database.log(st.session_state.user["user_id"], "activate_user", {"user_id": sel})
            st.rerun()


def page_audit():
    st.title("Audit Log")
    st.caption("Every sign-in, failed sign-in, prediction, batch, cleaning run, deletion and account change is recorded here.")
    c = st.columns(2)
    act = c[0].selectbox("Action", ["All"] + database.audit_actions())
    who = c[1].selectbox("User", ["All"] + [x["username"] for x in database.list_users()])
    log = pd.DataFrame(database.audit(limit=1000, action=None if act == "All" else act, username=None if who == "All" else who))
    if log.empty:
        st.info("No entries for this filter.")
        return
    m = st.columns(3)
    m[0].metric("Entries shown", len(log))
    m[1].metric("Failed sign-ins", int((log["action"] == "failed_login").sum()))
    m[2].metric("Predictions logged", int(log["action"].isin(["predict_single", "predict_batch"]).sum()))
    st.dataframe(log, width="stretch", hide_index=True)
    st.download_button("Export audit log (CSV)", log.to_csv(index=False).encode(), "audit_log.csv", "text/csv")


def page_account():
    st.title("My Account")
    u = st.session_state.user
    st.write(f"**{u['full_name']}** · {u['username']} · role: {u['role']}")
    if u["role"] == "admin":
        st.caption(f"Database in use: {database.describe()}")
    with st.form("pw"):
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password (min 8 characters)", type="password")
        new2 = st.text_input("Repeat new password", type="password")
        if st.form_submit_button("Change password"):
            if not database.authenticate(u["username"], old):
                st.error("Current password is wrong.")
            elif new != new2:
                st.error("New passwords do not match.")
            else:
                try:
                    database.change_password(u["user_id"], new)
                    database.log(u["user_id"], "change_password")
                    st.success("Password changed.")
                except ValueError as e:
                    st.error(str(e))


VIEWS = {"Dashboard": page_dashboard, "Risk Prediction": page_predict, "Batch Prediction": page_batch,
         "Prediction History": page_history, "Data Management": page_data, "Exploratory Data Analysis": page_eda,
         "Model Evaluation": page_models, "Explainability": page_explain, "Fairness Audit": page_fairness,
         "User Management": page_users, "Audit Log": page_audit, "My Account": page_account}


def header_bar(u):
    left, right = st.columns([5, 1.2], vertical_alignment="center")
    left.markdown(f"**Academic Risk Predictor**  |  {u['full_name']} ({u['role']})")
    if right.button("Sign out", icon=":material/logout:", width="stretch"):
        database.log(u["user_id"], "logout")
        del st.session_state["user"]
        st.rerun()


def main():
    _loading.empty()
    boot()
    if "user" not in st.session_state:
        st.navigation([st.Page(login_view, title="Sign In", url_path="sign-in")], position="hidden").run()
        return
    require_artifacts()
    u = st.session_state.user
    allowed = [p for p in PAGES if u["role"] in p[4]]
    start = st.session_state.get("start_page", allowed[0][1])
    sections = {}
    for fid, title, group, icon, roles in allowed:
        sections.setdefault(group, []).append(
            st.Page(VIEWS[title], title=title, icon=icon, url_path=title.lower().replace(" ", "-"), default=(title == start)))
    nav = st.navigation(sections, position="sidebar")
    header_bar(u)
    nav.run()


try:
    main()
except Exception as e:
    if type(e).__name__ in ("StopException", "RerunException", "RerunData"):
        raise
    _loading.empty()
    st.error(f"The app hit an error: {e}")
    st.code(traceback.format_exc())
    st.info("Copy the error above and send it for a fix. If it mentions a scikit-learn or joblib version, run: python run_pipeline.py")
