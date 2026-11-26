"""Suggested interventions for the Risk Prediction functionality (F03).

Each risk-increasing factor found by SHAP is mapped to a practical action a teacher or counsellor can take.
The suggestions are advice for staff, never automatic decisions about the student.
"""
import pandas as pd

# feature label (as shown by explain.pretty_name) -> (support area, suggested action)
ACTIONS = {
    "Attendance %": ("Attendance", "Contact the student and parents about attendance; set a weekly attendance target and review it."),
    "Absences": ("Attendance", "Track absences weekly and arrange a counsellor meeting to find the cause of repeated absence."),
    "Absence days": ("Attendance", "Track absence days and agree an attendance plan with the student and parents."),
    "Hours studied per week": ("Study habits", "Agree a weekly study timetable and check it in mentoring sessions."),
    "Weekly study time": ("Study habits", "Agree a weekly study timetable and check it in mentoring sessions."),
    "Period-1 grade": ("Academic support", "Offer remedial classes for the topics where the first-period marks were low."),
    "Period-2 grade": ("Academic support", "Offer remedial classes and an extra internal test before the final exam."),
    "Previous scores": ("Academic support", "Review earlier weak topics and assign practice work with feedback."),
    "Past failures": ("Academic support", "Assign a faculty mentor and plan support for the subjects failed before."),
    "Tutoring sessions": ("Tutoring", "Enrol the student in peer tutoring or extra tutoring sessions."),
    "Tutoring (0 = no, 1 = yes)": ("Tutoring", "Enrol the student in peer tutoring or extra tutoring sessions."),
    "Motivation level": ("Counselling", "Refer to the counsellor for motivation and goal-setting sessions."),
    "Parental involvement": ("Family engagement", "Hold a parent-teacher meeting and share a simple progress plan."),
    "Parental support": ("Family engagement", "Hold a parent-teacher meeting and share a simple progress plan."),
    "Internet access": ("Resources", "Give access to the college library, lab computers or offline study material."),
    "Sleep hours": ("Well-being", "Discuss sleep and workload with the student; refer to the counsellor if needed."),
    "Times raised hand": ("Class engagement", "Encourage class participation through small-group and question activities."),
    "Discussion posts": ("Class engagement", "Encourage participation in LMS discussions with short weekly tasks."),
    "Resources visited": ("Class engagement", "Share key LMS resources and track that the student opens them."),
    "Announcements viewed": ("Class engagement", "Remind the student to follow course announcements; send alerts by SMS or email."),
    "Test preparation": ("Academic support", "Enrol the student in a test-preparation programme."),
    "Family income": ("Financial support", "Inform the student about scholarships, fee concessions and free resources."),
    "Lunch type": ("Financial support", "Inform the student about scholarships, fee concessions and free resources."),
}

GENERAL = {
    "High": "Schedule a counsellor review within one week and monitor progress every week.",
    "Medium": "Monitor progress every two weeks and offer light-touch support.",
    "Low": "No intervention needed now; recheck when new marks or attendance data arrive.",
}


def suggest(risk_increasing, pred_class, top_k=3):
    """Return a table of suggested interventions for the top risk-increasing factors.

    risk_increasing: DataFrame with columns label, value, shap_high (from explain.explain_one).
    """
    rows = []
    seen = set()
    for r in risk_increasing.itertuples():
        if r.label not in ACTIONS:
            continue
        area, action = ACTIONS[r.label]
        if action in seen:
            continue
        seen.add(action)
        rows.append({"Factor": r.label, "Value": r.value, "Support area": area, "Suggested action": action})
        if len(rows) == top_k:
            break
    rows.append({"Factor": "Overall", "Value": pred_class, "Support area": "Follow-up", "Suggested action": GENERAL[pred_class]})
    return pd.DataFrame(rows)
