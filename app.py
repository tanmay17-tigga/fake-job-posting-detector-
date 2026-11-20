"""
app.py - Streamlit web app for the Fake Job Posting Detector.

Run from the project folder (after python src/train.py):

    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import MODEL_PATH, predict_one  # noqa: E402

import joblib  # noqa: E402

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🔎", layout="centered")

EXAMPLES = {
    "Write my own": {},
    "Example: suspicious ad": {
        "title": "Data Entry Clerk - Work From Home",
        "company_profile": "",
        "description": "Earn $3000 per week working from home! No experience needed. "
                       "Just a computer and internet. Apply using the link below and send "
                       "your bank details to start earning money today.",
        "requirements": "",
        "benefits": "Weekly pay. Flexible hours.",
        "has_company_logo": False,
        "has_questions": False,
        "telecommuting": True,
    },
    "Example: normal ad": {
        "title": "Junior Data Analyst",
        "company_profile": "We are a growing software company that builds analytics tools "
                           "for retail clients across Europe and Asia.",
        "description": "You will work with our product and client teams to build dashboards, "
                       "clean data in SQL and Python, and present insights to stakeholders.",
        "requirements": "Bachelor's degree in a quantitative field. Good SQL and Excel skills. "
                        "Comfortable communicating in English.",
        "benefits": "Health insurance, learning budget, hybrid working.",
        "has_company_logo": True,
        "has_questions": True,
        "telecommuting": False,
    },
}


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


st.title("🔎 Fake Job Posting Detector")
st.write(
    "Paste the details of a job ad. The model reads the words and a few simple "
    "signals (logo, company profile, questions) and estimates the chance the ad is fake."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Open a terminal in the project folder and run: "
             "`python src/train.py`, then refresh this page.")
    st.stop()

bundle = get_model()

choice = st.selectbox("Start from", list(EXAMPLES.keys()))
ex = EXAMPLES[choice]

with st.form("job_form"):
    title = st.text_input("Job title", ex.get("title", ""))
    company_profile = st.text_area("Company profile (leave empty if the ad has none)",
                                   ex.get("company_profile", ""), height=80)
    description = st.text_area("Job description", ex.get("description", ""), height=140)
    requirements = st.text_area("Requirements", ex.get("requirements", ""), height=80)
    benefits = st.text_area("Benefits", ex.get("benefits", ""), height=68)
    salary_range = st.text_input("Salary range (optional, e.g. 30000-40000)", "")
    c1, c2, c3 = st.columns(3)
    has_logo = c1.checkbox("Has company logo", ex.get("has_company_logo", False))
    has_q = c2.checkbox("Has screening questions", ex.get("has_questions", False))
    remote = c3.checkbox("Work from home", ex.get("telecommuting", False))
    submitted = st.form_submit_button("Check this ad", type="primary")

if submitted:
    if not (title + description + requirements + company_profile).strip():
        st.warning("Please type at least a title or a description.")
    else:
        ad = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "salary_range": salary_range,
            "has_company_logo": int(has_logo),
            "has_questions": int(has_q),
            "telecommuting": int(remote),
        }
        proba, is_fake = predict_one(bundle, ad)
        if is_fake:
            st.error(f"⚠️ Likely FAKE — chance of being fake: **{proba:.1%}**")
        else:
            st.success(f"✅ Likely REAL — chance of being fake: **{proba:.1%}**")
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(
            f"The ad is called fake when the chance is at least {bundle['threshold']:.0%}. "
            "This cut-off was picked on the validation set to balance catching fake ads "
            "and not raising false alarms. Always double-check before trusting any result."
        )

with st.expander("How good is the model? (test set results)"):
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        t = m["test"]
        a, b, c = st.columns(3)
        a.metric("Precision", f"{t['precision']:.0%}")
        b.metric("Recall", f"{t['recall']:.0%}")
        c.metric("F1 score", f"{t['f1']:.2f}")
        cm = m["confusion_matrix"]
        st.dataframe(
            pd.DataFrame(
                [[cm["true_real_pred_real"], cm["true_real_pred_fake"]],
                 [cm["true_fake_pred_real"], cm["true_fake_pred_fake"]]],
                index=["Really real", "Really fake"],
                columns=["Model said real", "Model said fake"],
            )
        )
        st.caption(f"Model: {m['best_model']}. Test set = {m['split_sizes']['test']:,} ads "
                   "the model never saw during training.")
    fig = ROOT / "reports" / "figures" / "08_top_words.png"
    if fig.exists():
        st.image(str(fig), caption="Words the model links with fake and real ads")
