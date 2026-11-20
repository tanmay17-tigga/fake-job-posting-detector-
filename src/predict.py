"""
predict.py
----------
Check one job ad from the command line.

Examples (run from the project folder, after python src/train.py):

    python src/predict.py --description "Earn $5000 a week from home! No experience needed."
    python src/predict.py --title "Data Analyst" --company_profile "We are a fintech startup..." \
        --description "You will build dashboards in SQL and Python." --has_company_logo 1
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocess import build_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fake_job_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run 'python src/train.py' first.")
    return joblib.load(MODEL_PATH)


def predict_one(bundle, ad: dict) -> tuple[float, bool]:
    """Return (probability the ad is fake, True/False decision)."""
    features = build_features(pd.DataFrame([ad]))
    proba = float(bundle["pipeline"].predict_proba(features)[0, 1])
    return proba, proba >= bundle["threshold"]


def main():
    p = argparse.ArgumentParser(description="Check if a job ad looks fake.")
    for field in ["title", "company_profile", "description", "requirements",
                  "benefits", "salary_range", "location"]:
        p.add_argument(f"--{field}", default="")
    p.add_argument("--has_company_logo", type=int, default=0, choices=[0, 1])
    p.add_argument("--has_questions", type=int, default=0, choices=[0, 1])
    p.add_argument("--telecommuting", type=int, default=0, choices=[0, 1])
    args = vars(p.parse_args())

    bundle = load_model()
    proba, is_fake = predict_one(bundle, args)
    label = "LIKELY FAKE" if is_fake else "LIKELY REAL"
    print(f"{label}  (chance of being fake: {proba:.1%}, cut-off: {bundle['threshold']:.0%})")


if __name__ == "__main__":
    main()
