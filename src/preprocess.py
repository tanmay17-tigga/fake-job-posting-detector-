"""
preprocess.py
-------------
Turns a raw job posting table into model-ready features.

The SAME function (build_features) is used for:
  * training (train.py)
  * single predictions (predict.py and app.py)
so the model always sees data prepared in exactly the same way.
"""

import re

import pandas as pd

# Text columns that we join into one big piece of text per job ad.
TEXT_COLS = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

# Yes/No (1/0) columns that already exist in the dataset.
FLAG_COLS = ["telecommuting", "has_company_logo", "has_questions"]

# Columns where "this field is empty" is itself a useful signal.
# Fake ads often leave the company profile, salary, etc. blank.
MISSING_CHECK_COLS = ["company_profile", "salary_range", "requirements", "benefits"]
MISSING_COLS = [f"no_{c}" for c in MISSING_CHECK_COLS]

NUMERIC_COLS = FLAG_COLS + MISSING_COLS


def clean_text(text: str) -> str:
    """Make text simple and uniform so the model can learn from words."""
    text = str(text).lower()
    # The dataset hides real links, emails and phone numbers as #URL_xxx#,
    # #EMAIL_xxx#, #PHONE_xxx#. We keep the fact that they were there,
    # but as one simple word each.
    text = re.sub(r"#url_[^#]*#", " urltoken ", text)
    text = re.sub(r"#email_[^#]*#", " emailtoken ", text)
    text = re.sub(r"#phone_[^#]*#", " phonetoken ", text)
    # Normal links / emails typed by a user in the app.
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+\.\S+", " emailtoken ", text)
    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)
    # Keep only letters; everything else becomes a space.
    text = re.sub(r"[^a-z]+", " ", text)
    # Squeeze many spaces into one.
    return re.sub(r"\s+", " ", text).strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table with one 'text' column plus simple 0/1 columns."""
    df = df.copy()

    # Make sure every expected column exists (useful for single predictions).
    for col in TEXT_COLS + ["salary_range"]:
        if col not in df.columns:
            df[col] = ""
    for col in FLAG_COLS:
        if col not in df.columns:
            df[col] = 0

    out = pd.DataFrame(index=df.index)

    joined = df[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1)
    out["text"] = joined.map(clean_text)

    for col in FLAG_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in MISSING_CHECK_COLS:
        is_blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        out[f"no_{col}"] = is_blank.astype(int)

    return out
