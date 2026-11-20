"""
eda.py  (Exploratory Data Analysis)
-----------------------------------
Looks at the data BEFORE building any model and saves charts to
reports/figures/. Run from the project folder:

    python src/eda.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import FAKE_COLOR, MUTED, REAL_COLOR, apply_style
from preprocess import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
FIG_DIR = ROOT / "reports" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")

    # ---------- 1. How many fake vs real? ----------
    counts = df["fraudulent"].value_counts().sort_index()
    fake_share = counts[1] / counts.sum() * 100
    print(f"Real ads: {counts[0]:,}   Fake ads: {counts[1]:,}  ({fake_share:.1f}% fake)")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Real", "Fake"], counts.values, color=[REAL_COLOR, FAKE_COLOR], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 200, f"{value:,}",
                ha="center", color=MUTED)
    ax.set_title(f"Only {fake_share:.1f}% of job ads are fake")
    ax.set_ylabel("Number of job ads")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "01_class_balance.png")
    plt.close(fig)

    # ---------- 2. Missing values per column ----------
    missing = (df.isna().mean() * 100).sort_values()
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(missing.index, missing.values, color=REAL_COLOR, height=0.6)
    ax.set_title("Percent of empty values in each column")
    ax.set_xlabel("% of rows that are empty")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "02_missing_values.png")
    plt.close(fig)

    # ---------- 3. Fake rate by simple yes/no signals ----------
    feats = build_features(df)
    feats["fraudulent"] = df["fraudulent"].values
    signals = {
        "has_company_logo": "Has company logo",
        "no_company_profile": "No company profile",
        "has_questions": "Has screening questions",
        "telecommuting": "Work from home",
        "no_salary_range": "No salary given",
    }
    rows = []
    for col, label in signals.items():
        rate = feats.groupby(col)["fraudulent"].mean() * 100
        rows.append((label, rate.get(0, 0.0), rate.get(1, 0.0)))
        print(f"{label:<26} fake rate when NO: {rate.get(0, 0):5.1f}%   when YES: {rate.get(1, 0):5.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(rows))
    ax.barh([i + 0.2 for i in y], [r[1] for r in rows], height=0.38, color=REAL_COLOR, label="No")
    ax.barh([i - 0.2 for i in y], [r[2] for r in rows], height=0.38, color=FAKE_COLOR, label="Yes")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("% of ads that are fake")
    ax.set_title("Which simple signals go with fake ads?")
    ax.legend(title="Is it true for the ad?", frameon=False)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "03_fake_rate_by_signal.png")
    plt.close(fig)

    # ---------- 4. Text length ----------
    feats["n_words"] = feats["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = range(0, 1600, 40)
    ax.hist(feats.loc[feats.fraudulent == 0, "n_words"], bins=bins, color=REAL_COLOR,
            alpha=0.75, density=True, label="Real")
    ax.hist(feats.loc[feats.fraudulent == 1, "n_words"], bins=bins, color=FAKE_COLOR,
            alpha=0.75, density=True, label="Fake")
    ax.set_title("Fake ads tend to be shorter")
    ax.set_xlabel("Number of words in the ad")
    ax.set_ylabel("Share of ads")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "04_text_length.png")
    plt.close(fig)
    med = feats.groupby("fraudulent")["n_words"].median()
    print(f"Median words  real: {med[0]:.0f}   fake: {med[1]:.0f}")

    print(f"Charts saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
