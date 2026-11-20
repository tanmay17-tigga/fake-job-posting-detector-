"""
train.py
--------
Trains and compares three models, picks the best one, and saves it.

Run from the project folder:

    python src/train.py

What it does, step by step:
  1. Load the data, build features (preprocess.py), remove repeated ads.
  2. Split into train (70%), validation (15%) and test (15%).
  3. Train 3 models on the train set:
        - Naive Bayes
        - Logistic Regression
        - Random Forest
  4. For each model, find the best cut-off (threshold) on the validation set.
  5. Pick the model with the highest F1 score on validation.
  6. Check the chosen model ONE time on the test set (the final, honest score).
  7. Save the model, the scores and some charts.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from plot_style import FAKE_COLOR, MUTED, REAL_COLOR, apply_style
from preprocess import NUMERIC_COLS, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42  # fixed number so results are the same every run


def make_preprocessor(max_features: int) -> ColumnTransformer:
    """TF-IDF for the text + pass the 0/1 columns through unchanged."""
    return ColumnTransformer(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),      # single words and word pairs
                    min_df=2,                # ignore words seen in only 1 ad
                    max_df=0.9,              # ignore words in >90% of ads
                    max_features=max_features,
                    sublinear_tf=True,
                    stop_words="english",
                ),
                "text",
            ),
            ("flags", "passthrough", NUMERIC_COLS),
        ]
    )


def make_models() -> dict:
    return {
        "Naive Bayes": Pipeline(
            [("prep", make_preprocessor(50_000)), ("model", MultinomialNB(alpha=0.1))]
        ),
        "Logistic Regression": Pipeline(
            [
                ("prep", make_preprocessor(50_000)),
                (
                    "model",
                    LogisticRegression(
                        C=10,
                        class_weight="balanced",  # give the rare fake class more weight
                        max_iter=3000,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", make_preprocessor(20_000)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        min_samples_leaf=1,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def best_threshold(y_true, proba) -> float:
    """Try many cut-offs and return the one with the highest F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall have one more value than thresholds; drop the last one
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def score(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    # 1. Load and prepare
    df = pd.read_csv(DATA_PATH)
    X = build_features(df)
    y = df["fraudulent"].astype(int).values

    # Remove repeated ads (same text). If a copy of an ad is in train AND
    # test, the model has "seen the answer" and the test score looks too good.
    keep = ~X.duplicated(subset=["text"]).values
    print(f"Removed {(~keep).sum()} repeated ads; {keep.sum():,} ads left")
    X, y = X[keep].reset_index(drop=True), y[keep]

    # 2. Split: 70% train, 15% validation, 15% test.
    #    stratify=y keeps the same ~4.8% fake share in every part.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {len(y_train):,}  Validation: {len(y_val):,}  Test: {len(y_test):,}")

    # 3 + 4. Train each model, tune its threshold on validation
    results, fitted = [], {}
    for name, pipe in make_models().items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        default = score(y_val, val_proba, 0.5)
        thr = best_threshold(y_val, val_proba)
        tuned = score(y_val, val_proba, thr)
        results.append(
            {
                "model": name,
                "f1_at_0.5": default["f1"],
                **{f"val_{k}": v for k, v in tuned.items()},
            }
        )
        fitted[name] = (pipe, thr)
        print(f"   F1 at 0.5 = {default['f1']:.3f} | tuned threshold {thr:.3f} -> F1 = {tuned['f1']:.3f}")

    comparison = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nValidation comparison:")
    print(comparison.to_string(index=False))

    # 5. Pick the best model by validation F1
    best_name = comparison.iloc[0]["model"]
    best_pipe, best_thr = fitted[best_name]
    print(f"\nBest model: {best_name} (threshold {best_thr:.3f})")

    # 6. Final check on the untouched test set
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_scores = score(y_test, test_proba, best_thr)
    test_pred = (test_proba >= best_thr).astype(int)
    cm = confusion_matrix(y_test, test_pred)
    print("Test scores:", test_scores)
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())

    # 7. Save everything
    joblib.dump(
        {"pipeline": best_pipe, "threshold": best_thr, "model_name": best_name},
        MODEL_DIR / "fake_job_model.joblib",
    )
    metrics = {
        "best_model": best_name,
        "threshold": round(best_thr, 4),
        "test": test_scores,
        "confusion_matrix": {
            "true_real_pred_real": int(cm[0, 0]),
            "true_real_pred_fake": int(cm[0, 1]),
            "true_fake_pred_real": int(cm[1, 0]),
            "true_fake_pred_fake": int(cm[1, 1]),
        },
        "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Chart: confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.grid(False)
    ax.set_title(f"Test set results ({best_name})")
    ax.set_xlabel("What the model said")
    ax.set_ylabel("The real answer")
    fig.savefig(FIG_DIR / "05_confusion_matrix.png")
    plt.close(fig)

    # Chart: precision-recall curve on the test set
    precision, recall, _ = precision_recall_curve(y_test, test_proba)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color=REAL_COLOR, linewidth=2)
    ax.scatter([test_scores["recall"]], [test_scores["precision"]], s=70,
               color=FAKE_COLOR, zorder=3, label=f"Chosen threshold ({best_thr:.2f})")
    ax.set_xlabel("Recall (share of fake ads we catch)")
    ax.set_ylabel("Precision (share of our 'fake' alerts that are right)")
    ax.set_title("Precision vs recall trade-off (test set)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIG_DIR / "06_precision_recall_curve.png")
    plt.close(fig)

    # Chart: model comparison
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    order = comparison.sort_values("val_f1")
    ax.barh(order["model"], order["val_f1"], color=REAL_COLOR, height=0.5)
    for i, v in enumerate(order["val_f1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score on validation set (higher is better)")
    ax.set_title("Model comparison")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "07_model_comparison.png")
    plt.close(fig)

    # Chart: words that push towards "fake" / "real" (Logistic Regression)
    lr_pipe = fitted["Logistic Regression"][0]
    names = lr_pipe.named_steps["prep"].get_feature_names_out()
    coefs = lr_pipe.named_steps["model"].coef_[0]
    words = pd.Series(coefs, index=[n.replace("tfidf__", "").replace("flags__", "[flag] ") for n in names])
    top_fake = words.sort_values(ascending=False).head(15)
    top_real = words.sort_values().head(15)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(top_fake.index[::-1], top_fake.values[::-1], color=FAKE_COLOR, height=0.6)
    axes[0].set_title("Words that push towards FAKE")
    axes[1].barh(top_real.index[::-1], -top_real.values[::-1], color=REAL_COLOR, height=0.6)
    axes[1].set_title("Words that push towards REAL")
    for ax in axes:
        ax.set_xlabel("Strength (model weight)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_top_words.png")
    plt.close(fig)
    pd.DataFrame({"fake_word": top_fake.index, "fake_weight": top_fake.values.round(3),
                  "real_word": top_real.index, "real_weight": top_real.values.round(3)}
                 ).to_csv(REPORT_DIR / "top_words.csv", index=False)

    print(f"\nSaved model to {MODEL_DIR / 'fake_job_model.joblib'}")
    print(f"Saved scores and charts to {REPORT_DIR}")


if __name__ == "__main__":
    main()
