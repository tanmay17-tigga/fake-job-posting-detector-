# Fake Job Posting Detector

A machine learning project that reads a job ad and estimates the chance it is **fake** (a scam).
It uses TF-IDF text features plus simple signals (company logo, company profile, screening
questions) and compares three models. The best one (Logistic Regression) is served in a
Streamlit web app.

## Results (test set, 2,618 ads the model never saw)

| Metric | Score |
|---|---|
| Precision (fake class) | 0.86 |
| Recall (fake class) | 0.82 |
| F1 score (fake class) | 0.84 |
| ROC-AUC | 0.99 |
| PR-AUC | 0.93 |

Model comparison on the validation set (F1 for the fake class, best threshold per model):

| Model | F1 |
|---|---|
| Logistic Regression | **0.89** |
| Random Forest | 0.81 |
| Naive Bayes | 0.80 |

Out of 128 fake ads in the test set, the model caught 105 and missed 23.
Out of 2,490 real ads, only 17 were wrongly flagged as fake.

## Dataset

[Real or Fake: Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
(EMSCAD, University of the Aegean): 17,880 job ads, 866 of them fake (4.8%).
A copy is included in `data/fake_job_postings.csv`.

## Project structure

```
fake-job-posting-detector/
├── app.py                  # Streamlit web app
├── docs/                   # detailed project guide (PDF)
├── requirements.txt
├── data/
│   └── fake_job_postings.csv
├── src/
│   ├── preprocess.py       # text cleaning + simple 0/1 features
│   ├── eda.py              # charts that explore the data
│   ├── train.py            # trains 3 models, picks the best, saves it
│   ├── predict.py          # check one ad from the command line
│   └── plot_style.py       # chart colors
├── models/                 # saved model (created by train.py)
└── reports/
    ├── metrics.json        # final test scores
    ├── model_comparison.csv
    ├── top_words.csv
    └── figures/            # all charts
```

## How to run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 2. install libraries
pip install -r requirements.txt

# 3. explore the data (saves charts in reports/figures)
python src/eda.py

# 4. train the models (about 1-2 minutes)
python src/train.py

# 5. check one ad from the terminal
python src/predict.py --title "Data Entry - Work From Home" --description "Earn $3000 a week from home, no experience needed"

# 6. open the web app
streamlit run app.py
```

## Method in short

1. Join all text fields of an ad into one text, clean it (lowercase, keep letters only,
   turn hidden links/emails/phone numbers into the words `urltoken`, `emailtoken`, `phonetoken`).
2. Add 0/1 signals: has logo, has questions, work from home, and whether the company profile,
   salary, requirements or benefits are empty.
3. Remove 430 repeated ads so the same ad cannot be in both train and test.
4. Split 70% train / 15% validation / 15% test, keeping the same fake share in each.
5. TF-IDF (single words and word pairs) + three models; class weights handle the imbalance.
6. Pick the decision cut-off (threshold) with the best F1 on validation, then score once on test.

## Key findings

- Ads with **no company profile** are fake 17.7% of the time vs 1.9% for ads with one.
- Ads **without a company logo** are fake 15.9% of the time vs 2.0% with a logo.
- Fake ads are shorter (median 244 words vs 378).
- Words like "administrative", "data entry", "earn", "money" and "apply using link" push
  towards fake; words like "software", "client", "team", "English" push towards real.

## Limitations

- The data comes from 2012-2014 and about 60% of the ads are from the US, so scam styles today
  (and in India) may differ.
- The model only reads words; it cannot check if a company really exists.
- It should be used as a helper that flags ads for a human to check, not as a final judge.
