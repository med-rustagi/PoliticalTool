#!/usr/bin/env python3
"""Baseline political-bias classifier: TF-IDF word features + logistic regression.

Needs no embeddings, no GPU, and no downloads, so it runs in seconds. Run this FIRST:
any fancier model has to beat this score to be worth using.

Usage:
  python3 baseline_tfidf.py --clean clean/clean_text.csv

Outputs:
  models/tfidf_baseline.joblib         model (used by predict.py and explain.py)
  results/predictions_tfidf.csv        one row per post, all splits (used by evaluate.py)
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import config
from common import load_clean, print_split_metrics, save_predictions, compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=config.CLEAN_CSV)
    ap.add_argument("--model-dir", default=config.MODEL_DIR)
    ap.add_argument("--results-dir", default=config.RESULTS_DIR)
    ap.add_argument("--C", type=float, default=1.0, help="regularization strength (smaller = simpler model)")
    ap.add_argument("--max-features", type=int, default=50000)
    a = ap.parse_args()

    df = load_clean(a.clean)
    classes = sorted(df["label"].unique())
    train = df[df["split"] == "train"]
    if train["label"].nunique() < 2:
        raise SystemExit("The training split needs at least 2 different labels. Check clean_text.csv.")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if len(train) >= 200 else 1,
                          sublinear_tf=True, max_features=a.max_features,
                          token_pattern=r"(?u)[#@]?\w\w+")   # keeps #hashtags and @mentions as features
    clf = LogisticRegression(C=a.C, max_iter=2000, class_weight="balanced", random_state=config.SEED)
    Xtr = vec.fit_transform(train["caption"])
    clf.fit(Xtr, train["label"])

    proba = clf.predict_proba(vec.transform(df["caption"]))
    pred = clf.classes_[proba.argmax(axis=1)]
    print("TF-IDF baseline (classes: %s)" % ", ".join(classes))
    print_split_metrics(df, pred, list(clf.classes_))

    test = df[df["split"] == "test"]
    if len(test):
        maj = train["label"].value_counts().idxmax()
        print("  For context, always guessing '%s' scores accuracy=%.3f on test." %
              (maj, float((test["label"] == maj).mean())))

    Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "classifier": clf, "classes": list(clf.classes_)},
                Path(a.model_dir) / "tfidf_baseline.joblib")
    save_predictions(df, pred, proba, list(clf.classes_), Path(a.results_dir) / "predictions_tfidf.csv")
    print("Saved model to %s/ and predictions to %s/" % (a.model_dir, a.results_dir))


if __name__ == "__main__":
    main()
