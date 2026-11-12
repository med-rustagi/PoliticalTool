#!/usr/bin/env python3
"""Predict political lean for NEW text with a trained model.

Usage:
  python3 predict.py --model models/tfidf_baseline.joblib --text "Healthcare is a human right #vote"
  python3 predict.py --model models/tfidf_baseline.joblib --input new_posts.csv --text-col caption --out results/new_predictions.csv
  python3 predict.py --model models/embedding_logreg.joblib --text "..."     # also needs torch + transformers

The output is the model's ESTIMATE from patterns in its training data, with probabilities.
It is not a verdict on whether a post is biased or true.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from clean_text import clean_text
from common import embed_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a .joblib file from baseline_tfidf.py or train_classifier.py")
    ap.add_argument("--text", default=None, help="a single piece of text")
    ap.add_argument("--input", default=None, help="a CSV with many texts")
    ap.add_argument("--text-col", default="caption")
    ap.add_argument("--out", default=None, help="where to save predictions for --input")
    a = ap.parse_args()

    if bool(a.text) == bool(a.input):
        raise SystemExit("Give exactly one of --text or --input.")
    if a.text:
        raw = [a.text]
    else:
        src = pd.read_csv(a.input)
        raw = src[a.text_col].fillna("").astype(str).tolist()
    texts = [clean_text(t) for t in raw]   # same cleaning as training

    bundle = joblib.load(a.model)
    clf, classes = bundle["classifier"], bundle["classes"]
    if "vectorizer" in bundle:
        proba = clf.predict_proba(bundle["vectorizer"].transform(texts))
    else:
        X = embed_texts(texts, bundle["embedding_model"], normalize=bundle.get("normalized", True))
        proba = clf.predict_proba(X)

    out = pd.DataFrame({"text": raw, "prediction": [classes[i] for i in proba.argmax(axis=1)],
                        "confidence": proba.max(axis=1).round(3)})
    for k, c in enumerate(classes):
        out["p_" + c] = proba[:, k].round(3)

    if a.input and a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print("Saved %d predictions to %s" % (len(out), a.out))
    else:
        for _, r in out.head(20).iterrows():
            probs = ", ".join("%s=%.2f" % (c, r["p_" + c]) for c in classes)
            print("%-8s (confidence %.2f)  [%s]\n   %s" % (r["prediction"], r["confidence"], probs, r["text"][:120]))
    print("\nNote: a model estimate from patterns in training data, not a verdict on the post.")


if __name__ == "__main__":
    main()
