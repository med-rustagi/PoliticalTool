#!/usr/bin/env python3
"""Show WHICH WORDS push the TF-IDF baseline toward each label.

Usage:
  python3 explain.py --model models/tfidf_baseline.joblib --top 15            # words most tied to each label
  python3 explain.py --model models/tfidf_baseline.joblib --text "Secure the border #vote"   # why this prediction

Works with the TF-IDF model only: embedding dimensions are not human-readable.

Read the results carefully. These are words that go with each label in the TRAINING data,
not proof of bias. With account-level labels, top words are often names, hashtags, or
catchphrases of specific accounts, which signals account style rather than political content.
"""
import argparse

import joblib
import numpy as np

from clean_text import clean_text


def class_coef(clf, k):
    """Weight vector for class index k (handles the binary case, which stores one vector)."""
    if len(clf.classes_) == 2:
        return clf.coef_[0] if k == 1 else -clf.coef_[0]
    return clf.coef_[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="models/tfidf_baseline.joblib")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--text", default=None, help="explain the prediction for this text")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    if "vectorizer" not in bundle:
        raise SystemExit("explain.py only works with the TF-IDF model (models/tfidf_baseline.joblib).")
    vec, clf = bundle["vectorizer"], bundle["classifier"]
    names = np.array(vec.get_feature_names_out())

    if a.text is None:
        for k, c in enumerate(clf.classes_):
            top = np.argsort(class_coef(clf, k))[::-1][:a.top]
            print("Words most associated with '%s':\n  %s\n" % (c, ", ".join(names[top])))
    else:
        x = vec.transform([clean_text(a.text)])
        proba = clf.predict_proba(x)[0]
        k = int(proba.argmax())
        print("Prediction: %s (confidence %.2f)" % (clf.classes_[k], proba[k]))
        contrib = x.toarray()[0] * class_coef(clf, k)
        order = np.argsort(contrib)[::-1]
        pos = [(names[i], contrib[i]) for i in order if contrib[i] > 0][:a.top]
        neg = [(names[i], contrib[i]) for i in order[::-1] if contrib[i] < 0][:a.top]
        print("\nWords pushing TOWARD '%s':" % clf.classes_[k])
        print("  " + (", ".join("%s (%+.2f)" % p for p in pos) if pos else "none found in the vocabulary"))
        print("Words pushing AWAY from it:")
        print("  " + (", ".join("%s (%+.2f)" % p for p in neg) if neg else "none"))
        print("\nWords not seen during training are ignored. This shows word associations, not proof of bias.")


if __name__ == "__main__":
    main()
