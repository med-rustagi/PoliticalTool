#!/usr/bin/env python3
"""Text analysis tool: estimate the political lean of text, answer "unclear" when the model
is not confident, and show the words behind the estimate.

Command line (run from the repository root):
  python3 src/text/text_tool.py --text "Healthcare is a human right #vote"
  python3 src/text/text_tool.py --text "..." --json
  python3 src/text/text_tool.py --input posts.csv --text-col caption --out results/analysis.csv

From other code (for example a web backend):
  import sys; sys.path.insert(0, "src/text")
  from text_tool import analyze_text
  result = analyze_text("some caption")      # returns a plain dict

Needs a trained model first: run baseline_tfidf.py (see README_TEXT.md).
The result is a model ESTIMATE from patterns in its training data, not a verdict on the text.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from clean_text import clean_text
from explain import class_coef

DEFAULT_MODEL = str(Path(config.MODEL_DIR) / "tfidf_baseline.joblib")
MIN_CHARS = 3
STATUS = "model_estimate_not_a_verdict"


@lru_cache(maxsize=4)
def load_model(path):
    if not Path(path).exists():
        raise FileNotFoundError("Model not found: %s. Train one first with baseline_tfidf.py." % path)
    return joblib.load(path)


def _unclear(reason, text, closest=None, confidence=None, probs=None, words=None):
    return {"label": "unclear", "closest_label": closest, "confidence": confidence,
            "probabilities": probs or {}, "top_words": words or [], "reason": reason,
            "input": text, "status": STATUS}


def analyze_text(text, model_path=DEFAULT_MODEL, min_confidence=0.65, top_words=5):
    """Return a dict: label (or 'unclear'), confidence, probabilities, top_words, reason."""
    text = "" if text is None else str(text)
    cleaned = clean_text(text)
    if len(cleaned) < MIN_CHARS:
        return _unclear("text is too short to analyze", text)

    bundle = load_model(str(model_path))
    clf, classes = bundle["classifier"], bundle["classes"]
    is_tfidf = "vectorizer" in bundle

    if is_tfidf:
        x = bundle["vectorizer"].transform([cleaned])
        if x.nnz == 0:
            return _unclear("none of these words were seen in the training data", text)
        proba = clf.predict_proba(x)[0]
    else:
        from common import embed_texts   # needs torch + transformers
        proba = clf.predict_proba(embed_texts([cleaned], bundle["embedding_model"],
                                              normalize=bundle.get("normalized", True)))[0]

    k = int(np.argmax(proba))
    conf = round(float(proba[k]), 3)
    probs = {c: round(float(p), 3) for c, p in zip(classes, proba)}

    words = []
    if is_tfidf and top_words:
        names = np.array(bundle["vectorizer"].get_feature_names_out())
        contrib = x.toarray()[0] * class_coef(clf, k)
        for i in np.argsort(contrib)[::-1][:top_words]:
            if contrib[i] > 0:
                words.append({"word": str(names[i]), "weight": round(float(contrib[i]), 3)})

    if conf < min_confidence:
        return _unclear("confidence %.2f is below the cutoff of %.2f" % (conf, min_confidence),
                        text, closest=classes[k], confidence=conf, probs=probs, words=words)
    return {"label": classes[k], "closest_label": classes[k], "confidence": conf,
            "probabilities": probs, "top_words": words, "reason": "", "input": text, "status": STATUS}


def format_result(r):
    if r["label"] == "unclear":
        head = "Result: unclear (%s)" % r["reason"]
        if r["closest_label"]:
            head += "\n  Closest label: %s" % r["closest_label"]
    else:
        head = "Result: %s (confidence %.2f)" % (r["label"], r["confidence"])
    lines = [head]
    if r["top_words"]:
        lines.append("  Words behind it: " + ", ".join("%s (%+.2f)" % (w["word"], w["weight"]) for w in r["top_words"]))
    lines.append("  This is a model estimate from patterns in training data, not a verdict.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default=None)
    ap.add_argument("--input", default=None, help="CSV with many texts")
    ap.add_argument("--text-col", default="caption")
    ap.add_argument("--out", default=None, help="save results for --input here")
    ap.add_argument("--min-confidence", type=float, default=0.65,
                    help="below this confidence the answer is 'unclear' (tune it with evaluate.py)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if bool(a.text) == bool(a.input):
        raise SystemExit("Give exactly one of --text or --input.")

    if a.text:
        r = analyze_text(a.text, a.model, a.min_confidence)
        print(json.dumps(r, indent=2) if a.json else format_result(r))
        return

    src = pd.read_csv(a.input)
    rows = []
    for t in src[a.text_col].fillna("").astype(str):
        r = analyze_text(t, a.model, a.min_confidence)
        row = {"text": t, "label": r["label"], "closest_label": r["closest_label"],
               "confidence": r["confidence"], "reason": r["reason"]}
        row.update({"p_" + c: p for c, p in r["probabilities"].items()})
        rows.append(row)
    out = pd.DataFrame(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print("Saved %d results to %s" % (len(out), a.out))
    else:
        print(out.head(20).to_string(index=False))
    print("Answered: %d of %d (the rest were 'unclear')" % ((out["label"] != "unclear").sum(), len(out)))


if __name__ == "__main__":
    main()
