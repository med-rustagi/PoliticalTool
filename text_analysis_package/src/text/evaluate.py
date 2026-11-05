#!/usr/bin/env python3
"""Evaluate one or more prediction files (from baseline_tfidf.py / train_classifier.py).

Usage:
  python3 evaluate.py --pred results/predictions_tfidf.csv
  python3 evaluate.py --pred results/predictions_tfidf.csv results/predictions_embedding_logreg.csv

Reports, for the chosen split (default: test):
  accuracy and macro-F1 with 95% confidence intervals, per-class precision/recall/F1,
  a confusion matrix, the score of always guessing the most common label, and
  (if the data has an account column) accuracy per account.

Confidence intervals resample whole ACCOUNTS when accounts are present, because posts
from one account are not independent. With only a few accounts the intervals are wide.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
from common import compute_metrics


def cluster_bootstrap(df, classes, n_boot, seed):
    rng = np.random.default_rng(seed)
    yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    if "group" in df.columns:
        keys = df["group"].fillna("__none__").astype(str).to_numpy()
        idx = {g: np.where(keys == g)[0] for g in np.unique(keys)}
        names = list(idx)
        draw = lambda: np.concatenate([idx[g] for g in rng.choice(names, size=len(names), replace=True)])
    else:
        draw = lambda: rng.integers(0, len(df), len(df))
    accs, f1s = [], []
    for _ in range(n_boot):
        i = draw()
        accs.append(float((yt[i] == yp[i]).mean()))
        f1s.append(float(f1_score(yt[i], yp[i], labels=classes, average="macro", zero_division=0)))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


def evaluate_file(path, split, n_boot, results_dir):
    full = pd.read_csv(path, dtype={"post_id": str})
    classes = sorted(set(full["y_true"].astype(str)))
    df = full[full["split"] == split].copy()
    if df.empty:
        raise SystemExit("No rows in split '%s' in %s" % (split, path))
    df["y_true"], df["y_pred"] = df["y_true"].astype(str), df["y_pred"].astype(str)

    m = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), classes)
    acc_ci, f1_ci = cluster_bootstrap(df, classes, n_boot, config.SEED)
    train = full[full["split"] == "train"]["y_true"]
    maj = train.value_counts().idxmax() if len(train) else df["y_true"].value_counts().idxmax()
    maj_acc = float((df["y_true"] == maj).mean())

    L = ["Evaluation: %s (split: %s)" % (Path(path).name, split), "",
         "Posts: %d" % m["n"],
         "Accuracy: %.3f  (95%% CI %.3f-%.3f)" % (m["accuracy"], acc_ci[0], acc_ci[1]),
         "Macro-F1: %.3f  (95%% CI %.3f-%.3f)" % (m["macro_f1"], f1_ci[0], f1_ci[1]),
         "Always guessing '%s': accuracy %.3f" % (maj, maj_acc), "",
         "Per class:"]
    for c, v in m["per_class"].items():
        L.append("  %-12s precision=%.3f recall=%.3f f1=%.3f support=%d" %
                 (c, v["precision"], v["recall"], v["f1"], v["support"]))
    L += ["", "Confusion matrix (rows = true, columns = predicted; order: %s):" % ", ".join(classes)]
    L += ["  " + "  ".join("%4d" % x for x in row) for row in m["confusion_matrix"]]
    if "confidence" in df.columns:
        L += ["", "If the tool says 'unclear' below a confidence cutoff (accuracy on the posts it still answers):"]
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            keep = df["confidence"] >= t
            if keep.sum() == 0:
                L.append("  cutoff %.1f: answers 0%% of posts" % t)
            else:
                L.append("  cutoff %.1f: answers %3.0f%% of posts, accuracy %.3f" %
                         (t, 100 * keep.mean(), float((df.loc[keep, "y_true"] == df.loc[keep, "y_pred"]).mean())))
    if "group" in df.columns:
        g = (df.assign(ok=df["y_true"] == df["y_pred"]).groupby("group")["ok"].agg(["mean", "size"])
             .sort_values("mean"))
        L += ["", "Accuracy by account (lowest first, up to 10):"]
        L += ["  %-25s %.3f (n=%d)" % (k, r["mean"], r["size"]) for k, r in g.head(10).iterrows()]
        L += ["", "Caveat: if labels come from the ACCOUNT (e.g. party), a model can score well by",
              "recognizing account style instead of political content. Check this with hand-labeled posts."]
        L += ["Note: only %d accounts in this split, so the intervals above are rough." % len(g)] if len(g) < 10 else []
    text = "\n".join(L)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    stem = "%s_%s" % (Path(path).stem.replace("predictions_", "evaluation_"), split)
    (Path(results_dir) / (stem + ".txt")).write_text(text)
    m.update({"accuracy_ci95": acc_ci.tolist(), "macro_f1_ci95": f1_ci.tolist(),
              "majority_baseline_accuracy": maj_acc})
    (Path(results_dir) / (stem + ".json")).write_text(json.dumps(m, indent=2))
    print(text + "\n")
    return Path(path).stem.replace("predictions_", ""), m, f1_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="one or more predictions_*.csv files")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--results-dir", default=config.RESULTS_DIR)
    a = ap.parse_args()

    summary = [evaluate_file(p, a.split, a.bootstrap, a.results_dir) for p in a.pred]
    if len(summary) > 1:
        print("Comparison on '%s':" % a.split)
        for name, m, ci in summary:
            print("  %-28s accuracy=%.3f  macro-F1=%.3f (CI %.3f-%.3f)" %
                  (name, m["accuracy"], m["macro_f1"], ci[0], ci[1]))
        print("If the confidence intervals overlap a lot, the difference may not be real.")


if __name__ == "__main__":
    main()
