#!/usr/bin/env python3
"""Train a political-bias classifier on saved TEXT EMBEDDINGS (from extract_text_embeddings.py).

Usage:
  python3 train_classifier.py --clean clean/clean_text.csv --emb-dir embeddings/
  python3 train_classifier.py --model-type mlp        # small neural network instead of logistic regression

For logistic regression, the regularization strength C is chosen on the validation split.

Outputs:
  models/embedding_<type>.joblib            model (used by predict.py)
  results/predictions_embedding_<type>.csv  one row per post, all splits (used by evaluate.py)
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

import config
from common import (attach_embeddings, load_clean, print_split_metrics, read_json, save_predictions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=config.CLEAN_CSV)
    ap.add_argument("--emb-dir", default=config.EMB_DIR)
    ap.add_argument("--model-type", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--model-dir", default=config.MODEL_DIR)
    ap.add_argument("--results-dir", default=config.RESULTS_DIR)
    a = ap.parse_args()

    rows, X = attach_embeddings(load_clean(a.clean), a.emb_dir)
    tr = (rows["split"] == "train").to_numpy()
    va = (rows["split"] == "val").to_numpy()
    y_text = rows["label"].to_numpy()
    all_classes = sorted(set(y_text))
    y = np.array([all_classes.index(v) for v in y_text])   # integer labels (some scikit-learn versions fail on text labels)
    if len(set(y[tr])) < 2:
        raise SystemExit("The training split needs at least 2 different labels.")

    if a.model_type == "logreg":
        best, best_f1 = None, -1.0
        can_tune = va.sum() > 0 and len(set(y[va])) > 1
        for C in ([0.01, 0.1, 1, 10, 100] if can_tune else [1.0]):
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=config.SEED)
            clf.fit(X[tr], y[tr])
            f1 = f1_score(y[va], clf.predict(X[va]), average="macro") if can_tune else 0.0
            if f1 > best_f1:
                best, best_f1 = (clf, C), f1
        clf, chosen = best
        print("Logistic regression on embeddings (C=%s%s)" %
              (chosen, ", picked on validation macro-F1=%.3f" % best_f1 if can_tune else ", no usable validation split"))
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=300,
                            random_state=config.SEED)
        clf.fit(X[tr], y[tr])
        print("Small neural network (MLP) on embeddings")

    names = [all_classes[i] for i in clf.classes_]          # class names, in predict_proba column order
    proba = clf.predict_proba(X)
    pred = np.array(names)[proba.argmax(axis=1)]
    print_split_metrics(rows, pred, names)

    meta = read_json(Path(a.emb_dir) / "embedding_meta.json")
    Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "classes": names,
                 "embedding_model": meta.get("model", "openai/clip-vit-base-patch32"),
                 "normalized": meta.get("normalized", True)},
                Path(a.model_dir) / ("embedding_%s.joblib" % a.model_type))
    save_predictions(rows, pred, proba, names,
                     Path(a.results_dir) / ("predictions_embedding_%s.csv" % a.model_type))
    print("Saved model to %s/ and predictions to %s/" % (a.model_dir, a.results_dir))


if __name__ == "__main__":
    main()
