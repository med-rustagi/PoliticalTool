#!/usr/bin/env python3
"""Smoke test for the text pipeline. Uses FAKE demo data in a temporary folder, so it never
touches your real data. Needs only numpy, pandas, scikit-learn (no torch, no internet).

Run from the repository root:   python3 tests/test_text_pipeline.py      (or: pytest tests/)
"""
import json
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TEXT = Path(__file__).resolve().parents[1] / "src" / "text"
sys.path.insert(0, str(TEXT))


def run(script, *args, cwd):
    r = subprocess.run([sys.executable, str(TEXT / script), *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, "%s failed:\n%s\n%s" % (script, r.stdout[-800:], r.stderr[-800:])
    return r.stdout


@lru_cache(maxsize=1)
def pipeline():
    work = Path(tempfile.mkdtemp(prefix="text_pipeline_test_"))
    run("make_demo_data.py", "--out", "raw.csv", cwd=work)
    run("clean_text.py", "--input", "raw.csv", "--id-col", "shortcode", "--text-col", "caption",
        "--label-col", "stance", "--group-col", "username", "--out-dir", "clean", cwd=work)
    run("baseline_tfidf.py", cwd=work)
    return work


def test_cleaning_outputs_and_no_account_in_two_splits():
    work = pipeline()
    df = pd.read_csv(work / "clean" / "clean_text.csv")
    assert {"post_id", "caption", "split", "label", "group"} <= set(df.columns)
    assert (df.groupby("group")["split"].nunique() == 1).all(), "an account appears in more than one split"
    assert df["post_id"].is_unique


def test_baseline_saves_model_and_predictions():
    work = pipeline()
    assert (work / "models" / "tfidf_baseline.joblib").exists()
    p = pd.read_csv(work / "results" / "predictions_tfidf.csv")
    assert {"post_id", "split", "y_true", "y_pred", "confidence"} <= set(p.columns)


def test_evaluate_reports_metrics_and_cutoff_table():
    out = run("evaluate.py", "--pred", "results/predictions_tfidf.csv", "--bootstrap", "50", cwd=pipeline())
    assert "Accuracy:" in out and "Macro-F1:" in out and "cutoff" in out


def test_explain_runs():
    out = run("explain.py", "--model", "models/tfidf_baseline.joblib", "--top", "3", cwd=pipeline())
    assert "Words most associated with" in out


def test_tool_answers_and_returns_expected_fields():
    out = run("text_tool.py", "--text", "healthcare climate union equality workers", "--json",
              "--min-confidence", "0.5", cwd=pipeline())
    r = json.loads(out)
    assert r["label"] in {"left", "right", "unclear"}
    assert {"confidence", "probabilities", "top_words", "reason", "status"} <= set(r)


def test_tool_says_unclear_for_bad_input():
    work = pipeline()
    for text in ("", "a", "zzzqqq xxyyzz"):
        r = json.loads(run("text_tool.py", "--text", text or " ", "--json", cwd=work))
        assert r["label"] == "unclear", (text, r)


def test_tool_is_deterministic():
    work = pipeline()
    a = run("text_tool.py", "--text", "border taxes freedom jobs", "--json", cwd=work)
    b = run("text_tool.py", "--text", "border taxes freedom jobs", "--json", cwd=work)
    assert a == b


def test_embedding_alignment_by_post_id():
    """Shuffled embeddings with a few posts missing must still line up with the right posts."""
    from common import attach_embeddings, load_clean
    work = pipeline()
    df = load_clean(work / "clean" / "clean_text.csv")
    rng = np.random.default_rng(1)
    ids = df["post_id"].tolist()
    rng.shuffle(ids)
    ids = ids[:-5]
    emb = work / "embeddings"
    emb.mkdir(exist_ok=True)
    np.save(emb / "text_embs.npy", np.array([[float(i.split("_")[-1]), float(len(i))] for i in ids], dtype=np.float32))
    pd.Series(ids, name="post_id").to_csv(emb / "embedding_ids.csv", index=False)
    rows, X = attach_embeddings(df, emb)
    assert len(rows) == len(df) - 5
    for pid, x in zip(rows["post_id"], X):
        assert x[0] == float(pid.split("_")[-1]) and x[1] == float(len(pid))


def test_embedding_classifier_trains():
    work = pipeline()
    # reuse the shuffled fake embeddings from the alignment test, but give them label signal
    df = pd.read_csv(work / "clean" / "clean_text.csv", dtype={"post_id": str})
    rng = np.random.default_rng(2)
    ids = df["post_id"].tolist()
    rng.shuffle(ids)
    lab = df.set_index("post_id")["label"]
    E = np.array([rng.normal(0, 1, 16) + (1.0 if lab[i] == "right" else -1.0) * np.r_[np.ones(6), np.zeros(10)]
                  for i in ids], dtype=np.float32)
    emb = work / "embeddings"
    emb.mkdir(exist_ok=True)
    np.save(emb / "text_embs.npy", E)
    pd.Series(ids, name="post_id").to_csv(emb / "embedding_ids.csv", index=False)
    for model_type in ("logreg", "mlp"):
        run("train_classifier.py", "--model-type", model_type, cwd=work)
        assert (work / "models" / ("embedding_%s.joblib" % model_type)).exists()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as e:
            failed += 1
            print("FAIL  %s\n      %s" % (name, str(e)[:600]))
    print("\n%d of %d tests passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
