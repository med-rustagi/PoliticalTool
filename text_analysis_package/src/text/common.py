"""Shared helpers. numpy / pandas / scikit-learn only (torch is imported lazily, and
only by embed_texts, which predict.py uses for embedding-based models)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def load_clean(clean_csv):
    """Load clean_text.csv and keep only labeled rows."""
    df = pd.read_csv(clean_csv, dtype={"post_id": str})
    for col in ("label", "split"):
        if col not in df.columns:
            raise SystemExit("clean_text.csv has no '%s' column. Re-run clean_text.py with "
                             "--label-col (labels are needed to train a bias detector)." % col)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(str)
    return df.reset_index(drop=True)


def attach_embeddings(df, emb_dir):
    """Join labeled rows to saved embeddings BY post_id. Returns (rows, X) with matching order."""
    emb = Path(emb_dir)
    ids = pd.read_csv(emb / "embedding_ids.csv", dtype={"post_id": str})["post_id"]
    T = np.load(emb / "text_embs.npy")
    if len(ids) != len(T):
        raise ValueError("Size mismatch: %d ids vs %d embedding rows" % (len(ids), len(T)))
    index = pd.DataFrame({"post_id": ids, "row": np.arange(len(ids))})
    merged = df.merge(index, on="post_id", how="inner")
    if len(merged) < len(df):
        print("Note: %d posts have no embedding and were left out." % (len(df) - len(merged)))
    return merged.reset_index(drop=True), T[merged["row"].to_numpy()]


def compute_metrics(y_true, y_pred, classes):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "classes": list(classes),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


def save_predictions(rows, y_pred, proba, classes, path):
    """Write one row per post: post_id, split, [group], y_true, y_pred, confidence, p_<class>."""
    out = pd.DataFrame({"post_id": rows["post_id"], "split": rows["split"]})
    if "group" in rows.columns:
        out["group"] = rows["group"]
    out["y_true"] = rows["label"].values
    out["y_pred"] = y_pred
    out["confidence"] = proba.max(axis=1)
    for k, c in enumerate(classes):
        out["p_" + c] = proba[:, k]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def print_split_metrics(rows, y_pred, classes):
    for split in ("train", "val", "test"):
        m = (rows["split"] == split).to_numpy()
        if m.sum() == 0:
            continue
        r = compute_metrics(rows["label"].values[m], np.asarray(y_pred)[m], classes)
        print("  %-5s n=%-5d accuracy=%.3f  macro-F1=%.3f" % (split, r["n"], r["accuracy"], r["macro_f1"]))


def embed_texts(texts, model_name, max_length=None, batch_size=32, normalize=True):
    """Embed NEW texts the same way extract_text_embeddings.py does. Needs torch + transformers."""
    import torch
    from transformers import AutoModel, AutoTokenizer, CLIPModel

    is_clip = "clip" in model_name.lower()
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = (CLIPModel if is_clip else AutoModel).from_pretrained(model_name).to(device).eval()
    max_len = max_length or (77 if is_clip else min(512, tok.model_max_length))
    chunks = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(list(texts[s:s + batch_size]), return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            if is_clip:
                feats = model.get_text_features(input_ids=enc["input_ids"],
                                                attention_mask=enc["attention_mask"])
            else:
                hidden = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                feats = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.float().cpu().numpy())
    return np.concatenate(chunks)


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
