#!/usr/bin/env python3
"""PyTorch DataLoaders for TEXT embeddings, aligned to posts by post_id (not row position).

Use in your own code:
    from text_data_loader import make_loaders
    loaders, label_map, datasets = make_loaders("clean/clean_text.csv", "embeddings/", batch_size=32)
    for batch in loaders["train"]:
        batch["text"]     # [B, D] text embeddings
        batch["label"]    # [B] integer labels (only if clean_text.csv has a label column)
        batch["post_id"]  # list of B post ids

Self-test + save:
    python3 text_data_loader.py --clean clean/clean_text.csv --emb-dir embeddings/ --save-dir loader_out/
Saves per split: aligned_<split>.pt, plus sample_batch.pt and label_map.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def align_text_embeddings(meta_csv, text_npy, ids_csv):
    """Join clean_text.csv to the embedding array by post_id. Returns (meta, T, label_map)."""
    meta = pd.read_csv(meta_csv, dtype={"post_id": str})
    ids = pd.read_csv(ids_csv, dtype={"post_id": str})["post_id"]
    T = np.load(text_npy)
    if len(ids) != len(T):
        raise ValueError("Size mismatch: %d ids vs %d embedding rows" % (len(ids), len(T)))

    index = pd.DataFrame({"post_id": ids, "row": np.arange(len(ids))})
    merged = meta.merge(index, on="post_id", how="inner")
    if len(merged) < len(meta):
        print("Note: %d posts in the clean file have no embedding and were left out." % (len(meta) - len(merged)))

    label_map = None
    if "label" in merged.columns:
        before = len(merged)
        merged = merged[merged["label"].notna()].copy()
        if len(merged) < before:
            print("Note: dropped %d posts with no label." % (before - len(merged)))
        classes = sorted(merged["label"].astype(str).unique())
        label_map = {c: k for k, c in enumerate(classes)}
        merged["label_id"] = merged["label"].astype(str).map(label_map)
    return merged.reset_index(drop=True), T, label_map


class TextEmbDataset(Dataset):
    def __init__(self, meta, T):
        self.post_ids = meta["post_id"].tolist()
        self.rows = meta["row"].to_numpy()
        self.T = torch.from_numpy(T)
        self.labels = (torch.tensor(meta["label_id"].to_numpy(), dtype=torch.long)
                       if "label_id" in meta.columns else None)

    def __len__(self):
        return len(self.post_ids)

    def __getitem__(self, k):
        item = {"text": self.T[int(self.rows[k])], "post_id": self.post_ids[k]}
        if self.labels is not None:
            item["label"] = self.labels[k]
        return item

    def as_tensors(self):
        r = torch.as_tensor(self.rows, dtype=torch.long)
        out = {"text": self.T[r], "post_ids": self.post_ids}
        if self.labels is not None:
            out["label"] = self.labels
        return out


def make_loaders(clean_csv, emb_dir, batch_size=32, num_workers=0, seed=42):
    emb = Path(emb_dir)
    meta, T, label_map = align_text_embeddings(clean_csv, emb / "text_embs.npy", emb / "embedding_ids.csv")
    if "split" not in meta.columns:
        meta["split"] = "all"
    datasets, loaders = {}, {}
    for split in meta["split"].unique():
        ds = TextEmbDataset(meta[meta["split"] == split], T)
        g = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, generator=g)
        datasets[split] = ds
    return loaders, label_map, datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="clean_text.csv")
    ap.add_argument("--emb-dir", required=True, help="folder from extract_text_embeddings.py")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()

    loaders, label_map, datasets = make_loaders(a.clean, a.emb_dir, a.batch_size)
    emb = Path(a.emb_dir)
    T = np.load(emb / "text_embs.npy")
    id_to_row = {p: k for k, p in enumerate(pd.read_csv(emb / "embedding_ids.csv", dtype=str)["post_id"])}

    print("Split sizes:", {s: len(d) for s, d in datasets.items()})
    if label_map:
        print("Label map:", label_map)

    for split, ds in datasets.items():
        for k in range(min(20, len(ds))):
            item = ds[k]
            assert np.allclose(item["text"].numpy(), T[id_to_row[item["post_id"]]]), \
                "MISALIGNED: %s item %d (%s)" % (split, k, item["post_id"])
    print("Alignment check passed: each embedding matches its own post_id.")

    for split, loader in loaders.items():
        b = next(iter(loader))
        print("First %s batch:" % split, {k: tuple(v.shape) for k, v in b.items() if hasattr(v, "shape")})

    if a.save_dir:
        out = Path(a.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for split, ds in datasets.items():
            torch.save(ds.as_tensors(), out / ("aligned_%s.pt" % split))
        first = "train" if "train" in loaders else next(iter(loaders))
        torch.save(next(iter(loaders[first])), out / "sample_batch.pt")
        if label_map:
            (out / "label_map.json").write_text(json.dumps(label_map, indent=2))
        print("Saved to %s/" % out)


if __name__ == "__main__":
    main()
