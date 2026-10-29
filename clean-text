#!/usr/bin/env python3
"""Text-only cleaning of Instagram captions: normalize text, drop bad rows and duplicates,
and make a group-aware train/val/test split (no account appears in more than one split).

Example:
  python3 clean_text.py --input raw.csv --id-col shortcode --text-col caption \
      --label-col label --group-col username --out-dir clean/

Outputs in --out-dir:
  clean_text.csv   post_id, caption, n_chars, split, [label], [group]
  drop_log.csv     every removed post and the reason
  report.txt       counts, split sizes, label balance, caption length stats
"""
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def load(path):
    s = path.suffix.lower()
    if s == ".csv":
        return pd.read_csv(path)
    if s == ".tsv":
        return pd.read_csv(path, sep="\t")
    if s == ".parquet":
        return pd.read_parquet(path)
    if s == ".jsonl":
        return pd.read_json(path, lines=True)
    if s == ".json":
        return pd.read_json(path)
    raise ValueError("Unsupported input format: " + s)


def clean_text(t, strip_urls=True):
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFC", t)
    if strip_urls:
        t = URL_RE.sub(" ", t)
    # Hashtags, @mentions and emojis are deliberately KEPT (possible stance signal).
    return WS_RE.sub(" ", t).strip()


def group_split(df, group_col, fracs, seed):
    rng = np.random.default_rng(seed)
    if group_col and group_col in df:
        keys = df[group_col].fillna("__none__").astype(str)
        groups = np.array(keys.unique(), dtype=object)
        rng.shuffle(groups)
        n = len(groups)
        a = int(round(fracs[0] * n))
        b = a + int(round(fracs[1] * n))
        m = {g: "train" for g in groups[:a]}
        m.update({g: "val" for g in groups[a:b]})
        m.update({g: "test" for g in groups[b:]})
        return keys.map(m)
    r = rng.random(len(df))
    return pd.Series(np.where(r < fracs[0], "train",
                     np.where(r < fracs[0] + fracs[1], "val", "test")), index=df.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--group-col", default=None, help="e.g. username; keeps accounts in one split")
    ap.add_argument("--out-dir", default="clean")
    ap.add_argument("--min-chars", type=int, default=3, help="min caption length after cleaning")
    ap.add_argument("--keep-urls", action="store_true")
    ap.add_argument("--keep-duplicate-captions", action="store_true",
                    help="by default, posts with an identical caption (ignoring case) are removed")
    ap.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test fractions")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fracs = [float(x) for x in a.split.split(",")]
    assert len(fracs) == 3 and abs(sum(fracs) - 1) < 1e-6, "--split must be three numbers summing to 1"

    raw = load(Path(a.input))
    n0 = len(raw)
    cols = {a.id_col: "post_id", a.text_col: "caption_raw"}
    if a.label_col:
        cols[a.label_col] = "label"
    if a.group_col:
        cols[a.group_col] = "group"
    df = raw.rename(columns=cols)[list(cols.values())].copy()
    df["post_id"] = df["post_id"].astype(str)

    drops = []

    def drop(mask, reason):
        nonlocal df
        for pid in df.loc[mask, "post_id"]:
            drops.append({"post_id": pid, "reason": reason})
        df = df[~mask].copy()

    drop(df["post_id"].isin(["nan", "None", ""]), "missing_id")
    drop(df.duplicated("post_id", keep="first"), "duplicate_post_id")

    df["caption"] = df["caption_raw"].map(lambda t: clean_text(t, not a.keep_urls))
    drop(df["caption"].str.len() < a.min_chars, "caption_empty_or_too_short")

    if not a.keep_duplicate_captions:
        drop(df["caption"].str.lower().duplicated(keep="first"), "duplicate_caption")

    df["n_chars"] = df["caption"].str.len()
    df["split"] = group_split(df, "group" if a.group_col else None, fracs, a.seed)

    keep = ["post_id", "caption", "n_chars", "split"]
    keep += [c for c in ("label", "group") if c in df]
    df[keep].reset_index(drop=True).to_csv(out / "clean_text.csv", index=False)
    pd.DataFrame(drops, columns=["post_id", "reason"]).to_csv(out / "drop_log.csv", index=False)

    lines = ["Input rows: %d" % n0, "Kept: %d" % len(df), "Dropped: %d" % len(drops)]
    if drops:
        lines += ["", "Drops by reason:"] + ["  %s: %d" % (k, v) for k, v in
                                              pd.DataFrame(drops)["reason"].value_counts().items()]
    lines += ["", "Split sizes:"] + ["  %s: %d" % (k, v) for k, v in df["split"].value_counts().items()]
    if "label" in df:
        lines += ["", "Label distribution by split:", df.groupby("split")["label"].value_counts().to_string()]
    if len(df):
        lines += ["", "Caption length (chars): median=%d, p95=%d" %
                  (df["n_chars"].median(), df["n_chars"].quantile(0.95)),
                  "Captions with a hashtag: %.0f%%" % (100 * df["caption"].str.contains("#").mean()),
                  "Captions with an @mention: %.0f%%" % (100 * df["caption"].str.contains("@").mean())]
    (out / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
