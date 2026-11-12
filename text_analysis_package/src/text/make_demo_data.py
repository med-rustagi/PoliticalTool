#!/usr/bin/env python3
"""Make FAKE demo posts so you can test the whole pipeline before you have real data.

  python3 src/text/make_demo_data.py --out demo_data/demo_posts_FAKE.csv

The posts are random word mixes, NOT real Instagram posts, and the results mean nothing about
the real world. Every fake account also uses its own hashtag, which is deliberate: it lets you
see (with explain.py) how a model can learn account style instead of political content.
Columns: shortcode, caption, stance, username
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LEFT = "healthcare climate union equality workers rights justice community voting schools".split()
RIGHT = "border taxes freedom family security energy jobs military liberty values".split()
NEUTRAL = "today great meeting thank team event photo live proud tonight visit".split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data/demo_posts_FAKE.csv")
    ap.add_argument("--accounts", type=int, default=24)
    ap.add_argument("--posts", type=int, default=25, help="posts per account")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rows = []
    for acct in range(a.accounts):
        side = "left" if acct < a.accounts // 2 else "right"
        for p in range(a.posts):
            own, other = (LEFT, RIGHT) if side == "left" else (RIGHT, LEFT)
            pool = own if rng.random() > 0.2 else other     # 20% of posts use the other side's words
            words = list(rng.choice(pool, 3)) + list(rng.choice(NEUTRAL, 3)) + ["#acct%d" % acct, str(rng.integers(10**6))]
            rng.shuffle(words)
            rows.append(("demo%d_%d" % (acct, p), " ".join(words), side, "demo_user%d" % acct))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["shortcode", "caption", "stance", "username"]).to_csv(a.out, index=False)
    print("Wrote %d FAKE posts to %s (demo only, not real data)" % (len(rows), a.out))


if __name__ == "__main__":
    main()
