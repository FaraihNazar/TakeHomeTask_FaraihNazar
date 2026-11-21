#!/usr/bin/env python
"""Pick random videos spread over all parquet files (for building a reference subset).

    python tools/pick_videos.py --input finevideo/data --per-file 1 --seed 0 > pick.txt
"""
import argparse
import glob
import os
import random

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--per-file", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)
for path in sorted(glob.glob(os.path.join(args.input, "*.parquet"))):
    try:
        n = pq.ParquetFile(path).metadata.num_rows
    except Exception:
        continue  # unreadable file
    for r in sorted(rng.sample(range(n), min(args.per_file, n))):
        print(f"{os.path.basename(path)}:{r}")
