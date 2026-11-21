#!/usr/bin/env python
"""Sanity-check a results file against the parquet files it came from.

    python tools/check_results.py --input finevideo/data --results results.jsonl

Checks: every video (file:row) has lines, seconds run 0..N with no gaps or duplicates,
no unknown videos, and reports error lines and total hours.
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    expected = set()
    for path in sorted(glob.glob(os.path.join(args.input, "*.parquet"))):
        name = os.path.basename(path)
        try:
            n = pq.ParquetFile(path).metadata.num_rows
        except Exception as e:
            print(f"unreadable parquet {name}: {e}")
            expected.add(f"{name}:0")
            continue
        expected.update(f"{name}:{r}" for r in range(n))

    secs = defaultdict(list)
    errors = {}
    opener = gzip.open if args.results.endswith(".gz") else open
    bad = 0
    with opener(args.results, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                bad += 1
                continue
            if "error" in d:
                errors[d["video"]] = d["error"]
            else:
                secs[d["video"]].append(d["second"])

    seen = set(secs) | set(errors)
    missing = sorted(expected - seen)
    unknown = sorted(seen - expected)
    broken = []
    for v, s in secs.items():
        s.sort()
        if s != list(range(len(s))) and v not in errors:
            broken.append(v)
    total = sum(len(s) for s in secs.values())
    print(f"expected videos: {len(expected)}  in results: {len(seen)}")
    print(f"missing: {len(missing)}  unknown: {len(unknown)}  gaps/duplicates: {len(broken)}  bad lines: {bad}")
    print(f"error lines: {len(errors)}  seconds: {total}  ({total / 3600:.1f} h)")
    for v in missing[:10]:
        print("  missing", v)
    for v in broken[:10]:
        print("  gaps/dupes", v)
    for v, e in list(errors.items())[:10]:
        print("  error", v, e)
    print("OK" if not (missing or unknown or broken or bad) else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
