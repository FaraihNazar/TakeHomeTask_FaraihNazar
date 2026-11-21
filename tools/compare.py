#!/usr/bin/env python
"""Score a results file against a reference using the three checks from section 5.

    python tools/compare.py ref.jsonl ours.jsonl

Only videos present in the reference are scored. Missing seconds count as wrong.
"""
import json
import sys
from collections import defaultdict


def load(path):
    secs, errors = {}, set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "error" in d:
                errors.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (set(d["objects"]), d["people"])
    return secs, errors


def main(ref_path, ours_path):
    ref, ref_err = load(ref_path)
    ours, ours_err = load(ours_path)
    videos = {v for v, _ in ref}
    tp = fp = fn = exact = within1 = 0
    per_class = defaultdict(lambda: [0, 0, 0])
    worst = defaultdict(int)
    for key, (robj, rppl) in ref.items():
        oobj, oppl = ours.get(key, (set(), None))
        tp += len(robj & oobj)
        fp += len(oobj - robj)
        fn += len(robj - oobj)
        for c in robj & oobj:
            per_class[c][0] += 1
        for c in oobj - robj:
            per_class[c][1] += 1
        for c in robj - oobj:
            per_class[c][2] += 1
        if oppl is not None and oppl == rppl:
            exact += 1
        else:
            worst[key[0]] += 1
        if oppl is not None and abs(oppl - rppl) <= 1:
            within1 += 1
    extra = [k for k in ours if k[0] in videos and k not in ref]
    n = len(ref)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    print(f"videos: {len(videos)}  seconds: {n}  extra seconds in ours: {len(extra)}")
    print(f"errors  ref: {len(ref_err)}  ours: {len(ours_err)}")
    print(f"objects F1:        {f1:.4f}   (need >= 0.95)   tp={tp} fp={fp} fn={fn}")
    print(f"people exact:      {exact / max(1, n):.4f}   (need >= 0.85)")
    print(f"people within 1:   {within1 / max(1, n):.4f}   (need >= 0.95)")
    for c, (a, b, d) in sorted(per_class.items()):
        print(f"  {c:11s} F1 {2 * a / max(1, 2 * a + b + d):.4f}  tp={a} fp={b} fn={d}")
    ok = f1 >= 0.95 and exact >= 0.85 * n and within1 >= 0.95 * n
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
