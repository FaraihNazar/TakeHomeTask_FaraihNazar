#!/usr/bin/env python
"""Slow, independent reference: plain Ultralytics API on EVERY frame (like the graders).

Reads each video with OpenCV (which applies rotation tags), runs YOLO(...).predict
with conf=0.4 / imgsz=640 / default everything else, then applies the section-5 rules.
Use it on a handful of videos to validate run.py:

    python tools/reference.py --input finevideo/data --output ref.jsonl --max-videos 20
    python run.py --input finevideo/data --output ours.jsonl --max-videos 20
    python tools/compare.py ref.jsonl ours.jsonl
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import TARGET_IDS, TARGET_NAMES, aggregate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--weights", default="yolo11l.pt")
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    import cv2
    import pyarrow.parquet as pq
    from ultralytics import YOLO

    model = YOLO(args.weights)
    n = 0
    with open(args.output, "w") as out:
        for path in sorted(glob.glob(os.path.join(args.input, "*.parquet"))):
            fname = os.path.basename(path)
            pf = pq.ParquetFile(path)
            row = 0
            for g in range(pf.metadata.num_row_groups):
                col = pf.read_row_group(g, columns=["mp4"]).column("mp4")
                for i in range(len(col)):
                    if args.max_videos and n >= args.max_videos:
                        return
                    vid = f"{fname}:{row}"
                    row += 1
                    n += 1
                    data = col[i].as_py()
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                        tf.write(data)
                        tmp = tf.name
                    cap = cv2.VideoCapture(tmp)
                    secs, counts, t0 = [], [], None
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        t0 = ts if t0 is None else t0
                        r = model.predict(frame, imgsz=640, conf=0.4, half=args.half, verbose=False)[0]
                        cls = r.boxes.cls.cpu().numpy().astype(int)
                        counts.append([int((cls == c).sum()) for c in TARGET_IDS])
                        secs.append(int(np.floor(ts - t0 + 1e-9)))
                    cap.release()
                    os.remove(tmp)
                    if not secs:
                        out.write(json.dumps({"video": vid, "error": "could not decode video"}) + "\n")
                        continue
                    res = aggregate(np.array(secs), np.array(counts), max(secs) + 1)
                    for s, (objs, ppl) in enumerate(res):
                        out.write(json.dumps({"video": vid, "second": s, "objects": objs, "people": ppl}) + "\n")
                    print(f"{vid}: {len(secs)} frames, {max(secs) + 1} s", file=sys.stderr)


if __name__ == "__main__":
    main()
