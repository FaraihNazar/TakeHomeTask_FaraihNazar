#!/usr/bin/env python
"""One-time preparation (untimed, has internet):
  1. download the official yolo11l.pt weights next to run.py
  2. if a CUDA GPU and TensorRT are available, build a TensorRT FP16 engine
     (yolo11l.engine, dynamic shapes, max batch 32) for this exact GPU.

run.py uses yolo11l.engine when it exists and falls back to yolo11l.pt otherwise.
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32, help="max batch size of the TensorRT engine")
    ap.add_argument("--no-engine", action="store_true", help="only download yolo11l.pt")
    args = ap.parse_args()

    os.chdir(HERE)
    from ultralytics import YOLO

    pt = os.path.join(HERE, "yolo11l.pt")
    YOLO("yolo11l.pt")  # downloads into the current directory if missing
    if not os.path.exists(pt):
        sys.exit("failed to download yolo11l.pt")
    print(f"weights: {pt}")

    if args.no_engine:
        return
    import torch

    if not torch.cuda.is_available():
        print("no CUDA GPU: skipping TensorRT export, run.py will use yolo11l.pt")
        return
    try:
        out = YOLO(pt).export(format="engine", imgsz=640, half=True, dynamic=True, batch=args.batch,
                              simplify=True, device=0, workspace=None)
        eng = os.path.join(HERE, "yolo11l.engine")
        if out and os.path.abspath(out) != eng:
            shutil.move(out, eng)
        print(f"engine: {eng}")
    except Exception as e:  # PyTorch fallback still works
        print(f"TensorRT export failed ({type(e).__name__}: {e}); run.py will use yolo11l.pt")
        try:
            os.remove(os.path.join(HERE, "yolo11l.engine"))
        except OSError:
            pass


if __name__ == "__main__":
    main()
