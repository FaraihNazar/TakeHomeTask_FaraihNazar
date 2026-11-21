#!/usr/bin/env python
"""Video indexing at speed: per-second object presence + people count with YOLO11l.

Usage:
    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline (see README.md for details):
    reader thread  -> reads mp4 bytes from parquet row groups -> task queue
    N decoder procs -> decode with PyAV, keep K evenly spaced frames per second,
                       letterbox-resize (Ultralytics-identical) into shared memory
    feeder thread  -> copies shared-memory batches into pinned host tensors
    GPU thread     -> pad/normalise on GPU, YOLO11l (TensorRT FP16 or PyTorch),
                       batched class-aware NMS, per-frame counts, per-second vote,
                       writes JSON lines
"""
import argparse
import glob
import io
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
IMGSZ = 640
STRIDE = 32
CONF = 0.4
IOU = 0.7
PAD_VALUE = 114
# COCO class ids used by YOLO11 -> output names (order = section 1 of the brief)
TARGET_IDS = [0, 2, 16, 63, 67]
TARGET_NAMES = ["person", "car", "dog", "laptop", "cell phone"]
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# Geometry helpers (mirror ultralytics.data.augment.LetterBox, auto=True)
# ----------------------------------------------------------------------------
def letterbox_geometry(w, h, rect=True):
    """Return (new_w, new_h, pad_w_total_shape W, H, left, top) for a display w x h frame."""
    r = min(IMGSZ / h, IMGSZ / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = IMGSZ - new_w, IMGSZ - new_h
    if rect:
        dw, dh = dw % STRIDE, dh % STRIDE
    W, H = new_w + dw, new_h + dh
    left = int(round(dw / 2 - 0.1))
    top = int(round(dh / 2 - 0.1))
    return new_w, new_h, W, H, left, top


def rotation_k(frame, stream):
    """Number of 90-degree counter-clockwise turns (np.rot90 k) to display the frame upright."""
    rot = None
    try:
        rot = frame.rotation  # PyAV >= 14: CCW degrees from the display matrix
    except Exception:
        rot = None
    if not rot:
        try:
            tag = stream.metadata.get("rotate")  # legacy tag: clockwise degrees
            if tag:
                rot = -int(float(tag))
        except Exception:
            pass
    if not rot:
        return 0
    return int(round(float(rot) / 90.0)) % 4


# ----------------------------------------------------------------------------
# Decoder worker process
# ----------------------------------------------------------------------------
def worker_main(wid, task_q, result_q, free_q, shm_names, slot_bytes, samples_per_sec, batch, threads):
    import av
    import cv2
    from multiprocessing import resource_tracker, shared_memory

    cv2.setNumThreads(1)
    try:
        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass
    shms = []
    for name in shm_names:
        s = shared_memory.SharedMemory(name=name)
        try:  # the parent owns the segments; don't let this process's tracker unlink them
            resource_tracker.unregister(s._name, "shared_memory")
        except Exception:
            pass
        shms.append(s)

    while True:
        task = task_q.get()
        if task is None:
            break
        vid, data = task
        result_q.put(("start", wid, vid))
        try:
            decode_video(vid, data, wid, result_q, free_q, shms, samples_per_sec, batch, threads, av, cv2)
        except Exception as e:  # never let a bad video kill the worker
            result_q.put(("error", wid, vid, f"worker failure: {type(e).__name__}: {e}"[:200], None))
        del data
    result_q.put(("exit", wid))


def decode_video(vid, data, wid, result_q, free_q, shms, K, batch, threads, av, cv2):
    t_start = time.perf_counter()
    try:
        container = av.open(io.BytesIO(data), mode="r", metadata_errors="ignore")
    except Exception as e:
        result_q.put(("error", wid, vid, f"could not open video: {type(e).__name__}"[:200], None))
        return
    try:
        if not container.streams.video:
            result_q.put(("error", wid, vid, "no video stream", None))
            return
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if threads > 0:
            stream.thread_count = threads
        tb = stream.time_base
        tb_num, tb_den = (tb.numerator, tb.denominator) if tb else (1, 1)
        rate = stream.average_rate or stream.guessed_rate or 30
        all_frames = K <= 0

        slot = None
        buf = None
        secs = []
        n_in_chunk = 0
        geom = None  # (dec_w, dec_h, k, new_w, new_h, W, H, left, top)
        pts0 = None
        last_sec = -1
        cur_sec = -1
        taken = set()  # slots taken in current second
        n_frames = 0
        n_sampled = 0
        idx = 0

        def flush():
            nonlocal slot, buf, secs, n_in_chunk
            if n_in_chunk:
                result_q.put(("frames", wid, vid, slot, n_in_chunk, geom, np.asarray(secs, dtype=np.int32)))
            elif slot is not None:
                free_q.put(slot)
            slot, buf, secs, n_in_chunk = None, None, [], 0

        try:
            for frame in container.decode(stream):
                # ---- timestamp -> (second, sub-slot), exact integer arithmetic
                pts = frame.pts
                if pts is None:
                    num, den = idx * rate.denominator, rate.numerator  # seconds = idx / fps
                    if pts0 is None:
                        pts0 = 0
                else:
                    if pts0 is None:
                        pts0 = pts
                    num, den = (pts - pts0) * tb_num, tb_den
                idx += 1
                n_frames += 1
                if num < 0:
                    continue  # before the first decoded frame
                sec = num // den
                if sec > cur_sec:
                    if cur_sec >= 0 and sec != cur_sec:
                        taken.clear()
                    cur_sec = sec
                elif sec < cur_sec:
                    # out-of-order timestamp: treat as belonging to its own second, no sampling
                    last_sec = max(last_sec, sec)
                    continue
                last_sec = max(last_sec, sec)
                if all_frames:
                    sub = n_frames
                else:
                    sub = (num * K) // den - sec * K
                if sub in taken:
                    continue
                taken.add(sub)

                # ---- geometry from the first frame
                if geom is None:
                    k = rotation_k(frame, stream)
                    fw, fh = frame.width, frame.height
                    dw_, dh_ = (fh, fw) if k % 2 else (fw, fh)
                    new_w, new_h, W, H, left, top = letterbox_geometry(dw_, dh_)
                    geom = (k, new_w, new_h, W, H, left, top)
                k, new_w, new_h = geom[0], geom[1], geom[2]

                img = frame.to_ndarray(format="bgr24")
                if k:
                    img = np.rot90(img, k)
                if img.shape[1] != new_w or img.shape[0] != new_h:
                    img = cv2.resize(np.ascontiguousarray(img), (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                if slot is None:
                    slot = free_q.get()
                    buf = np.ndarray((batch, new_h, new_w, 3), dtype=np.uint8, buffer=shms[slot].buf)
                buf[n_in_chunk] = img
                secs.append(sec)
                n_in_chunk += 1
                n_sampled += 1
                if n_in_chunk == batch:
                    flush()
        except Exception as e:
            flush()
            # seconds strictly before the one being decoded are complete
            result_q.put(("error", wid, vid, f"decode failed: {type(e).__name__}"[:200], cur_sec - 1))
            return
        flush()
        if n_frames == 0 or last_sec < 0:
            result_q.put(("error", wid, vid, "could not decode video", None))
            return
        result_q.put(("done", wid, vid, int(last_sec), n_frames, n_sampled, time.perf_counter() - t_start))
    finally:
        try:
            container.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Parquet reader (thread in the main process)
# ----------------------------------------------------------------------------
def reader_thread(files, task_q, result_q, skip, max_videos, stop_evt):
    import pyarrow.parquet as pq

    total = 0
    try:
        for path in files:
            fname = os.path.basename(path)
            try:
                pf = pq.ParquetFile(path)
                md = pf.metadata
                n_groups = md.num_row_groups
            except Exception as e:
                vid = f"{fname}:0"
                if vid not in skip:
                    result_q.put(("error", None, vid, f"could not read parquet file: {type(e).__name__}"[:200], None))
                    total += 1
                continue
            row0 = 0
            for g in range(n_groups):
                if stop_evt.is_set():
                    return
                n_rows = md.row_group(g).num_rows
                rows = [r for r in range(row0, row0 + n_rows) if f"{fname}:{r}" not in skip]
                if max_videos and total >= max_videos:
                    break
                if rows:
                    col = None
                    err = None
                    try:
                        col = pf.read_row_group(g, columns=["mp4"]).column("mp4")
                    except Exception as e:
                        err = f"could not read video bytes: {type(e).__name__}"
                    for r in rows:
                        if max_videos and total >= max_videos:
                            break
                        vid = f"{fname}:{r}"
                        total += 1
                        data = None
                        if col is not None:
                            try:
                                v = col[r - row0].as_py()
                                if isinstance(v, dict):
                                    v = v.get("bytes")
                                data = v
                            except Exception as e:
                                err = f"could not read video bytes: {type(e).__name__}"
                        if not data:
                            result_q.put(("error", None, vid, (err or "empty video bytes")[:200], None))
                            continue
                        task_q.put((vid, data))
                    del col
                row0 += n_rows
            if max_videos and total >= max_videos:
                break
    except Exception as e:
        log(f"reader crashed: {e}")
        traceback.print_exc()
    finally:
        result_q.put(("reader_done", total))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model(weights, device, half):
    from ultralytics.nn.autobackend import AutoBackend

    model = AutoBackend(weights=weights, device=device, fp16=half, fuse=True, verbose=False)
    model.eval()
    return model


def resolve_weights(arg):
    if arg and arg != "auto":
        return arg
    for name in ("yolo11l.engine", "yolo11l.pt"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            return p
    return "yolo11l.pt"  # ultralytics will try to download it


class Detector:
    def __init__(self, weights, device, half):
        import torch
        import torchvision

        self.torch = torch
        self.nms = torchvision.ops.batched_nms
        self.device = device
        self.model = load_model(weights, device, half)
        self.dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
        lut = torch.full((80,), -1, dtype=torch.long)
        for i, c in enumerate(TARGET_IDS):
            lut[c] = i
        self.lut = lut.to(device)
        self.targets = torch.tensor(TARGET_IDS, device=device)

    def __call__(self, x_u8, geom):
        """x_u8: (n, new_h, new_w, 3) uint8 BGR tensor on device. Returns (n, 5) int counts on CPU."""
        torch = self.torch
        k, new_w, new_h, W, H, left, top = geom
        n = x_u8.shape[0]
        x = torch.full((n, 3, H, W), PAD_VALUE / 255.0, dtype=self.dtype, device=self.device)
        # BGR -> RGB, HWC -> CHW, /255 (same as ultralytics preprocess)
        x[:, :, top:top + new_h, left:left + new_w] = x_u8.flip(-1).permute(0, 3, 1, 2).to(self.dtype) / 255.0
        with torch.inference_mode():
            pred = self.model(x)
        while isinstance(pred, (list, tuple)):
            pred = pred[0]
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred, device=self.device)
        pred = pred.to(self.device)
        # pred: (n, 4 + 80, A)
        p = pred.transpose(1, 2)
        cls = p[..., 4:84]
        conf, j = cls.max(-1)
        keep = (conf >= CONF) & torch.isin(j, self.targets)
        b, a = keep.nonzero(as_tuple=True)
        counts = torch.zeros(n * 5, dtype=torch.long, device=self.device)
        if b.numel():
            xywh = p[b, a, :4].float()
            boxes = torch.empty_like(xywh)
            boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
            boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
            scores = conf[b, a].float()
            cj = j[b, a]
            kept = self.nms(boxes, scores, b * 80 + cj, IOU)
            idx = b[kept] * 5 + self.lut[cj[kept]]
            counts += torch.bincount(idx, minlength=n * 5)
        return counts.view(n, 5).cpu().numpy()


# ----------------------------------------------------------------------------
# Per-second aggregation (reference rules from section 5 of the brief)
# ----------------------------------------------------------------------------
def aggregate(secs, counts, n_seconds):
    """secs: (m,) int, counts: (m,5) int. Returns list of (objects, people) for 0..n_seconds-1."""
    out = [([], 0)] * n_seconds
    if n_seconds <= 0 or secs.size == 0:
        return out
    mask = secs < n_seconds
    secs, counts = secs[mask], counts[mask]
    n_per = np.bincount(secs, minlength=n_seconds)
    present = counts > 0
    votes = np.stack([np.bincount(secs, weights=present[:, c], minlength=n_seconds) for c in range(5)], 1)
    is_present = (votes * 2 >= n_per[:, None]) & (n_per[:, None] > 0)
    # median of people per second, rounded down on halves
    people = counts[:, 0]
    order = np.lexsort((people, secs))
    sp = people[order]
    starts = np.concatenate([[0], np.cumsum(n_per)[:-1]])
    med = np.zeros(n_seconds, dtype=np.int64)
    has = n_per > 0
    lo = sp[(starts + (n_per - 1) // 2)[has]]
    hi = sp[(starts + n_per // 2)[has]]
    med[has] = (lo + hi) // 2
    out = []
    for s in range(n_seconds):
        objs = [TARGET_NAMES[c] for c in range(5) if is_present[s, c]]
        out.append((objs, int(med[s])))
    return out


class VideoState:
    __slots__ = ("secs", "counts", "max_sec")

    def __init__(self):
        self.secs, self.counts, self.max_sec = [], [], -1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl file")
    ap.add_argument("--samples-per-sec", type=int, default=5,
                    help="frames analysed per second of video (evenly spaced)")
    ap.add_argument("--all-frames", action="store_true", help="analyse every frame (reference mode)")
    ap.add_argument("--workers", type=int, default=0, help="decoder processes (0 = auto)")
    ap.add_argument("--decoder-threads", type=int, default=0, help="ffmpeg threads per decoder (0 = auto)")
    ap.add_argument("--batch", type=int, default=32, help="frames per GPU batch (<= engine max batch)")
    ap.add_argument("--weights", default="auto", help="yolo11l.engine / yolo11l.pt (auto picks engine if present)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-half", action="store_true", help="run PyTorch weights in FP32")
    ap.add_argument("--resume", action="store_true", help="append to output, skipping videos already in it")
    ap.add_argument("--max-videos", type=int, default=0, help="stop after N videos (testing)")
    ap.add_argument("--video-timeout", type=float, default=900, help="seconds without progress before a worker is killed")
    return ap.parse_args()


def load_done(path):
    """For --resume: videos already present in the output. Drops a trailing partial line."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "rb") as f:
        data = f.read()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with open(path, "r+b") as f:
            f.truncate(cut)
        data = data[:cut]
    for line in data.splitlines():
        try:
            done.add(json.loads(line)["video"])
        except Exception:
            pass
    return done


def main():
    args = parse_args()
    t_begin = time.perf_counter()
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not files:
        log(f"no .parquet files found in {args.input}")
    skip = load_done(args.output) if args.resume else set()
    if skip:
        log(f"resume: {len(skip)} videos already in {args.output}")
    out = open(args.output, "a" if args.resume else "w", encoding="utf-8")

    cpu = os.cpu_count() or 2
    n_workers = args.workers or max(2, min(24, cpu - 2))
    dec_threads = args.decoder_threads or max(1, min(4, cpu // n_workers))
    K = 0 if args.all_frames else max(1, args.samples_per_sec)
    batch = max(1, args.batch)

    # ---- shared-memory frame slots
    from multiprocessing import shared_memory

    ctx = mp.get_context("spawn")
    n_slots = 2 * n_workers + 4
    slot_bytes = batch * IMGSZ * IMGSZ * 3
    shms = [shared_memory.SharedMemory(create=True, size=slot_bytes) for _ in range(n_slots)]
    shm_names = [s.name for s in shms]
    free_q = ctx.Queue()
    for i in range(n_slots):
        free_q.put(i)
    task_q = ctx.Queue(maxsize=2 * n_workers)
    result_q = ctx.Queue(maxsize=4 * n_slots)

    def spawn(wid):
        p = ctx.Process(target=worker_main, daemon=True,
                        args=(wid, task_q, result_q, free_q, shm_names, slot_bytes, K, batch, dec_threads))
        p.start()
        return p

    workers = {wid: spawn(wid) for wid in range(n_workers)}
    stop_evt = threading.Event()
    rt = threading.Thread(target=reader_thread, args=(files, task_q, result_q, skip, args.max_videos, stop_evt), daemon=True)
    rt.start()
    log(f"{len(files)} parquet files, {n_workers} decoders x {dec_threads} threads, "
        f"{'all frames' if K == 0 else f'{K} frames/s'}, batch {batch}")

    # ---- model (loads while decoders warm up)
    import torch

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    half = device.type == "cuda" and not args.no_half
    weights = resolve_weights(args.weights)
    log(f"loading {os.path.basename(weights)} on {device} (fp16={half})")
    det = Detector(weights, device, half)
    with torch.inference_mode():  # warm-up
        det(torch.zeros((batch, 384, 640, 3), dtype=torch.uint8, device=device), (0, 640, 384, 640, 384, 0, 0))
    log(f"model ready after {time.perf_counter() - t_begin:.1f}s")

    # ---- feeder thread: result_q -> pinned host tensors (keeps GPU thread busy)
    local_q = queue.Queue(maxsize=6)
    pin = device.type == "cuda"

    def feeder():
        while True:
            try:
                msg = result_q.get(timeout=1.0)
            except queue.Empty:
                if stop_evt.is_set():
                    return
                local_q.put(("tick",))
                continue
            if msg[0] == "frames":
                _, wid, vid, slot, n, geom, secs = msg
                k, new_w, new_h = geom[0], geom[1], geom[2]
                view = np.ndarray((n, new_h, new_w, 3), dtype=np.uint8, buffer=shms[slot].buf)
                t = torch.empty((n, new_h, new_w, 3), dtype=torch.uint8, pin_memory=pin)
                t.numpy()[:] = view
                del view
                free_q.put(slot)
                local_q.put(("frames", wid, vid, t, geom, secs))
            else:
                local_q.put(msg)

    ft = threading.Thread(target=feeder, daemon=True)
    ft.start()

    # ---- GPU / bookkeeping loop
    states = {}          # vid -> VideoState
    current = {}         # wid -> (vid, last_activity)
    finalized = set()
    total = None
    exited = set()
    stats = dict(videos=0, errors=0, seconds=0, frames_decoded=0, frames_infer=0, decode_time=0.0,
                 gpu_time=0.0, wait_time=0.0)
    last_report = time.perf_counter()

    def finalize(vid, last_sec, error=None):
        if vid in finalized:
            return
        finalized.add(vid)
        st = states.pop(vid, None)
        lines = []
        n_sec = last_sec + 1 if last_sec is not None and last_sec >= 0 else 0
        if n_sec > 0:
            if st is not None and st.secs:
                secs = np.concatenate(st.secs)
                cnt = np.concatenate(st.counts)
            else:
                secs, cnt = np.zeros(0, np.int64), np.zeros((0, 5), np.int64)
            for s, (objs, ppl) in enumerate(aggregate(secs.astype(np.int64), cnt, n_sec)):
                lines.append(json.dumps({"video": vid, "second": s, "objects": objs, "people": ppl}) + "\n")
            stats["seconds"] += n_sec
        if error is not None:
            lines.append(json.dumps({"video": vid, "error": error}) + "\n")
            stats["errors"] += 1
        stats["videos"] += 1
        out.write("".join(lines))
        out.flush()

    try:
        while True:
            if total is not None and len(finalized) >= total:
                break
            t0 = time.perf_counter()
            msg = local_q.get()
            stats["wait_time"] += time.perf_counter() - t0
            kind = msg[0]
            now = time.perf_counter()

            if kind == "frames":
                _, wid, vid, t, geom, secs = msg
                if wid in current:
                    current[wid] = (vid, now)
                if vid in finalized:
                    continue
                t1 = time.perf_counter()
                x = t.to(device, non_blocking=True)
                counts = det(x, geom)
                stats["gpu_time"] += time.perf_counter() - t1
                stats["frames_infer"] += len(secs)
                st = states.setdefault(vid, VideoState())
                st.secs.append(secs)
                st.counts.append(counts)
                st.max_sec = max(st.max_sec, int(secs.max()))
            elif kind == "start":
                _, wid, vid = msg
                current[wid] = (vid, now)
                states.setdefault(vid, VideoState())
            elif kind == "done":
                _, wid, vid, last_sec, n_frames, n_sampled, dt = msg
                current.pop(wid, None)
                stats["frames_decoded"] += n_frames
                stats["decode_time"] += dt
                finalize(vid, last_sec)
            elif kind == "error":
                _, wid, vid, err, last_ok = msg
                if wid is not None:
                    current.pop(wid, None)
                finalize(vid, last_ok, error=err)
            elif kind == "reader_done":
                total = msg[1]
                for _ in range(n_workers):
                    task_q.put(None)
            elif kind == "exit":
                exited.add(msg[1])

            # ---- worker health: crashed or hung decoders
            for wid, p in list(workers.items()):
                cur = current.get(wid)
                # exitcode 0 = clean exit (its last messages may still be queued); only a
                # non-zero exit is a crash, and we wait until the queues are drained to handle it
                dead = (p.exitcode not in (None, 0) and wid not in exited
                        and kind == "tick" and local_q.empty())
                hung = cur is not None and now - cur[1] > args.video_timeout
                if dead or hung:
                    if hung and p.is_alive():
                        p.terminate()
                        p.join(5)
                    if cur is not None:
                        vid = cur[0]
                        st = states.get(vid)
                        log(f"decoder {wid} {'hung' if hung else 'crashed'} on {vid}")
                        finalize(vid, (st.max_sec - 1) if st else None, error="decoder crashed on this video")
                        current.pop(wid, None)
                    workers[wid] = spawn(wid)
                    if total is not None:
                        task_q.put(None)  # the dead worker may have eaten its sentinel

            if now - last_report > 30:
                last_report = now
                el = now - t_begin
                log(f"{stats['videos']} videos ({stats['errors']} errors), {stats['seconds']} s of video, "
                    f"{stats['frames_infer']} frames inferred, {el:.0f}s elapsed, "
                    f"{stats['frames_infer'] / max(el, 1e-9):.0f} inf-fps, gpu busy {100 * stats['gpu_time'] / el:.0f}%")
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        stop_evt.set()
        out.close()
        for p in workers.values():
            if p.is_alive():
                p.terminate()
        for p in workers.values():
            p.join(2)
        for s in shms:
            try:
                s.close()
                s.unlink()
            except Exception:
                pass

    el = time.perf_counter() - t_begin
    log(f"finished: {stats['videos']} videos, {stats['errors']} errors, {stats['seconds']} seconds of video, "
        f"{stats['frames_decoded']} frames decoded, {stats['frames_infer']} frames inferred in {el:.1f}s")
    log(f"time split: GPU thread busy {stats['gpu_time']:.1f}s, GPU thread waiting for frames "
        f"{stats['wait_time']:.1f}s, decoder time summed over workers {stats['decode_time']:.1f}s")


if __name__ == "__main__":
    main()
