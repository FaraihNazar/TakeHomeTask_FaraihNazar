# Video indexing at speed — YOLO11l over FineVideo

For each second of each video, `run.py` records which of *person, car, dog, laptop,
cell phone* are present and how many people are visible, using Ultralytics YOLO11l
(`yolo11l.pt`, imgsz 640, conf 0.4, class-aware NMS IoU 0.7).

## Install and run

Tested on Ubuntu with Python 3.11–3.13 and an NVIDIA driver supporting CUDA 12.
No `apt-get` packages are needed (PyAV wheels bundle FFmpeg).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # requirements.txt adds NVIDIA's index for TensorRT
```

**Preparation command** (run once, untimed, needs internet):

```bash
python prepare.py
```

It downloads `yolo11l.pt` next to `run.py` and builds a TensorRT FP16 engine
(`yolo11l.engine`, dynamic shapes, max batch 32) for the local GPU. It takes a few minutes.
If the engine build fails, `run.py` falls back to the `.pt` weights in FP16.

**Timed run** (no internet needed):

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

Useful options: `--samples-per-sec K` (default 5), `--all-frames` (every frame, i.e. reference
mode), `--workers N` (decoder processes, default `min(24, cpus-2)`), `--resume` (append to an
existing output and skip videos already in it), `--max-videos N` (testing).

## How it works

```
reader thread ── mp4 bytes per parquet row ──► task queue
N decoder processes (PyAV) ── K frames/s, letterboxed ──► shared-memory slots
feeder thread ── copy into pinned host tensors ──► GPU thread
GPU thread: pad + normalise on GPU → TensorRT FP16 YOLO11l → batched NMS on GPU
            → per-frame counts → per-second vote/median → JSON lines
```

* **Sampling.** Every frame is decoded (timestamps decide which second a frame belongs to,
  using exact integer `pts × time_base` arithmetic from the first decoded frame), but only
  `K` evenly spaced frames per second are converted and sent to the model. Per second, an object
  is *present* if it is detected in at least half of the analysed frames, and *people* is the
  median person count rounded down — the same rules the reference applies to all frames.
* **Preprocessing identical to Ultralytics.** Rotation tags are applied first (`frame.rotation`,
  checked pixel-for-pixel against OpenCV's auto-rotation), then the same rectangular letterbox
  as Ultralytics `LetterBox(auto=True, stride=32)` with `cv2.INTER_LINEAR` and padding 114,
  BGR→RGB, /255.
* **Inference.** A small TensorRT 10 runner on a dedicated CUDA stream (Ultralytics'
  `AutoBackend` mishandles dynamic-batch engines). NMS for all frames of a batch is one
  `torchvision.ops.batched_nms` call, restricted to the five target classes. That gives the same
  result as Ultralytics' class-aware NMS, because classes never suppress each other.
* **Robustness.** Unreadable parquet files, row groups or videos produce an error line;
  a video that fails partway keeps the seconds decoded before the failure plus an error line.
  Decoder processes that crash or hang are replaced and their video is reported as an error.

## Validation

`tools/reference.py` is an independent, slow implementation using only the plain Ultralytics API
(`YOLO.predict` on every frame read with OpenCV), and `tools/compare.py` scores a results
file with the three checks from the brief. On FineVideo videos (Colab T4):

| setting | objects F1 | people exact | people ±1 |
|---|---|---|---|
| `--all-frames` vs `tools/reference.py` (2 videos, 813 s) | 1.0000 | 1.0000 | 1.0000 |
| 1 frame/s vs all frames (20 videos, 5763 s) | 0.9428 | 0.8756 | 0.9821 |
| 2 frames/s | 0.9601 | 0.9134 | 0.9925 |
| 3 frames/s | 0.9765 | 0.9386 | 0.9960 |
| **5 frames/s (default)** | **0.9834** | **0.9573** | **0.9984** |

`tools/check_results.py` checks that a results file covers every row of every parquet file
with contiguous seconds.

## Files

| file | contents |
|---|---|
| `run.py` | the pipeline |
| `prepare.py` | weight download + TensorRT engine build |
| `tools/reference.py`, `tools/compare.py`, `tools/check_results.py` | validation |
| `results.jsonl.gz` | results for the first 50 FineVideo parquet files |
| `HARDWARE.md`, `WRITEUP.md` | hardware used and write-up answers |
| `ai_chats/` | AI chat history |
