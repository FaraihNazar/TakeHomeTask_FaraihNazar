# HARDWARE

All 50 Parquet files (`train-00000` to `train-00049`: 1,650 videos, 128.3 hours of video) were processed on **Google Colab, free tier, with an NVIDIA T4 GPU**.

## Machine

| | |
|---|---|
| **Where** | Google Colab (free tier), standard T4 runtime |
| **GPU** | NVIDIA Tesla T4, 15,360 MiB (16 GB class) |
| **CPU** | Intel Xeon @ 2.00 GHz, 2 vCPUs |
| **RAM** | 12 GiB |
| **Disk** | SSD (reported as non-rotational), about 113 GB local VM disk |
| **OS** | Ubuntu 24.04.4 LTS |
| **Python** | 3.13 (Colab default; the grading machine uses 3.11) |
| **Key packages** | torch 2.6.0+cu124, TensorRT 10.13.3.9 (`tensorrt-cu12`), ultralytics 8.3.100, av 14.2.0 |

The machine details above were read in the second session. The first session used the same standard free Colab T4 runtime.

## Run time

The full run was done in **two sessions on two Google accounts**. The first account's free GPU quota ran out after about 81% of the footage. The second session continued with `--resume`, which skips every video already in `results.jsonl`, and appended to the same file. Both sessions used the same `run.py` with default options (5 sampled frames per second, TensorRT FP16 engine, batch size 32).

| Session | Started (UTC, 2026-11-21) | Videos | Footage | Time |
|---|---|---|---|---|
| 1 | 13:52:38 | 1,300 | 364,679 s (101.3 h) | 9,941 s (2 h 45.7 min) |
| 2 | 18:33:50 | 350 | 97,078 s (27.0 h) | 2,808.8 s (46.8 min) |
| **Total** | | **1,650** | **461,757 s (128.3 h)** | **12,750 s (3 h 32.5 min)** |

Session 1 stopped when the free GPU quota ran out, so its log has no final `finished` line. Its time is the `elapsed` value on its last progress line (16:38:19 UTC: 1,300 videos, 364,679 s of video). The session ended within the following 30 s, before the next progress line was due. Session 2 resumed from exactly those 1,300 videos. The logs are `run_log_session1.txt` and `run_log_session2.txt`. Session 2's log also contains an empty attempt at 17:53, which found no new videos to process.

- **Speed:** session 1 ran at 36.7 times real time (364,679 s of video in 9,941 s) and session 2 at 34.6 times. Overall that is 461,757 s of video in 12,750 s, about 36 times real time.
- **Wall clock:** 13:52:38 to 19:20:39 UTC, about 5 h 28 min, including the gap between the sessions.
- **Not included in the times above:** the untimed preparation before each session, about 13 minutes to build the TensorRT engine and 4-10 minutes to download the 25 GB of data.
- **Not a single uninterrupted run:** the times above are for this Colab T4 with 2 vCPUs, where video decoding is the limit. The task's timing is done on the grading machine.
