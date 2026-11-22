1-I am not sure about the runtime of my first working version on all 50 files,so I can't give an exact comparison. My final version took around 3 hours to process all 50 files.The biggest measured improvement came from sampling 5 frames per second instead of processing every frame. On the same 20 videos this reduced processing time from around 12.5 minutes to 2.6 minutes.This was the most significant improvement I measured.

2-The script first reads the input folder and opens each Parquet file using PyArrow. It iterates through row groups, extracts video bytes, and uses multiple PyAV worker processes to decode videos frame by frame. It samples 5 evenly spaced frames per second to reduce inference time. The frames are then preprocessed and sent in batches to YOLO11l using TensorRT FP16.The model detects five classes, person, car, dog, laptop, and cell phone, with a confidence threshold of 0.4. Finally, detections are aggregated per second, and one JSONL line is written for each second of video.


3-The GPU inference stage was the main bottleneck. I measured this using the script's built in timers. In session2, the GPU thread was actively processing for 2,087.7 seconds and waiting for frames for 703.2 seconds and the decoder processes were approximately 99% busy.In first session,GPU utilization was around 83%.These measurements show that GPU inference takes the largest share of processing time, while video decoding also puts pressure on the pipeline.
4-I checked my results against a separate reference script that used YOLO on every frame.First, I tested my optimized pipeline with all frames to make sure the results matched. Then I tested different sampling rates 1,2,3 and 5 FPS using compare.py. At 5 FPS, I got an object detection F1 score of 0.9834 and 95.73% exact people count accuracy.I also used check_results.py to make sure no video seconds were missing.

5-The AI mostly suggest implementing several parts at once.I avoided this and worked step by step so I could test each part separately. Its suggestions were mostly helpful, but it did not test the pipeline on multiple videos after processing all the videos.I noticed this gap and tested it myself to make sure everything worked correctly.

6- If I had one more week, I would focus on improving video decoding and testing the pipeline on more hardware.My logs showed that the CPU decoders were mostly busy, so I would test on an RTX 4090 instead of relying only on the T4 to tune the number of workers and batch size. Also, I would test more rotated videos and weaker object classes to improve reliability.



