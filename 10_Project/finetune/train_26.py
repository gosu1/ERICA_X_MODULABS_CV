"""YOLO26 s·m·l을 yolo11s v2와 같은 데이터·설정으로 차례대로 파인튜닝한다.

GPU 메모리(8GB)가 부족하면 batch를 8 → 4로 줄여 다시 시도한다.
"""

import gc

import torch
from ultralytics import YOLO

P = r"D:\CV프로젝트\테이블학습\runs"
W = r"D:\CV프로젝트\YOLO비교\weights"

for size in ["s", "m", "l"]:
    for batch in (8, 4):
        try:
            m = YOLO(rf"{W}\yolo26{size}.pt")
            m.train(data="dataset_v2/data.yaml", epochs=25, imgsz=960, batch=batch, workers=0,
                    optimizer="AdamW", lr0=0.001, project=P, name=f"v2_yolo26{size}", exist_ok=True,
                    seed=0, verbose=False)
            print(f"DONE yolo26{size} batch={batch}", flush=True)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"OOM yolo26{size} batch={batch}", flush=True)
        finally:
            del m
            gc.collect()
            torch.cuda.empty_cache()
