"""프레임 폴더에 YOLO 모델을 하나씩 돌려 사람과 테이블 검출 결과를 CSV로 저장한다."""
import argparse
import csv
import resource
import sys
import time
from multiprocessing import get_context
from pathlib import Path

from zone import in_zone_conf, load_zone

MODELS = [f"{v}{s}" for v in ("yolov8", "yolo11", "yolo26") for s in "nsmlx"]
PERSON, TABLE = 0, 60  # COCO 클래스 번호, 60은 dining table


def fmt(boxes):
    return ";".join(" ".join(map(str, b)) for b in boxes)  # x1 y1 x2 y2 conf


def run(model_name, frames, imgsz, conf, device, zone_path):
    from ultralytics import YOLO

    zone = load_zone(zone_path) if zone_path else None
    model = YOLO(f"{model_name}.pt")
    model.predict(frames[0], imgsz=imgsz, device=device, verbose=False)  # 워밍업
    rows = []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=imgsz, conf=conf, classes=[PERSON, TABLE], device=device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = [(int(b.cls), [*map(round, b.xyxy[0].tolist()), round(float(b.conf), 3)]) for b in r.boxes]
        persons = [d for c, d in dets if c == PERSON]
        tables = [d for c, d in dets if c == TABLE]
        row = {
            "model": model_name,
            "frame": Path(f).name,
            "infer_ms": round(ms, 1),
            "person_conf": max((b[4] for b in persons), default=0),
            "persons": fmt(persons),
            "table_conf": max((b[4] for b in tables), default=0),
            "tables": fmt(tables),
        }
        if zone is not None:
            row["in_zone_conf"] = in_zone_conf(persons, zone)
        rows.append(row)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # macOS는 바이트, Linux는 KB
    return rows, peak_mb


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("frames", help="capture.py가 저장한 프레임 폴더")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.05, help="낮게 두고 임계값은 CSV를 보고 정한다")
    p.add_argument("--device", default="cpu", help="cpu 또는 mps")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--zone", help="zone.py로 만든 감지 영역 JSON. 주면 영역 안 사람의 최고 신뢰도를 기록한다")
    args = p.parse_args()

    frames = sorted(str(f) for f in Path(args.frames).glob("*.jpg"))
    ctx = get_context("spawn")
    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, ["model", "frame", "infer_ms", "person_conf", "persons", "table_conf", "tables", "in_zone_conf"])
        w.writeheader()
        print(f"프레임 {len(frames)}장, imgsz {args.imgsz}, device {args.device}")
        for m in args.models:
            # 모델마다 새 프로세스에서 돌려야 최대 메모리를 따로 잴 수 있다
            with ctx.Pool(1) as pool:
                rows, peak_mb = pool.apply(run, (m, frames, args.imgsz, args.conf, args.device, args.zone))
            w.writerows(rows)
            avg_ms = sum(r["infer_ms"] for r in rows) / len(rows)
            print(f"{m:8s}  평균 {avg_ms:6.0f}ms/장  최대 메모리 {peak_mb:5.0f}MB")
    print(f"결과: {args.out}")
