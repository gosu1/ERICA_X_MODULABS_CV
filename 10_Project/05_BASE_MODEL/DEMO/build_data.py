"""관리자 대시보드용 데이터 만들기.

1차.mp4(천장 고정 카메라, 22분)를 1초마다 1장씩 꺼내
- mvp/detector.py 의 RoomDetector(파인튜닝 YOLO26s v2)로 감지 영역·사람을 판정하고
- 같은 모델로 짐(노트북·휴대폰 등)을 찾고
- 화면은 사람을 모자이크한 640x360 JPEG로 저장한다. (기획서: 얼굴 미식별, 사람 모자이크)

결과
  demo/data/frames/000000.jpg ...   모자이크 화면 (박스는 대시보드에서 그림)
  demo/data/detections.json         초별 판정 결과
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
sys.path.insert(0, str(ROOT / "mvp"))
from detector import RoomDetector  # noqa: E402

VIDEO = r"G:/내 드라이브/Cloud_Downloads/1차.mp4"
WEIGHTS = ROOT / "알박기_yolo26s_v2.pt"
OUT_W, OUT_H = 640, 360
ITEMS = {24: "가방", 26: "핸드백", 39: "물병", 41: "컵", 63: "노트북", 64: "마우스", 67: "휴대폰", 73: "책"}


def mosaic(img, boxes, block=22, pad=0.10):
    """사람 박스를 모두 합친 영역을 같은 크기의 굵은 칸으로 모자이크한다.

    박스마다 따로 하면 머리만 잡힌 작은 박스가 더 촘촘하게 덮여 얼굴이 드러날 수 있어서,
    화면 전체를 한 번 굵게 모자이크한 뒤 합친 영역에만 덮어씌운다.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        cv2.rectangle(mask, (max(0, int(x1 - px)), max(0, int(y1 - py))),
                      (min(w, int(x2 + px)), min(h, int(y2 + py))), 255, -1)
    if not mask.any():
        return img.copy()
    coarse = cv2.resize(cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA),
                        (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    out[mask > 0] = coarse[mask > 0]
    return out


def norm(b, w, h):
    return [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]


def main():
    frames_dir = DEMO / "data" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    room = RoomDetector(str(WEIGHTS))
    model = room.model
    masker = YOLO(str(ROOT / "YOLO비교" / "weights" / "yolov10s.pt"))

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = round(fps)
    rows, i, sec = [], 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            _, f = cap.retrieve()
            h, w = f.shape[:2]
            zone, persons, inside = room.detect(f)
            items_r = model.predict(f, imgsz=960, conf=0.3, classes=list(ITEMS), verbose=False)[0]
            items = [{"name": ITEMS[int(c)], "conf": round(s, 2), "box": norm(b, w, h)}
                     for b, c, s in zip(items_r.boxes.xyxy.tolist(), items_r.boxes.cls.tolist(),
                                        items_r.boxes.conf.tolist())]
            # 모자이크는 놓치지 않게 두 모델을 낮은 기준으로 합친다
            mask_boxes = []
            for m in (model, masker):
                r = m.predict(f, imgsz=960, conf=0.15, classes=[0], verbose=False)[0]
                mask_boxes += r.boxes.xyxy.tolist()
            img = cv2.resize(mosaic(f, mask_boxes), (OUT_W, OUT_H))
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])[1].tofile(str(frames_dir / f"{sec:06d}.jpg"))
            inside_ids = {id(p) for p in inside}
            rows.append({
                "t": sec,
                "present": bool(inside),
                "zone": norm((zone.x1, zone.y1, zone.x2, zone.y2), w, h),
                "table": norm((room.table.x1, room.table.y1, room.table.x2, room.table.y2), w, h) if room.table else None,
                "persons": [{"conf": round(p.conf, 2), "box": norm((p.x1, p.y1, p.x2, p.y2), w, h),
                             "inside": id(p) in inside_ids} for p in persons],
                "items": items,
            })
            sec += 1
            if sec % 120 == 0:
                print(f"{sec}s / {total / fps:.0f}s", flush=True)
        i += 1

    meta = {"video": "1차.mp4", "fps_sampled": 1, "seconds": sec, "frame_size": [OUT_W, OUT_H],
            "model": WEIGHTS.name, "rules": {"on_sec": 2, "off_sec": 10, "expand": 0.30}}
    data = {"meta": meta, "rows": rows}
    (DEMO / "data" / "detections.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # index.html 을 서버 없이 바로 열어도 되도록 같은 내용을 스크립트로도 저장한다
    (DEMO / "data" / "detections.js").write_text(
        "window.DETECTIONS=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";", encoding="utf-8")
    print("done", sec, "frames")


if __name__ == "__main__":
    main()
