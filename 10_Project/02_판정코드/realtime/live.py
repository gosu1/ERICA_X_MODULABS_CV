"""라즈베리파이 스트림(또는 카메라, 영상 파일)에 YOLO를 실시간으로 돌려 화면에 보여 준다.

q 또는 Esc: 종료
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

# RTSP를 UDP로 받으면 Wi-Fi에서 패킷이 빠져 깨진 프레임이 들어오므로 TCP로 받는다
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 10_Project의 zone.py를 쓴다
from zone import in_zone_conf, load_zone

ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class LatestFrame:
    """별도 스레드에서 프레임을 계속 받아 가장 최근 것만 들고 있는다. 추론이 느려도 화면이 밀리지 않는다."""

    def __init__(self, source):
        self.source = int(source) if source.isdigit() else source  # 숫자면 USB 카메라 번호
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임 수신 실패, 2초 뒤 다시 연결합니다")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.source)
                continue
            with self.lock:
                self.frame = frame

    def get(self):
        with self.lock:
            return self.frame


def draw(frame, persons, zone, th):
    """사람 박스를 그린다. 감지 영역이 있으면 영역 안 사람은 초록, 밖은 회색으로 그린다."""
    if zone is not None:
        cv2.polylines(frame, [zone.astype(np.int32)], True, (0, 200, 255), 2)
    for x1, y1, x2, y2, c in persons:
        if c < th:
            continue
        inside = zone is None or cv2.pointPolygonTest(zone, ((x1 + x2) / 2, (y1 + y2) / 2), False) >= 0
        color = (0, 220, 0) if inside else (160, 160, 160)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"person {c:.2f}", (int(x1), max(20, int(y1) - 6)), 0, 0.6, color, 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="RTSP 주소, USB 카메라 번호(0) 또는 영상 파일")
    p.add_argument("--model", default="yolo26s")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--th", type=float, default=0.30, help="사람으로 인정하는 신뢰도")
    p.add_argument("--zone", help="감지 영역 JSON. 주면 영역 안 사람 수를 표시한다")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="카메라가 돌아가 있을 때 화면 회전")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    model = YOLO(f"{args.model}.pt")
    zone = load_zone(args.zone) if args.zone else None
    stream = LatestFrame(args.source)
    print(f"{args.model}, device {args.device}, 종료: q")

    fps, last = 0.0, time.time()
    while True:
        frame = stream.get()
        if frame is None:
            time.sleep(0.05)
            continue
        frame = frame.copy()
        if args.rotate:
            frame = cv2.rotate(frame, ROTATE[args.rotate])
        t0 = time.perf_counter()
        r = model.predict(frame, imgsz=args.imgsz, conf=0.1, classes=[0], device=args.device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        persons = [[*b.xyxy[0].tolist(), float(b.conf)] for b in r.boxes]
        draw(frame, persons, zone, args.th)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)  # 화면에 보이는 FPS를 부드럽게 만든다
        last = now
        count = sum(c >= args.th for *_, c in persons)
        status = f"{args.model}  {ms:.0f} ms  {fps:.1f} FPS  people {count}"
        if zone is not None:
            status += f"  in zone {in_zone_conf(persons, zone):.2f}"
        cv2.putText(frame, status, (10, 30), 0, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, status, (10, 30), 0, 0.8, (255, 255, 255), 2)
        cv2.imshow("live", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
