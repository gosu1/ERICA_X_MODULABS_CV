"""라즈베리파이 스트림을 받아 감지 영역 안 사람 유무로 신호등 색을 정한다.

빨강: 사용중 / 노랑: 자리 비움 limit 미만 / 초록: 자리 비움 limit 이상(사용 가능)
"""
import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

import cv2
from ultralytics import YOLO

from capture import frames
from zone import in_zone_conf, load_zone


def send_signal(color, led_url):
    """신호등 색을 라즈베리파이에 보낸다. 전송 방식이 정해지면 이 함수만 고친다."""
    print(f"신호: {color}")
    if not led_url:
        return
    req = urllib.request.Request(led_url, json.dumps({"color": color}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=2)
    except OSError as e:
        print(f"신호 전송 실패: {e}")


def save_snapshot(frame, persons, folder, name, keep_sec=24 * 3600):
    """사람 부분을 모자이크한 화면을 저장하고, 하루가 지난 파일은 지운다."""
    img = frame.copy()
    for x1, y1, x2, y2, _ in persons:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        roi = img[y1:y2, x1:x2]
        if roi.size:
            small = cv2.resize(roi, (max(1, roi.shape[1] // 16), max(1, roi.shape[0] // 16)))
            img[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / name), img)
    for f in folder.glob("*.jpg"):
        if time.time() - f.stat().st_mtime > keep_sec:
            f.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="스트림 URL 또는 영상 파일 경로")
    p.add_argument("--zone", required=True, help="zone.py로 만든 감지 영역 JSON")
    p.add_argument("--model", default="yolo26s")
    p.add_argument("--device", default="mps", help="mps 또는 cpu")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--th", type=float, default=0.30, help="사람으로 인정하는 신뢰도")
    p.add_argument("--confirm", type=int, default=2, help="몇 프레임 연속 잡혀야 타이머를 초기화할지")
    p.add_argument("--interval", type=float, default=5.0, help="판정 간격(초)")
    p.add_argument("--limit-min", type=float, default=30, help="초록불로 바꾸는 자리 비움 시간(분)")
    p.add_argument("--led-url", help="신호등 색을 POST로 보낼 주소. 없으면 화면에만 출력한다")
    p.add_argument("--snapshots", default="snapshots", help="상태가 바뀐 순간의 모자이크 화면을 저장할 폴더")
    p.add_argument("--log", help="프레임별 판정 기록을 남길 CSV 경로")
    args = p.parse_args()

    model = YOLO(f"{args.model}.pt")
    zone = load_zone(args.zone)
    # 줄 단위로 바로 저장해서, 강제 종료돼도 기록이 남게 한다
    log = csv.writer(open(args.log, "w", newline="", buffering=1)) if args.log else None
    if log:
        log.writerow(["t", "in_zone_conf", "state"])

    streak, last_seen, state, down_since = 0, None, None, None
    for t, frame in frames(args.source, args.interval):
        if frame is None:
            if down_since is None:
                print("스트림이 끊겼습니다. 타이머를 멈춥니다")
                down_since = t
            continue
        if down_since is not None:
            if last_seen is not None:
                last_seen += t - down_since  # 끊긴 시간은 자리 비움 시간에 넣지 않는다
            down_since = None

        # 모자이크용으로 낮은 신뢰도 박스까지 받고, 판정은 th 기준으로 한다
        r = model.predict(frame, imgsz=args.imgsz, conf=0.1, classes=[0], device=args.device, verbose=False)[0]
        persons = [[*b.xyxy[0].tolist(), float(b.conf)] for b in r.boxes]
        conf = in_zone_conf(persons, zone)
        streak = streak + 1 if conf >= args.th else 0
        if streak >= args.confirm or last_seen is None:
            last_seen = t
        if streak >= args.confirm:
            new = "red"
        elif t - last_seen < args.limit_min * 60:
            new = "yellow"
        else:
            new = "green"

        if new != state:
            print(f"[{int(t) // 60}:{int(t) % 60:02d}] {state} → {new} (영역 안 사람 신뢰도 {conf:.2f})")
            send_signal(new, args.led_url)
            save_snapshot(frame, persons, Path(args.snapshots), f"{time.strftime('%Y%m%d-%H%M%S')}_t{t:07.1f}_{new}.jpg")
            state = new
        if log:
            log.writerow([round(t, 1), round(conf, 3), new])
