"""라즈베리파이 스트림(또는 영상 파일)에서 N초마다 프레임을 저장한다."""
import argparse
import os
import time
from pathlib import Path

import cv2


def frames(source, interval):
    """interval초마다 (t, frame)을 내준다. 파일은 영상 시간, 스트림은 실제 시간 기준이다.

    스트림이 끊기면 (t, None)을 내주고 2초 뒤 다시 연결한다.
    """
    is_file = os.path.isfile(source)
    cap = cv2.VideoCapture(source)
    start = time.time()
    next_t = 0.0
    while True:
        # 스트림은 계속 grab 해서 버퍼에 오래된 프레임이 쌓이지 않게 한다
        if not cap.grab():
            if is_file:
                return
            yield time.time() - start, None
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(source)
            continue
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else time.time() - start
        if t < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        yield t, frame
        next_t = t + interval


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="스트림 URL 또는 영상 파일 경로")
    p.add_argument("--out", default="frames", help="프레임을 저장할 폴더")
    p.add_argument("--interval", type=float, default=5.0, help="저장 간격(초)")
    p.add_argument("--duration", type=float, default=0, help="수집 시간(초), 0이면 끝날 때까지")
    args = p.parse_args()

    is_file = os.path.isfile(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for t, frame in frames(args.source, args.interval):
        if args.duration and t > args.duration:
            break
        if frame is None:
            print("프레임 수신 실패, 2초 뒤 다시 연결합니다")
            continue
        label = f"t{t:07.1f}s" if is_file else time.strftime("%H%M%S")
        cv2.imwrite(str(out / f"{saved:05d}_{label}.jpg"), frame)
        saved += 1

    print(f"{saved}장 저장: {out}")
