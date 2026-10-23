"""성능 확인 창: 내 카메라(웹캠)로 판정 1초 + 화면 실시간을 눈으로 확인한다.

    python 성능확인.py                 # 웹캠 0번
    python 성능확인.py --source 1      # 다른 웹캠
    python 성능확인.py --source 1차.mp4

알박기_실행.py와 같은 판정 코드를 쓰고, 두 가지를 따로 돌린다.
  - 화면: 카메라가 보내는 대로 바로 그린다 (실시간).
  - 판정: --interval 초마다 가장 최근 화면 한 장으로 YOLO26s v2 + SAM 감지 범위 + 신호등 규칙을 돌린다.
화면 위 박스·감지 범위·신호등 색은 '마지막 판정' 결과라서, 사람이 움직이면 박스가 최대 1초 늦게 따라온다.

웹캠은 기본 YOLO26s(파인튜닝 전)를 쓴다. 파인튜닝 모델(v2)은 천장에서 내려다본 스터디룸 사진으로만 학습해서
정면 웹캠 앞 사람을 거의 못 찾는다 (웹캠 14장 중 2장 검출, 기본 YOLO26s는 14장 모두 0.9 이상).
구조가 같아 속도는 같다. 천장 카메라 영상은 v2로 시작한다. 실행 중 t 키로 두 모델을 바로 바꿔 비교할 수 있다.
왼쪽 아래에 화면 fps, 판정 시간, 이 컴퓨터로 1초 간격 판정이 가능한 룸 수를 보여 준다.
키: q 종료 · m 사람 모자이크 켜기/끄기 · t 판정 모델 바꾸기 (기본 YOLO26s ↔ 파인튜닝 v2)
시작하면 초록불 시간을 시:분:초로 묻는다 (1분 30초 → 00:01:30 또는 000130, 그냥 Enter → 00:30:00).
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from 알박기_실행 import DEFAULT_WEIGHTS, HERE, LatestFrame, SharedModelDetector, ask_limit, first_existing
from room_logic import RoomTracker, Rules
from run_sam import draw, font, rule_text


class Judge(threading.Thread):
    """--interval 초마다 가장 최근 화면으로 판정하고, 결과를 한 묶음으로 넘겨준다."""

    def __init__(self, cam: LatestFrame, det: SharedModelDetector, tracker: RoomTracker, interval: float):
        super().__init__(daemon=True)
        self.cam, self.det, self.tracker, self.interval = cam, det, tracker, interval
        self.lock = threading.Lock()
        self.result = None                     # 마지막 판정 결과
        self.times: dict[str, list[float]] = {}  # 모델별 판정 한 번에 걸린 시간(ms)
        self.stamps: list[float] = []          # 판정한 시각 (실제 판정 간격을 재려고)
        self.alive = True
        self.t0 = time.monotonic()

    def run(self):
        next_t = time.monotonic()
        while self.alive:
            frame = self.cam.latest()
            if frame is not None:
                now = time.monotonic() - self.t0
                updates, model = self.det.updates, self.det.model_name
                c0 = time.perf_counter()
                persons, inside, _ = self.det.detect(frame, now)
                ms = (time.perf_counter() - c0) * 1000
                self.tracker.update(bool(inside), now, items=bool(self.det.items_inside))
                self.times.setdefault(model, []).append(ms); self.stamps.append(time.monotonic())
                with self.lock:
                    self.result = {"persons": persons, "inside": inside, "at": time.monotonic(), "ms": ms,
                                   "sam": self.det.updates != updates, "model": model}
            next_t += self.interval
            wait = next_t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.monotonic()

    def latest(self):
        with self.lock:
            return self.result


def perf_panel(vis, lines, fs):
    """왼쪽 아래에 성능 정보를 검은 상자로 그린다."""
    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(img)
    f = font(fs); pad = 10; gap = 6
    boxes = [dr.textbbox((0, 0), t, font=f) for t in lines]
    lh = max(b[3] - b[1] for b in boxes)
    w = max(b[2] - b[0] for b in boxes) + pad * 2
    h = (lh + gap) * len(lines) - gap + pad * 2
    x, y = 12, vis.shape[0] - h - 12
    dr.rounded_rectangle((x, y, x + w, y + h), radius=10, fill=(0, 0, 0))
    for i, t in enumerate(lines):
        dr.text((x + pad, y + pad + i * (lh + gap) - boxes[i][1]), t, font=f,
                fill=(255, 230, 140) if i == 0 else (255, 255, 255))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="성능 확인 창: 판정 1초 + 카메라 화면 실시간")
    ap.add_argument("--source", default="0", help="웹캠 번호(0, 1 …) 또는 영상 파일")
    ap.add_argument("--weights", help="처음 쓸 판정 모델. 기본: 웹캠은 yolo26s.pt, 영상은 알박기_yolo26s_v2.pt")
    ap.add_argument("--sam", default="sam2.1_t.pt", help="SAM 가중치 (처음 실행 시 자동으로 받음)")
    ap.add_argument("--interval", type=float, default=1.0, help="판정 간격(초)")
    ap.add_argument("--overlap", type=float, default=0.10)
    ap.add_argument("--grow", type=float, default=0.15)
    ap.add_argument("--limit", help="초록불로 바꾸는 자리 비움 시간 (00:01:30 또는 000130). 안 주면 시작할 때 물어봄")
    ap.add_argument("--privacy", action="store_true", help="처음부터 사람 모자이크 켜기 (m 키로도 전환)")
    ap.add_argument("--duration", type=float, default=0, help="이만큼(초) 보고 닫음. 0이면 q를 누를 때까지")
    ap.add_argument("--shot", help="닫을 때 마지막 화면을 이 파일로 저장")
    ap.add_argument("--switch-every", type=float, default=0, help="시연용: 이만큼(초)마다 판정 모델을 자동으로 바꿈")
    a = ap.parse_args()
    a.limit_sec = ask_limit(a.limit)
    base = str(first_existing([HERE / "yolo26s.pt", HERE.parent / "yolo26s.pt", Path("yolo26s.pt")]))  # 없으면 자동으로 받음
    models = {"기본 YOLO26s": base}
    if Path(DEFAULT_WEIGHTS).exists():
        models["파인튜닝 v2"] = str(DEFAULT_WEIGHTS)
    if a.weights:
        if not Path(a.weights).exists():
            raise SystemExit(f"모델 파일을 찾을 수 없습니다: {a.weights}")
        models = {Path(a.weights).stem: a.weights, **{k: v for k, v in models.items() if Path(v).resolve() != Path(a.weights).resolve()}}
    elif not a.source.isdigit() and "파인튜닝 v2" in models:   # 영상(천장 카메라)은 v2로 시작
        models = {"파인튜닝 v2": models["파인튜닝 v2"], "기본 YOLO26s": base}
    if a.source.isdigit() and not a.weights:
        print("웹캠은 기본 YOLO26s로 시작합니다 (파인튜닝 v2는 천장 카메라용이라 정면 사람을 거의 못 찾음). t 키로 바꿔 볼 수 있습니다.")

    import torch
    from ultralytics import SAM, YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"모델 불러오는 중: {', '.join(f'{k}({Path(v).name})' for k, v in models.items())} (장치: {dev_name})")
    blank = np.zeros((540, 960, 3), np.uint8)                     # 첫 판정이 늦지 않게 미리 한 번씩 돌려 둔다
    yolos = {}
    for name, w in models.items():
        yolos[name] = YOLO(w)
        yolos[name].predict(blank, imgsz=960, device=device, verbose=False)
    names = list(yolos)
    det = SharedModelDetector(models[names[0]], a.sam, overlap=a.overlap, grow=a.grow, device=device, move_sec=30.0)
    det.yolo, det.sam, det.model_name = yolos[names[0]], SAM(a.sam), names[0]
    det.sam(blank, bboxes=[[100.0, 100.0, 400.0, 300.0]], device=device, verbose=False)
    tracker = RoomTracker("내 카메라", Rules(limit_sec=a.limit_sec), start=0.0)

    cam = LatestFrame(a.source, loop=True)
    cam.start()
    judge = Judge(cam, det, tracker, a.interval)
    judge.start()
    print("성능 확인 창을 엽니다. q: 종료, m: 사람 모자이크 켜기/끄기, t: 판정 모델 바꾸기")

    win = "알박기 성능 확인 (q 종료 · m 모자이크 · t 모델)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    privacy, last_id, shown, vis = a.privacy, None, [], None
    started = last_switch = time.monotonic()
    try:
        while not (a.duration and time.monotonic() - started > a.duration):
            with cam.lock:
                raw = cam.frame
            if raw is None or raw is last_id:                     # 새 화면이 올 때만 다시 그린다
                if cam.ended:
                    break
                if cv2.waitKey(3) & 0xFF == ord("q"):
                    break
                continue
            last_id, frame = raw, raw.copy()
            shown.append(time.monotonic()); shown = shown[-60:]
            fps = (len(shown) - 1) / (shown[-1] - shown[0]) if len(shown) > 1 and shown[-1] > shown[0] else 0.0
            r = judge.latest()
            fs = max(14, frame.shape[1] // 55)
            if r is None:
                vis = perf_panel(frame, ["성능 확인 · 판정 준비 중", f"화면 {fps:.0f} fps", f"장치: {dev_name}"], fs)
            else:
                now = time.monotonic() - judge.t0
                vis = draw(frame, det, r["persons"], r["inside"], tracker, now, 1, privacy=privacy)
                med = statistics.median(judge.times[r["model"]][-30:])
                lines = [
                    f"성능 확인 · 판정 {a.interval:g}초마다 + 화면 실시간",
                    f"화면 {fps:.0f} fps (카메라 그대로)",
                    f"마지막 판정 {r['ms']:.0f}ms{' (SAM 범위 갱신 포함)' if r['sam'] else ''} · 최근 중앙값 {med:.0f}ms",
                    f"박스·색은 {time.monotonic() - r['at']:.1f}초 전 판정 결과",
                    f"장치: {dev_name} · 모델: {r['model']}" + (" → 바꾸는 중" if det.model_name != r["model"] else ""),
                    f"이 컴퓨터로 {a.interval:g}초 간격 판정 가능한 룸: 약 {int(a.interval * 1000 // max(med, 1))}개",
                    f"모자이크 {'켜짐' if privacy else '꺼짐'} (m) · 모델 바꾸기 (t) · 종료 (q)",
                ]
                lines.insert(5, f"초록불 규칙: 자리 비움 {rule_text(a.limit_sec)}" + ("" if a.limit_sec == 1800 else " (시험용)"))
                vis = perf_panel(vis, lines, fs)
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                privacy = not privacy
            auto = a.switch_every and time.monotonic() - last_switch > a.switch_every
            if (key == ord("t") or auto) and len(names) > 1:     # 다음 판정부터 다른 모델로
                last_switch = time.monotonic()
                nxt = names[(names.index(det.model_name) + 1) % len(names)]
                det.yolo, det.model_name = yolos[nxt], nxt
                print(f"판정 모델을 {nxt}(으)로 바꿨습니다.")
    except KeyboardInterrupt:
        pass
    finally:
        judge.alive = False; cam.alive = False
        cv2.destroyAllWindows()
        if a.shot and vis is not None:
            cv2.imencode(".jpg", vis)[1].tofile(a.shot)
        if judge.times:
            print("\n==== 성능 확인 결과 ====")
            print(f"  장치: {dev_name}")
            gaps = [y - x for x, y in zip(judge.stamps, judge.stamps[1:])]
            if gaps:
                print(f"  실제 판정 간격 중앙값: {statistics.median(gaps):.2f}초 (목표 {a.interval:g}초)")
            for name, ts in judge.times.items():
                ts = ts[1:] or ts                                  # 모델마다 첫 판정은 준비 시간이라 뺀다
                med = statistics.median(ts)
                p95 = sorted(ts)[min(len(ts) - 1, int(len(ts) * 0.95))]
                print(f"  [{name}] 판정 {len(ts)}회 · 한 번 중앙값 {med:.0f}ms, p95 {p95:.0f}ms"
                      f" · {a.interval:g}초 간격으로 판정 가능한 룸 약 {int(a.interval * 1000 // max(med, 1))}개")


if __name__ == "__main__":
    main()
