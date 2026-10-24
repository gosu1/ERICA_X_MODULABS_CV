"""SAM 감지 범위 + YOLO26s v2 알박기 감지 실행.

    python mvp_sam/run_sam.py --source 영상.mp4 --time-scale 60 --save 결과.mp4
    python mvp_sam/run_sam.py --source 0            # 웹캠, 실제 30분 기준

화면에 그리는 것
  흰 선      SAM이 딴 테이블 테두리
  색 영역    부풀린 감지 범위 (신호등 색: 빨강 사용중 / 노랑 자리비움 / 초록 이용가능)
  흰 박스    감지 범위 안의 사람 (재석으로 셈), 회색 박스는 범위 밖 사람 (무시)
  하늘색 박스 감지 범위 안의 짐 (사람도 짐도 없으면 빈 책상 → 바로 초록)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "mvp"))
from room_logic import RoomState, RoomTracker, Rules  # noqa: E402
from sam_zone import ITEMS, SamZoneDetector  # noqa: E402

RGB = {"red": (230, 57, 70), "yellow": (250, 190, 40), "green": (46, 196, 102), "off": (70, 70, 70)}
FONT = next((p for p in ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"] if Path(p).exists()), None)


def rule_text(sec):
    """초 → '30분', '1분 30초' 같은 한국어 시간."""
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return " ".join(f"{v}{u}" for v, u in ((h, "시간"), (m, "분"), (s, "초")) if v) or "0초"


def font(size):
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def fmt(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def blur(img, b):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))
    if x2 - x1 > 4 and y2 - y1 > 4:
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.resize(cv2.resize(roi, (max(1, (x2 - x1) // 14), max(1, (y2 - y1) // 14))),
                                       (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def draw(frame, det, persons, inside, tr: RoomTracker, now, scale, privacy=True):
    out = frame.copy()
    if privacy:
        for b, _ in persons:
            blur(out, b)
    col = RGB[tr.state.light]; bgr = col[::-1]
    z = det.zone
    # 감지 범위: 반투명 채우기 + 테두리
    ov = out.copy(); ov[z.zone_mask] = bgr
    out = cv2.addWeighted(ov, 0.22, out, 0.78, 0)
    cs, _ = cv2.findContours(z.zone_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, bgr, 3)
    if z.source == "sam":
        ct, _ = cv2.findContours(z.table_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ct, -1, (255, 255, 255), 2)
    ins = {tuple(b) for b, _ in inside}
    for b, s in persons:
        c = (255, 255, 255) if tuple(b) in ins else (140, 140, 140)
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
        cv2.putText(out, f"{s:.2f}", (int(b[0]) + 3, int(b[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    for b, s, _ in det.items_inside:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (235, 200, 80), 2)

    h, w = out.shape[:2]; fs = max(16, w // 50)
    img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(img)
    r = fs; pw, ph = r * 2 + 24, r * 6 + 40; px, py = w - pw - 14, 14
    dr.rounded_rectangle((px, py, px + pw, py + ph), radius=14, fill=(25, 25, 25))
    for i, n in enumerate(["red", "yellow", "green"]):
        cx, cy = px + pw // 2, py + 12 + r + i * (2 * r + 8)
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=RGB[n] if tr.state.light == n else RGB["off"])
    src = {"sam": "SAM 테이블 테두리 기준", "box": "테이블 박스 기준(SAM 실패)", "temp": "임시 박스 (빈 룸이 되면 SAM으로 정함)", "full": "테이블 미검출 → 화면 전체"}[z.source]
    lines = [(f"{tr.room_id}  {tr.state.value}", font(fs), col),
             (f"감지 범위: {src} · 테두리 +{int(det.grow * 100)}% · 범위 갱신 {det.updates}회", font(int(fs * .7)), (255, 255, 255))]
    if tr.state is RoomState.AWAY:
        lines.append((f"자리 비움 {fmt(tr.away_sec(now) * scale)} · 초록불까지 {fmt(tr.left_sec(now) * scale)}", font(int(fs * .7)), (255, 255, 255)))
    elif tr.state is RoomState.OCCUPIED:
        lines.append((f"감지 범위 안 사람 {len(inside)}명 (범위 밖 {len(persons) - len(inside)}명은 무시)", font(int(fs * .7)), (255, 255, 255)))
    elif tr.empty:
        lines.append((f"빈 책상 (사람도 짐도 없음) · 바로 사용 가능", font(int(fs * .7)), (255, 255, 255)))
    else:
        lines.append((f"{rule_text(tr.rules.limit_sec * scale)} 이상 비어 있음 · 짐은 바구니로 옮기고 사용", font(int(fs * .7)), (255, 255, 255)))
    if det.items_inside:
        names = ", ".join(sorted({ITEMS[c] for _, _, c in det.items_inside}))
        lines.append((f"감지 범위 안 짐: {names}", font(int(fs * .7)), (255, 255, 255)))
    y = 12
    for t, f, c in lines:
        tb = dr.textbbox((14, y), t, font=f); dr.rectangle((tb[0] - 6, tb[1] - 4, tb[2] + 6, tb[3] + 4), fill=(0, 0, 0))
        dr.text((14, y), t, font=f, fill=c); y = tb[3] + 10
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="SAM 감지 범위 + YOLO26s v2 알박기 감지")
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=str(ROOT / "알박기_yolo26s_v2.pt"))
    ap.add_argument("--sam", default="sam2.1_t.pt", help="SAM 가중치 (처음 실행 시 자동으로 받음)")
    ap.add_argument("--grow", type=float, default=0.15, help="테이블 짧은 변의 몇 배만큼 부풀릴지")
    ap.add_argument("--overlap", type=float, default=0.10, help="사람 박스 중 감지 범위와 겹쳐야 하는 비율")
    ap.add_argument("--room", default="3층-1")
    ap.add_argument("--limit-min", type=float, default=30)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--start", type=float, default=0.0, help="영상 시작 위치(초)")
    ap.add_argument("--duration", type=float, default=0.0, help="처리할 길이(초), 0이면 끝까지")
    ap.add_argument("--save", default=None)
    ap.add_argument("--snap", default=None, help="상태가 바뀔 때 화면을 저장할 폴더")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-privacy", action="store_true")
    a = ap.parse_args()

    det = SamZoneDetector(a.weights, a.sam, grow=a.grow, overlap=a.overlap, move_sec=30.0 / a.time_scale * a.time_scale)
    tr = RoomTracker(a.room, Rules(limit_sec=a.limit_min * 60 / a.time_scale), start=0.0)
    cam = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source) if cam else a.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if not cam and a.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * src_fps))
    step = max(1, round(src_fps / a.fps))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"영상을 열 수 없습니다: {a.source}")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h)) if a.save else None
    snap = Path(a.snap) if a.snap else None
    if snap:
        snap.mkdir(parents=True, exist_ok=True)
    t0, idx, wall0 = time.monotonic(), 0, datetime.now()
    try:
        while ok:
            now = (time.monotonic() - t0) if cam else idx / src_fps
            if a.duration and now > a.duration:
                break
            persons, inside, _ = det.detect(frame, now)
            prev = tr.state; state = tr.update(bool(inside), now, items=bool(det.items_inside))
            vis = draw(frame, det, persons, inside, tr, now, a.time_scale, privacy=not a.no_privacy)
            if state is not prev or idx == 0:
                stamp = (wall0 + timedelta(seconds=now * a.time_scale)).strftime("%H:%M:%S")
                print(f"[{stamp}] {a.room}: {prev.value} → {state.value} (범위 안 {len(inside)}명, 짐 {len(det.items_inside)}개"
                      f"{', 빈 책상' if tr.empty else ''}, 범위 갱신 {det.updates}회)")
                if snap:
                    cv2.imencode(".jpg", vis)[1].tofile(str(snap / f"{idx:06d}_{state.value}.jpg"))
            if writer:
                writer.write(vis)
            if not a.no_show:
                cv2.imshow("study room (SAM zone)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if cam:
                ok, frame = cap.read()
            else:
                for _ in range(step - 1):
                    cap.grab()
                ok, frame = cap.read(); idx += step
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
