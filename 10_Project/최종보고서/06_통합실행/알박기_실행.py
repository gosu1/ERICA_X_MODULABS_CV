"""알박기 감지 통합 실행: 카메라 여러 대 → 판정 → 신호등 · 기록 · 실시간 현황 화면.

    # 룸 설정 파일로 실행 (카메라 RTSP 주소나 영상 파일)
    python 알박기_실행.py --rooms rooms.json

    # 영상 하나를 룸 14개처럼 복제해 동시 처리 속도 시험 (병렬 테스트)
    python 알박기_실행.py --demo-video 1차.mp4 --demo-rooms 14 --duration 120

시작하면 초록불로 바꿀 자리 비움 시간을 시:분:초로 묻는다. 시험할 때 짧게 줄여 쓴다.
    1분 30초 → 00:01:30 또는 000130 · 그냥 Enter → 00:30:00 (학교 규칙) · --limit 00:01:30 으로 주면 묻지 않음

한 번 실행하면 아래가 모두 돌아간다.
  1. 룸마다 카메라(또는 영상)에서 가장 최근 화면만 받아 둔다 (밀린 화면은 버림).
  2. --interval 초마다 모든 룸을 차례로 판정한다.
     YOLO26s v2로 사람·테이블·짐 검출 → SAM 테이블 테두리 +15% 감지 범위 → 사람 박스 10% 이상 겹치면 재석
     → 2초 연속 보이면 사용중, 10초 안 보이면 부재, 입력한 시간(기본 30분)이면 초록, 사람도 짐도 없으면 바로 초록
  3. 색이 바뀌면 신호등 주소로 보내고, CSV 기록과 사람 모자이크 스냅샷(1일 보관)을 남긴다.
  4. http://localhost:8800 에서 MVP 관제 대시보드와 같은 모양의 현황 화면(현황/ 폴더)을 본다.
     룸 지도·KPI·알림·상태 기록·스냅샷은 1초마다, 선택한 룸의 카메라 화면은 실시간(모자이크)으로 갱신된다.
  5. 끝나면 한 바퀴(모든 룸 판정) 처리 시간 통계를 출력한다. 간격보다 짧아야 실시간으로 돌 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def first_existing(paths):
    return next((p for p in paths if p.exists()), paths[0])


# 판정 코드(room_logic·sam_zone·run_sam): 이 폴더에 같이 있으면 그것을, 없으면 팀 저장소 구조에서 찾는다
SAM_DIR = first_existing([HERE / "sam_zone.py", HERE.parent / "04_BASE_MODEL" / "DEMO" / "SAM감지범위" / "sam_zone.py"]).parent
sys.path.insert(0, str(SAM_DIR))
from room_logic import RoomState, RoomTracker, Rules  # noqa: E402
from run_sam import blur, draw, rule_text  # noqa: E402
from sam_zone import ITEMS, SamZoneDetector  # noqa: E402

WEB_DIR = HERE / "현황"                       # 현황 화면 (MVP 관제 대시보드 디자인)
CODE = {RoomState.OCCUPIED: 0, RoomState.AWAY: 1, RoomState.FREE: 2}

# 모델 파일: 이 폴더 → 팀 저장소(03_모델) → 개인 저장소(03_모델비교·파인튜닝) → 상위 폴더 순서로 찾는다
DEFAULT_WEIGHTS = first_existing([
    HERE / "알박기_yolo26s_v2.pt",
    HERE.parent / "03_모델" / "알박기_yolo26s_v2.pt",
    HERE.parent / "03_모델비교·파인튜닝" / "파인튜닝" / "모델" / "알박기_yolo26s_v2.pt",
    HERE.parent / "알박기_yolo26s_v2.pt",
])


def parse_hms(text):
    """'00:01:30' 또는 '000130' → 90초. 형식이 틀리면 ValueError."""
    t = text.strip()
    if len(t) == 6 and t.isdigit():
        t = f"{t[:2]}:{t[2:4]}:{t[4:]}"
    h, m, s = (int(x) for x in t.split(":"))
    if len(t.split(":")[0]) < 2 or h < 0 or not (0 <= m < 60 and 0 <= s < 60):
        raise ValueError(text)
    return h * 3600 + m * 60 + s


def ask_limit(given, off_sec=10.0):
    """초록불로 바꿀 자리 비움 시간을 시:분:초로 받는다. --limit 로 주면 묻지 않는다."""
    if given:
        try:
            sec = parse_hms(given)
        except ValueError:
            raise SystemExit(f"--limit 형식이 틀렸습니다: {given} · 예) 1분 30초 → 00:01:30 또는 000130")
    elif not sys.stdin.isatty():                       # 입력을 받을 수 없는 실행이면 학교 규칙대로
        sec = 30 * 60
    else:
        print("초록불로 바꿀 자리 비움 시간을 입력하세요. 형식은 시:분:초(00:00:00)입니다.")
        print("  예) 1분 30초 → 00:01:30 또는 000130 · 10초 → 00:00:10 · 그냥 Enter → 00:30:00 (학교 규칙)")
        while True:
            text = input("초록불 시간: ").strip()
            if not text:
                sec = 30 * 60
                break
            try:
                sec = parse_hms(text)
            except ValueError:
                print("  형식이 틀렸습니다. 00:01:30 또는 000130처럼 입력해 주세요.")
                continue
            if sec > 0:
                break
            print("  0초보다 길게 입력해 주세요.")
    if sec <= off_sec:
        print(f"  참고: {off_sec:g}초 이하로 하면 부재 확인({off_sec:g}초)이 끝나는 순간 이미 시간이 지나, 노랑 없이 빨강 → 초록으로 바로 바뀝니다.")
    print(f"초록불 규칙: 자리 비움 {rule_text(sec)}" + ("" if sec == 30 * 60 else " (시험용, 학교 규칙은 30분)"))
    return sec


# ---------------------------------------------------------------- 카메라
class LatestFrame(threading.Thread):
    """카메라(또는 영상)를 계속 읽어 가장 최근 화면 한 장만 들고 있는다.

    실시간 카메라는 받는 대로 덮어쓰고, 영상 파일은 원래 속도로 재생하듯 읽는다.
    판정이 느려도 밀린 화면을 쌓지 않으므로 지연이 누적되지 않는다.
    """

    def __init__(self, source, start_sec=0.0, loop=True):
        super().__init__(daemon=True)
        self.source, self.start_sec, self.loop = source, start_sec, loop
        self.is_file = isinstance(source, str) and Path(source).exists()
        self.frame = None
        self.lock = threading.Lock()
        self.alive = True
        self.ended = False
        self.cap = None

    def _open(self):
        src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if self.is_file and self.start_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000)
        return cap

    def run(self):
        self.cap = self._open()
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
        t0 = time.monotonic(); n = 0
        while self.alive:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop:
                    self.cap.release(); self.cap = self._open(); t0 = time.monotonic(); n = 0
                    continue
                if self.is_file:
                    self.ended = True
                    break
                time.sleep(0.5)                       # 실시간 카메라가 끊기면 다시 연결
                self.cap.release(); self.cap = self._open()
                continue
            with self.lock:
                self.frame = frame
            if self.is_file:                          # 영상 파일은 원래 속도로
                n += 1
                wait = t0 + n / fps - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
        self.cap.release()

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ---------------------------------------------------------------- 룸
@dataclass
class Room:
    id: str
    source: str
    signal_url: str | None
    camera: LatestFrame
    det: SamZoneDetector
    tracker: RoomTracker
    view: dict = field(default_factory=dict)     # 현황 화면용 마지막 판정 결과
    view_lock: threading.Lock = field(default_factory=threading.Lock)
    events: list = field(default_factory=list)   # 상태 변화 기록 (최근 것이 앞)
    runs: list = field(default_factory=list)     # 상태 기록 띠 [[시작 초, 상태 번호], ...]
    since_wall: str = ""                         # 지금 상태가 시작된 시각

    @property
    def group(self):                             # "3층-1" → 층 "3층", 번호 "1"
        return self.id.split("-")[0] if "-" in self.id else "룸"

    @property
    def no(self):
        return self.id.split("-", 1)[1] if "-" in self.id else self.id

    @property
    def kind(self):
        s = self.source.lower()
        return "RTSP 카메라" if s.startswith(("rtsp", "http")) else "웹캠" if s.isdigit() else "영상 파일"


class SharedModelDetector(SamZoneDetector):
    """룸마다 감지 범위는 따로 두고, YOLO·SAM 모델은 하나를 같이 쓴다 (메모리 절약)."""

    def __post_init__(self):
        pass


def send_signal(url, room_id, color):
    """신호등 색을 POST로 보낸다. 신호등 쪽 받는 방식이 정해지면 이 함수만 고친다."""
    if not url:
        return
    body = json.dumps({"room": room_id, "color": color}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=2)
    except OSError as e:
        print(f"  ! 신호등 전송 실패 ({room_id}, {url}): {e}")


def overlay(frame, det, persons, inside, color, pad=0.25):
    """실시간 화면용: 사람은 넓게 모자이크하고, 마지막 판정의 감지 범위·테이블·사람·짐을 MVP 색으로 겹친다.

    판정은 1초마다라 사람이 움직이면 박스가 늦게 따라오므로, 모자이크는 박스보다 25% 넓게 씌운다.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    for b, _ in persons:
        bw, bh = b[2] - b[0], b[3] - b[1]
        blur(out, (max(0, b[0] - bw * pad), max(0, b[1] - bh * pad), min(w, b[2] + bw * pad), min(h, b[3] + bh * pad)))
    z = det.zone
    if z is not None and z.zone_mask.shape == (h, w):
        zone_bgr = (208, 127, 47)                                 # MVP --zone (#2f7fd0)
        ov = out.copy(); ov[z.zone_mask] = zone_bgr
        out = cv2.addWeighted(ov, 0.18, out, 0.82, 0)
        cs, _ = cv2.findContours(z.zone_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, zone_bgr, 3)
        if z.source == "sam":
            ct, _ = cv2.findContours(z.table_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, ct, -1, (43, 138, 192), 2)        # MVP --table (#c08a2b)
    ins = {tuple(b) for b, _ in inside}
    for b, _ in persons:
        c = (59, 59, 209) if tuple(b) in ins else (150, 150, 150)  # 범위 안 빨강, 밖 회색
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
    for b, _, _ in det.items_inside:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (196, 92, 123), 2)  # MVP --items
    light = {"red": (59, 59, 209), "yellow": (18, 154, 217), "green": (87, 148, 30)}[color]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), light, 6)         # 테두리 = 지금 신호등 색
    return out


def save_snapshot(vis, folder: Path, name: str, keep_sec=24 * 3600):
    """상태가 바뀐 순간 화면(사람 모자이크 된 것)을 저장하고, 하루 지난 파일은 지운다."""
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", vis)[1].tofile(str(folder / name))
    for f in folder.glob("*.jpg"):
        if time.time() - f.stat().st_mtime > keep_sec:
            f.unlink()


# ---------------------------------------------------------------- 현황 화면
STATIC = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
NO_WEB = "<!doctype html><meta charset=utf-8><p>현황 화면 파일(현황/index.html)이 없습니다. 알박기_실행.py 옆에 현황 폴더를 같이 두세요."


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.unquote(self.path.split("?")[0])
            if path == "/status.json":
                return self._send(json.dumps(app.status(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
            if path.startswith("/stream/") and path.endswith(".mjpg"):
                return app.stream(self, path[len("/stream/"):-5])
            if path.startswith("/snap/"):                         # 상태 변화 스냅샷 (모자이크 된 것만 저장됨)
                f = app.out / "스냅샷" / Path(path[len("/snap/"):]).name
                if f.suffix == ".jpg" and f.exists():
                    return self._send(f.read_bytes(), "image/jpeg")
                return self.send_error(404)
            if path.startswith("/frame/") and path.endswith(".jpg"):
                jpg = app.frame_jpg(path[len("/frame/"):-4])
                if jpg is not None:
                    return self._send(jpg, "image/jpeg")
                return self.send_error(404)
            name = "index.html" if path in ("/", "") else Path(path).name
            f = WEB_DIR / name
            if f.suffix in STATIC and f.exists():
                return self._send(f.read_bytes(), STATIC[f.suffix])
            if name == "index.html":
                return self._send(NO_WEB.encode(), "text/html; charset=utf-8")
            self.send_error(404)
    return H


# ---------------------------------------------------------------- 실행
class App:
    def __init__(self, a):
        import torch
        from ultralytics import SAM, YOLO

        self.a = a
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.device == 0 else "CPU"
        print(f"모델 불러오는 중: {a.weights} (장치: {self.device_name})")
        yolo, sam = YOLO(a.weights), SAM(a.sam)
        blank = np.zeros((540, 960, 3), np.uint8)                 # 첫 판정이 늦지 않게 미리 한 번씩 돌려 둔다
        yolo.predict(blank, imgsz=960, device=self.device, verbose=False)
        sam(blank, bboxes=[[100.0, 100.0, 400.0, 300.0]], device=self.device, verbose=False)
        rules = Rules(limit_sec=a.limit_sec)
        self.rooms: list[Room] = []
        for i, rc in enumerate(load_rooms(a)):
            det = SharedModelDetector(str(a.weights), a.sam, overlap=a.overlap, grow=a.grow, device=self.device,
                                      move_sec=30.0)
            det.yolo, det.sam = yolo, sam
            cam = LatestFrame(str(rc["source"]), start_sec=float(rc.get("start_sec", 0)), loop=not a.no_loop)
            self.rooms.append(Room(rc["id"], str(rc["source"]), rc.get("signal_url") or a.signal_url, cam, det,
                                   RoomTracker(rc["id"], rules, start=0.0)))
        self.cycles: list[float] = []
        self.announced: set[str] = set()          # 첫 판정을 이미 알린 룸
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0, self.wall0 = time.monotonic(), time.time()        # 판정 시각 0초 = 프로그램을 켠 순간
        self.stopping = False

    def wall(self, sec):
        """판정 시각(켠 뒤 몇 초)을 실제 시계 시각 문자열로."""
        return datetime.fromtimestamp(self.wall0 + sec).strftime("%H:%M:%S")

    # 현황 화면
    def status(self):
        a, now = self.a, time.monotonic() - self.t0
        rooms = []
        for r in self.rooms:
            with r.view_lock:
                v = dict(r.view)
            tr = r.tracker
            judged = "now" in v
            code = CODE[tr.state] if judged else -1
            away = tr.away_sec(now) if judged else 0.0
            pend = 0.0                                             # 양수: 부재 확인 중(초), 음수: 재석 확인 중(초)
            if judged and tr.present and not v["inside"] and tr.last_seen is not None:
                pend = now - tr.last_seen
            elif judged and not tr.present and tr._seen_run_start is not None:
                pend = -(now - tr._seen_run_start)
            items = r.det.items_inside if judged else []
            rooms.append({
                "id": r.id, "no": r.no, "group": r.group, "kind": r.kind, "code": code,
                "inside": len(v.get("inside", [])), "persons": len(v.get("persons", [])),
                "items": len(items), "item_names": sorted({ITEMS[c] for _, _, c in items}),
                "empty": bool(tr.empty), "away": away, "limit": a.limit_sec,
                "left": max(0.0, tr.rules.limit_sec - away),
                "away_wall": self.wall(tr.away_since), "green_wall": self.wall(tr.away_since + tr.rules.limit_sec),
                "since": r.since_wall, "pend": round(pend, 1),
                "judged_ago": round(now - v["now"], 1) if judged else 0, "events": r.events[:20], "runs": r.runs,
            })
        recent = self.cycles[-10:]
        rules = self.rooms[0].tracker.rules if self.rooms else Rules()
        return {"rooms": rooms, "clock": datetime.now().strftime("%H:%M:%S"), "date": datetime.now().strftime("%Y-%m-%d"),
                "started": self.wall(0), "elapsed": now, "limit_sec": a.limit_sec, "limit_text": rule_text(a.limit_sec), "device": self.device_name,
                "cycle_ms": round(statistics.median(recent)) if recent else 0, "interval_ms": int(a.interval * 1000),
                "rules": {"on": rules.on_sec, "off": rules.off_sec, "empty": rules.empty_sec}}

    def stream(self, h: BaseHTTPRequestHandler, room_id, fps=15):
        """선택한 룸의 카메라 화면을 실시간(MJPEG)으로 보낸다. 박스·감지 범위는 마지막 판정 결과."""
        r = next((x for x in self.rooms if x.id == room_id), None)
        if r is None:
            return h.send_error(404)
        h.send_response(200)
        h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        try:
            while not self.stopping:
                t = time.monotonic()
                frame = r.camera.latest()
                with r.view_lock:
                    v = dict(r.view)
                if frame is None or "now" not in v:                # 첫 판정 전에는 사람 위치를 몰라 화면을 보내지 않는다
                    time.sleep(0.2)
                    continue
                vis = overlay(frame, r.det, v["persons"], v["inside"], v["color"])
                if vis.shape[1] > 960:
                    vis = cv2.resize(vis, (960, int(960 * vis.shape[0] / vis.shape[1])))
                jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                h.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                              + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(max(0.0, 1 / fps - (time.monotonic() - t)))
        except OSError:
            pass                                                   # 브라우저가 다른 룸을 고르거나 닫음

    def frame_jpg(self, room_id):
        r = next((x for x in self.rooms if x.id == room_id), None)
        if r is None:
            return None
        with r.view_lock:
            v = dict(r.view)
        if "frame" not in v:
            return None
        vis = draw(v["frame"], r.det, v["persons"], v["inside"], r.tracker, v["now"], 1, privacy=True)
        vis = cv2.resize(vis, (640, int(640 * vis.shape[0] / vis.shape[1])))
        return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

    def run(self):
        a = self.a
        for r in self.rooms:
            r.camera.start()
        server = None
        self.t0, self.wall0 = time.monotonic(), time.time()
        if not a.no_web:
            server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(self))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"실시간 현황 화면: http://localhost:{a.port}")
            if a.open:
                webbrowser.open(f"http://localhost:{a.port}")
        log_f = open(self.out / "판정기록.csv", "w", newline="", encoding="utf-8-sig")
        log = csv.writer(log_f)
        log.writerow(["시각", "룸", "상태", "신호등", "범위 안 사람", "범위 안 짐", "빈 책상", "비운 시간(초)", "한 바퀴(ms)"])

        print(f"룸 {len(self.rooms)}개, {a.interval}초마다 판정 시작 (Ctrl+C로 종료)")
        t0 = self.t0; next_t = t0
        try:
            while True:
                now = time.monotonic() - t0
                if a.duration and now > a.duration:
                    break
                if all(r.camera.ended for r in self.rooms):
                    break
                c0 = time.perf_counter()
                for r in self.rooms:
                    frame = r.camera.latest()
                    if frame is None:
                        continue
                    persons, inside, _ = r.det.detect(frame, now)
                    prev = r.tracker.state
                    state = r.tracker.update(bool(inside), now, items=bool(r.det.items_inside))
                    color = state.light
                    detail = self.detail(r, now, inside)
                    with r.view_lock:
                        r.view = {"state": state.value, "color": color, "detail": detail, "frame": frame,
                                  "persons": persons, "inside": inside, "now": now}
                    if state is not prev or r.id not in self.announced:
                        stamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{stamp}] {r.id}: {prev.value} → {state.value} ({detail})")
                        send_signal(r.signal_url, r.id, color)
                        snap = None
                        if not a.no_snapshot:
                            vis = draw(frame, r.det, persons, inside, r.tracker, now, 1, privacy=True)
                            snap = f"{datetime.now():%Y%m%d-%H%M%S}_{r.id}_{state.value}.jpg"
                            save_snapshot(vis, self.out / "스냅샷", snap)
                        first = r.id not in self.announced
                        r.events.insert(0, {"t": stamp, "ts": round(now, 1), "from": -1 if first else CODE[prev],
                                            "to": CODE[state], "why": self.why(r, state), "snap": snap,
                                            "empty": bool(r.tracker.empty), "away_wall": self.wall(r.tracker.away_since)})
                        del r.events[50:]
                        r.runs.append([round(now, 1), CODE[state]])
                        r.since_wall = stamp
                        self.announced.add(r.id)
                    log.writerow([f"{now:.1f}", r.id, state.value, color, len(inside), len(r.det.items_inside),
                                  int(r.tracker.empty), f"{r.tracker.away_sec(now):.0f}", ""])
                cyc = (time.perf_counter() - c0) * 1000
                self.cycles.append(cyc)
                log.writerow([f"{now:.1f}", "(한 바퀴)", "", "", "", "", "", "", f"{cyc:.0f}"])
                next_t += a.interval
                wait = next_t - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_t = time.monotonic()                     # 밀렸으면 기준 시각을 다시 맞춘다
        except KeyboardInterrupt:
            pass
        finally:
            self.stopping = True
            for r in self.rooms:
                r.camera.alive = False
            log_f.close()
            if server:
                server.shutdown()
            self.report()

    def why(self, r: Room, state):
        """상태가 바뀐 이유 (현황 화면 스냅샷 목록에 표시)."""
        tr = r.tracker
        if state is RoomState.OCCUPIED:
            return f"사람이 {tr.rules.on_sec:g}초 연속 보여 재석으로 판정"
        if state is RoomState.AWAY:
            return f"마지막으로 보인 {self.wall(tr.away_since)}부터 부재 ({tr.rules.off_sec:g}초 확인 후 판정)"
        if tr.empty:
            return f"사람도 짐도 {tr.rules.empty_sec:g}초 연속 없어 빈 책상으로 판정 (바로 초록)"
        return f"{self.wall(tr.away_since)}부터 {rule_text(tr.rules.limit_sec)} 경과"

    def detail(self, r: Room, now, inside):
        tr = r.tracker
        if tr.state is RoomState.OCCUPIED:
            return f"범위 안 사람 {len(inside)}명"
        if tr.empty:
            return "빈 책상 · 바로 사용 가능"
        if tr.state is RoomState.AWAY:
            return f"짐 {len(r.det.items_inside)}개 · 초록까지 {rule_text(tr.left_sec(now))}"
        return f"{rule_text(tr.rules.limit_sec)} 넘게 비어 있음 · 짐은 바구니로"

    def report(self):
        if not self.cycles:
            return
        cs = sorted(self.cycles[1:] or self.cycles)              # 첫 바퀴는 모델 준비 시간이라 뺀다
        p95 = cs[min(len(cs) - 1, int(len(cs) * 0.95))]
        n = len(self.rooms)
        res = {"룸 수": n, "판정 간격(ms)": int(self.a.interval * 1000), "바퀴 수": len(cs),
               "한 바퀴 중앙값(ms)": round(statistics.median(cs)), "한 바퀴 p95(ms)": round(p95), "한 바퀴 최대(ms)": round(cs[-1]),
               "룸 1개당 중앙값(ms)": round(statistics.median(cs) / n, 1),
               "실시간 가능": bool(p95 < self.a.interval * 1000)}
        print("\n==== 동시 처리 결과 ====")
        for k, v in res.items():
            print(f"  {k}: {v}")
        (self.out / "동시처리_결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"기록: {self.out / '판정기록.csv'}, {self.out / '동시처리_결과.json'}")


def load_rooms(a):
    if a.demo_video:
        n, length = a.demo_rooms, video_seconds(a.demo_video)
        # 같은 영상을 룸마다 다른 위치에서 시작해, 룸 상태가 제각각이 되게 한다
        return [{"id": f"데모-{i + 1:02d}", "source": a.demo_video, "start_sec": (length * i / n) if length else 0}
                for i in range(n)]
    cfg = json.loads(Path(a.rooms).read_text(encoding="utf-8"))
    return cfg["rooms"]


def video_seconds(path):
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 1
    cap.release()
    return n / fps if n > 0 else 0


def main():
    ap = argparse.ArgumentParser(description="알박기 감지 통합 실행 (카메라 여러 대 → 판정 → 신호등·기록·현황 화면)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rooms", help="룸 설정 JSON (rooms.example.json 참고)")
    g.add_argument("--demo-video", help="영상 하나를 여러 룸처럼 복제해 시험")
    ap.add_argument("--demo-rooms", type=int, default=14, help="--demo-video 로 만들 룸 수")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--sam", default="sam2.1_t.pt", help="SAM 가중치 (처음 실행 시 자동으로 받음)")
    ap.add_argument("--interval", type=float, default=1.0, help="모든 룸을 한 번씩 판정하는 간격(초)")
    ap.add_argument("--overlap", type=float, default=0.10, help="사람 박스 중 감지 범위와 겹쳐야 하는 비율")
    ap.add_argument("--grow", type=float, default=0.15, help="테이블 짧은 변의 몇 배만큼 부풀릴지")
    ap.add_argument("--limit", help="초록불로 바꾸는 자리 비움 시간 (00:01:30 또는 000130). 안 주면 시작할 때 물어봄")
    ap.add_argument("--signal-url", help="신호등 색을 POST로 보낼 주소 (룸별 signal_url이 없을 때)")
    ap.add_argument("--duration", type=float, default=0, help="이만큼(초) 돌고 끝냄. 0이면 계속")
    ap.add_argument("--no-loop", action="store_true", help="영상 파일이 끝나면 다시 처음부터 돌리지 않음")
    ap.add_argument("--out", default=str(HERE / "결과"), help="기록·스냅샷 저장 폴더")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-web", action="store_true", help="실시간 현황 화면을 띄우지 않음")
    ap.add_argument("--no-snapshot", action="store_true")
    ap.add_argument("--open", action="store_true", help="현황 화면이 준비되면 브라우저로 열기")
    a = ap.parse_args()
    if not Path(a.weights).exists():
        raise SystemExit(f"모델 파일을 찾을 수 없습니다: {a.weights} · --weights 로 알박기_yolo26s_v2.pt 위치를 알려 주세요.")
    if a.rooms and not Path(a.rooms).exists():
        raise SystemExit(f"룸 설정 파일이 없습니다: {a.rooms} · rooms.example.json을 복사해 rooms.json을 만들고 카메라 주소를 적어 주세요.")
    if a.demo_video and not Path(a.demo_video).exists():
        raise SystemExit(f"영상 파일이 없습니다: {a.demo_video}")
    a.limit_sec = ask_limit(a.limit)
    App(a).run()


if __name__ == "__main__":
    main()
