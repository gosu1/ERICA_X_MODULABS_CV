"""알박기 감지 통합 실행: 카메라 여러 대 → 판정 → 신호등 · 기록 · 실시간 현황 화면.

    # 룸 설정 파일로 실행 (카메라 RTSP 주소나 영상 파일)
    python 알박기_실행.py --rooms rooms.json

    # 영상 하나를 룸 14개처럼 복제해 동시 처리 속도 시험 (병렬 테스트)
    python 알박기_실행.py --demo-video 1차.mp4 --demo-rooms 14 --duration 120 --time-scale 60

한 번 실행하면 아래가 모두 돌아간다.
  1. 룸마다 카메라(또는 영상)에서 가장 최근 화면만 받아 둔다 (밀린 화면은 버림).
  2. --interval 초마다 모든 룸을 차례로 판정한다.
     YOLO26s v2로 사람·테이블·짐 검출 → SAM 테이블 테두리 +15% 감지 범위 → 사람 박스 10% 이상 겹치면 재석
     → 2초 연속 보이면 사용중, 10초 안 보이면 부재, 30분이면 초록, 사람도 짐도 없으면 바로 초록
  3. 색이 바뀌면 신호등 주소로 보내고, CSV 기록과 사람 모자이크 스냅샷(1일 보관)을 남긴다.
  4. http://localhost:8800 에서 14개 룸 현황과 모자이크 카메라 화면을 실시간으로 본다.
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
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SAM_DIR = HERE.parent / "05_BASE_MODEL" / "DEMO" / "SAM감지범위"
sys.path.insert(0, str(SAM_DIR))
from room_logic import RoomState, RoomTracker, Rules  # noqa: E402
from run_sam import draw  # noqa: E402
from sam_zone import SamZoneDetector  # noqa: E402

DEFAULT_WEIGHTS = HERE.parent / "04_모델" / "알박기_yolo26s_v2.pt"


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


def save_snapshot(vis, folder: Path, name: str, keep_sec=24 * 3600):
    """상태가 바뀐 순간 화면(사람 모자이크 된 것)을 저장하고, 하루 지난 파일은 지운다."""
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", vis)[1].tofile(str(folder / name))
    for f in folder.glob("*.jpg"):
        if time.time() - f.stat().st_mtime > keep_sec:
            f.unlink()


# ---------------------------------------------------------------- 현황 화면
PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>알박기 감지 · 실시간 현황</title>
<style>
body{margin:0;font-family:"Malgun Gothic",sans-serif;background:#f3f2ec;color:#1d2a2c}
header{background:#16343a;color:#eef3ef;padding:14px 20px;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
header b{font-size:20px} header span{opacity:.8;font-size:14px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:16px}
.c{background:#fbfbf7;border:1px solid #d6d8cf;border-radius:12px;overflow:hidden;border-top:8px solid #888}
.c.red{border-top-color:#d9434f}.c.yellow{border-top-color:#e0a400}.c.green{border-top-color:#1f9d55}
.c img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#222}
.c div{padding:8px 12px;font-size:14px} .c h3{margin:0;font-size:17px}
</style>
<header><b>알박기 감지 · 실시간 현황</b><span id="sum"></span><span id="perf"></span></header>
<main id="g"></main>
<script>
async function tick(){
  const s=await (await fetch('status.json',{cache:'no-store'})).json();
  document.getElementById('sum').textContent=`사용중 ${s.count.red} · 자리비움 ${s.count.yellow} · 이용가능 ${s.count.green} · 활성화율 ${s.active_rate}%`;
  document.getElementById('perf').textContent=`한 바퀴 ${s.cycle_ms}ms / 간격 ${s.interval_ms}ms`;
  const g=document.getElementById('g');
  g.innerHTML=s.rooms.map(r=>`<section class="c ${r.color}"><img src="frame/${encodeURIComponent(r.id)}.jpg?t=${Date.now()}" alt="${r.id} 카메라 화면 (사람 모자이크)">
  <div><h3>${r.id} · ${r.state}</h3>${r.detail}</div></section>`).join('');
}
tick(); setInterval(tick,2000);
</script></html>"""


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
            if path in ("/", "/index.html"):
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if path == "/status.json":
                return self._send(json.dumps(app.status(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
            if path.startswith("/frame/") and path.endswith(".jpg"):
                jpg = app.frame_jpg(path[len("/frame/"):-4])
                if jpg is not None:
                    return self._send(jpg, "image/jpeg")
            self.send_error(404)
    return H


# ---------------------------------------------------------------- 실행
class App:
    def __init__(self, a):
        import torch
        from ultralytics import SAM, YOLO

        self.a = a
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"모델 불러오는 중: {a.weights} (장치: {'GPU' if self.device == 0 else 'CPU'})")
        yolo, sam = YOLO(a.weights), SAM(a.sam)
        blank = np.zeros((540, 960, 3), np.uint8)                 # 첫 판정이 늦지 않게 미리 한 번씩 돌려 둔다
        yolo.predict(blank, imgsz=960, device=self.device, verbose=False)
        sam(blank, bboxes=[[100.0, 100.0, 400.0, 300.0]], device=self.device, verbose=False)
        rules = Rules(limit_sec=a.limit_min * 60 / a.time_scale)
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

    # 현황 화면
    def status(self):
        rooms, cnt = [], {"red": 0, "yellow": 0, "green": 0}
        for r in self.rooms:
            with r.view_lock:
                v = dict(r.view)
            color = v.get("color", "yellow")
            cnt[color] = cnt.get(color, 0) + 1
            rooms.append({"id": r.id, "state": v.get("state", "준비 중"), "color": color, "detail": v.get("detail", "")})
        recent = self.cycles[-10:]
        return {"rooms": rooms, "count": cnt, "active_rate": round(100 * cnt["red"] / max(1, len(self.rooms))),
                "cycle_ms": round(statistics.median(recent)) if recent else 0, "interval_ms": int(self.a.interval * 1000)}

    def frame_jpg(self, room_id):
        r = next((x for x in self.rooms if x.id == room_id), None)
        if r is None:
            return None
        with r.view_lock:
            v = dict(r.view)
        if "frame" not in v:
            return None
        vis = draw(v["frame"], r.det, v["persons"], v["inside"], r.tracker, v["now"], self.a.time_scale, privacy=True)
        vis = cv2.resize(vis, (640, int(640 * vis.shape[0] / vis.shape[1])))
        return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

    def run(self):
        a = self.a
        for r in self.rooms:
            r.camera.start()
        server = None
        if not a.no_web:
            server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(self))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"실시간 현황 화면: http://localhost:{a.port}")
        log_f = open(self.out / "판정기록.csv", "w", newline="", encoding="utf-8-sig")
        log = csv.writer(log_f)
        log.writerow(["시각", "룸", "상태", "신호등", "범위 안 사람", "범위 안 짐", "빈 책상", "비운 시간(초)", "한 바퀴(ms)"])

        print(f"룸 {len(self.rooms)}개, {a.interval}초마다 판정 시작 (Ctrl+C로 종료)")
        t0 = time.monotonic(); next_t = t0
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
                        if not a.no_snapshot:
                            vis = draw(frame, r.det, persons, inside, r.tracker, now, a.time_scale, privacy=True)
                            save_snapshot(vis, self.out / "스냅샷", f"{datetime.now():%Y%m%d-%H%M%S}_{r.id}_{state.value}.jpg")
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
            for r in self.rooms:
                r.camera.alive = False
            log_f.close()
            if server:
                server.shutdown()
            self.report()

    def detail(self, r: Room, now, inside):
        tr = r.tracker
        if tr.state is RoomState.OCCUPIED:
            return f"범위 안 사람 {len(inside)}명"
        if tr.empty:
            return "빈 책상 · 바로 사용 가능"
        if tr.state is RoomState.AWAY:
            left = tr.left_sec(now) * self.a.time_scale
            return f"짐 {len(r.det.items_inside)}개 · 초록까지 {int(left // 60)}분 {int(left % 60)}초"
        return "30분 넘게 비어 있음 · 짐은 바구니로"

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
    ap.add_argument("--limit-min", type=float, default=30, help="초록불로 바꾸는 자리 비움 시간(분)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="시연용으로 30분을 줄이는 배수 (60이면 30초)")
    ap.add_argument("--signal-url", help="신호등 색을 POST로 보낼 주소 (룸별 signal_url이 없을 때)")
    ap.add_argument("--duration", type=float, default=0, help="이만큼(초) 돌고 끝냄. 0이면 계속")
    ap.add_argument("--no-loop", action="store_true", help="영상 파일이 끝나면 다시 처음부터 돌리지 않음")
    ap.add_argument("--out", default=str(HERE / "결과"), help="기록·스냅샷 저장 폴더")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-web", action="store_true", help="실시간 현황 화면을 띄우지 않음")
    ap.add_argument("--no-snapshot", action="store_true")
    a = ap.parse_args()
    App(a).run()


if __name__ == "__main__":
    main()
