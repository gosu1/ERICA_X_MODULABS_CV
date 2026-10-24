"""SAM으로 테이블 테두리를 따고, 바깥으로 부풀려 '감지 범위'를 만든다.

1단계(설치할 때 + 테이블 위치 이동이 감지됐을 때만):
  YOLO26s v2가 찾은 테이블 박스 → SAM에 박스로 넣어 테이블 마스크 → 팽창(dilation)으로 부풀림 → 감지 범위
2단계(매 프레임):
  YOLO26s v2로 사람 + 테이블 검출 → 사람 박스 면적의 일정 비율 이상이 감지 범위와 겹치면 재석 (방법 A)
  → 테이블이 저장된 위치와 다르면(IoU < 0.7) 그 상태가 일정 시간 이어지고 사람이 없을 때 1단계를 다시 한다.
  → 같은 검출에서 짐(가방·노트북·책 등)도 찾아, 감지 범위 안에 짐이 있는지(items_inside)를 남긴다.
     사람도 짐도 없으면 판정 쪽에서 '빈 책상'으로 보고 바로 초록으로 바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import SAM, YOLO

PERSON, TABLE = 0, 60
# 짐으로 보는 COCO 클래스. 가방류는 YOLO26s v2가 거의 못 잡아서(웹캠 0%) 신뢰도 기준을 따로 낮춘다.
ITEMS = {24: "가방", 25: "우산", 26: "핸드백", 28: "캐리어", 39: "물병", 41: "컵",
         63: "노트북", 64: "마우스", 66: "키보드", 67: "휴대폰", 73: "책"}
BAGS = {24, 26, 28}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Zone:
    table_box: tuple | None          # 저장된 테이블 박스 (x1, y1, x2, y2)
    table_mask: np.ndarray           # SAM 테이블 마스크 (bool, H×W)
    zone_mask: np.ndarray            # 부풀린 감지 범위 (bool, H×W)
    source: str                      # "sam" | "box" | "full"


@dataclass
class SamZoneDetector:
    weights: str
    sam_weights: str = "sam2.1_t.pt"
    imgsz: int = 960
    person_conf: float = 0.35
    table_conf: float = 0.25
    grow: float = 0.15               # 테이블 짧은 변 길이의 몇 배만큼 바깥으로 부풀릴지
    overlap: float = 0.10            # 사람 박스 면적 중 감지 범위와 겹쳐야 하는 비율 (룸 판정 실측 후 10%로 결정)
    move_iou: float = 0.7            # 저장된 테이블과 이보다 덜 겹치면 '위치가 다르다'
    move_sec: float = 30.0           # 위치가 다른 상태가 이만큼 이어지면 이동으로 봄
    item_conf: float = 0.25          # 짐 신뢰도 기준
    bag_conf: float = 0.10           # 가방류 신뢰도 기준 (놓치면 짐 있는 책상이 빈 책상으로 보이므로 낮게)
    item_overlap: float = 0.5        # 짐 박스 면적 중 감지 범위와 겹쳐야 하는 비율
    device: str | None = None
    zone: Zone | None = None
    updates: int = 0
    pending: bool = False            # 아직 빈 룸에서 SAM 영역을 못 정함
    _moved_since: float | None = None
    items: list = field(default_factory=list)          # 이번 프레임 화면 전체의 짐 [(박스, 신뢰도, 클래스)]
    items_inside: list = field(default_factory=list)   # 그중 감지 범위 안의 짐
    yolo: YOLO = field(init=False)
    sam: SAM = field(init=False)

    def __post_init__(self):
        self.yolo = YOLO(self.weights)
        self.sam = SAM(self.sam_weights)

    # ---------- 1단계: 감지 범위 정하기 ----------
    def build_zone(self, frame: np.ndarray, table_box, use_sam: bool = True) -> Zone:
        h, w = frame.shape[:2]
        if table_box is None:                                   # 테이블을 못 찾으면 화면 전체
            full = np.ones((h, w), bool)
            return Zone(None, full, full, "full")
        r = self.sam(frame, bboxes=[list(map(float, table_box))], verbose=False, device=self.device)[0] if use_sam else None
        if r is not None and r.masks is not None and len(r.masks.data):
            raw = r.masks.data[0].cpu().numpy().astype(np.uint8)
            # 가장 큰 조각만 남기고 볼록하게 감싸 구멍·조각을 없앤다 (테이블은 볼록한 모양)
            cs, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tmask = np.zeros((h, w), np.uint8)
            if cs:
                cv2.fillPoly(tmask, [cv2.convexHull(max(cs, key=cv2.contourArea))], 1)
            tmask = tmask.astype(bool)
            src = "sam"
        else:                                                   # SAM이 실패하면 박스로 대체
            tmask = np.zeros((h, w), bool)
            x1, y1, x2, y2 = map(int, table_box); tmask[y1:y2, x1:x2] = True
            src = "box" if use_sam else "temp"
        ys, xs = np.nonzero(tmask)
        short = min(xs.max() - xs.min(), ys.max() - ys.min()) if len(xs) else min(h, w) // 4
        k = max(3, int(short * self.grow) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        zmask = cv2.dilate(tmask.astype(np.uint8), kernel).astype(bool)
        return Zone(tuple(map(float, table_box)), tmask, zmask, src)

    # ---------- 2단계: 매 프레임 ----------
    def detect(self, frame: np.ndarray, now: float):
        r = self.yolo.predict(frame, imgsz=self.imgsz,
                              conf=min(self.person_conf, self.table_conf, self.item_conf, self.bag_conf),
                              classes=[PERSON, TABLE, *ITEMS], device=self.device, verbose=False)[0]
        persons, tables, items = [], [], []
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            c = int(c)
            if c == PERSON and s >= self.person_conf:
                persons.append((b, s))
            elif c == TABLE and s >= self.table_conf:
                tables.append((b, s))
            elif c in ITEMS and s >= (self.bag_conf if c in BAGS else self.item_conf):
                items.append((b, s, c))
        table = max(tables, key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))[0] if tables else None

        if self.zone is None:                                   # 설치할 때: 일단 박스로 임시 영역
            self.zone = self.build_zone(frame, table, use_sam=False); self.pending = True

        inside = [p for p in persons if self._overlap(p[0]) >= self.overlap]
        self.items = items
        self.items_inside = [it for it in items if self._overlap(it[0]) >= self.item_overlap]

        # 사람에 가려지지 않은 빈 룸이 되면 SAM으로 감지 범위를 정한다 (설치 직후 1번)
        if self.pending and table is not None and not persons:
            self.zone = self.build_zone(frame, table); self.updates += 1; self.pending = False

        # 테이블 위치 이동 감지 → 사람이 없을 때 1단계 다시
        moved = (table is not None and self.zone.table_box is not None and iou(table, self.zone.table_box) < self.move_iou) \
            or (table is not None and self.zone.table_box is None)
        if moved:
            self._moved_since = now if self._moved_since is None else self._moved_since
            if now - self._moved_since >= self.move_sec and not persons:
                self.zone = self.build_zone(frame, table); self.updates += 1; self._moved_since = None
        else:
            self._moved_since = None
        return persons, inside, table

    def _overlap(self, b) -> float:
        h, w = self.zone.zone_mask.shape
        x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))
        area = max(1, (x2 - x1) * (y2 - y1))
        return float(self.zone.zone_mask[y1:y2, x1:x2].sum()) / area
