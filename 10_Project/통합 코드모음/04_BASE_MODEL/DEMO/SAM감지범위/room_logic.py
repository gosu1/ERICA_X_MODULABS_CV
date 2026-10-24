"""스터디룸 상태 판정 (YOLO·OpenCV와 무관한 순수 파이썬).

매 프레임 '감지 영역 안에 사람이 있는가' 하나만 받아서 3색 상태를 정한다.

    🔴 사용중    OCCUPIED  사람이 있음
    🟡 자리비움  AWAY      사람이 없음, 30분 미만        → 짐에 손대지 않음
    🟢 이용가능  FREE      사람이 없음, 30분 이상        → 짐을 바구니로 옮기고 사용
                           또는 사람도 짐도 없는 빈 책상  → 노랑을 거치지 않고 바로 초록

- 사람이 다시 보이면 부재 타이머는 0으로 초기화된다.
- 잠깐 가려지거나 검출을 놓쳐도 상태가 흔들리지 않도록
  사람이 off_sec 동안 연속으로 안 보여야 '없음'으로 바꾼다 (부재 시각은 마지막으로 보인 순간).
- 시스템을 켠 직후 사람이 없으면, 언제부터 비었는지 모르므로 켠 시각부터 부재로 센다.
- 짐 정보(items)를 함께 넘기면, 사람이 없고 감지 영역 안에 짐도 empty_sec 동안 연속으로 없을 때
  '빈 책상'으로 보고 바로 초록으로 바꾼다. 짐이 한 번이라도 보이면 빈 책상 판정은 다시 처음부터 센다.
  items를 넘기지 않으면(None) 예전처럼 사람 유무와 30분 규칙만 본다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RoomState(str, Enum):
    OCCUPIED = "사용중"
    AWAY = "자리비움"
    FREE = "이용가능"

    @property
    def light(self) -> str:
        return {"사용중": "red", "자리비움": "yellow", "이용가능": "green"}[self.value]


@dataclass
class Rules:
    limit_sec: float = 30 * 60   # 이만큼 비면 초록 (학교 30분 규칙)
    on_sec: float = 2.0          # 사람이 이만큼 연속으로 보여야 '있음' (오검출 방지)
    off_sec: float = 10.0        # 사람이 이만큼 연속으로 안 보여야 '없음' (가림·놓침 방지)
    empty_sec: float = 10.0      # 사람이 없고 짐도 이만큼 연속으로 안 보여야 '빈 책상' (깜빡임 방지)


class RoomTracker:
    def __init__(self, room_id: str, rules: Rules, start: float = 0.0):
        self.room_id, self.rules = room_id, rules
        self.present = False
        self.last_seen: float | None = None     # 사람이 마지막으로 보인 시각
        self.away_since: float = start          # 부재 시작 시각
        self._seen_run_start: float | None = None
        self._no_items_since: float | None = None   # 짐이 안 보이기 시작한 시각
        self.empty = False                           # 사람도 짐도 없는 빈 책상인지
        self.state = RoomState.AWAY

    def update(self, person: bool, now: float, items: bool | None = None) -> RoomState:
        r = self.rules
        if items is None or items:
            self._no_items_since = None
        elif self._no_items_since is None:
            self._no_items_since = now
        if person:
            self.last_seen = now
            if self._seen_run_start is None:
                self._seen_run_start = now
            if not self.present and now - self._seen_run_start >= r.on_sec:
                self.present = True
        else:
            self._seen_run_start = None
            if self.present and self.last_seen is not None and now - self.last_seen >= r.off_sec:
                self.present = False
                self.away_since = self.last_seen

        self.empty = (not self.present and not person and self._no_items_since is not None
                      and now - self._no_items_since >= r.empty_sec)
        if self.present:
            self.state = RoomState.OCCUPIED
        elif self.empty:                                   # 빈 책상: 바로 초록
            self.state = RoomState.FREE
        elif self.away_sec(now) >= r.limit_sec:
            self.state = RoomState.FREE
        else:
            self.state = RoomState.AWAY
        return self.state

    def away_sec(self, now: float) -> float:
        return 0.0 if self.present else max(0.0, now - self.away_since)

    def left_sec(self, now: float) -> float | None:
        """초록불까지 남은 시간 (노랑일 때만)."""
        if self.state is not RoomState.AWAY:
            return None
        return max(0.0, self.rules.limit_sec - self.away_sec(now))
