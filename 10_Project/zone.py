"""첫 프레임 위에서 감지 영역(테이블과 앉는 자리)을 클릭해 다각형으로 저장한다.

왼쪽 클릭: 꼭짓점 추가 / 오른쪽 클릭: 마지막 꼭짓점 삭제 / Enter: 저장 / Esc: 취소
"""
import argparse
import json

import cv2
import numpy as np


def load_zone(path):
    return np.array(json.load(open(path))["points"], dtype=np.float32)


def in_zone_conf(persons, zone):
    """박스 중심이 감지 영역 안에 있는 사람 중 가장 높은 신뢰도를 돌려준다. 없으면 0."""
    return max((c for x1, y1, x2, y2, c in persons
                if cv2.pointPolygonTest(zone, ((x1 + x2) / 2, (y1 + y2) / 2), False) >= 0), default=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="스트림 URL, 영상 파일 또는 이미지")
    p.add_argument("--out", default="zone.json")
    args = p.parse_args()

    ok, frame = cv2.VideoCapture(args.source).read()
    if not ok:
        raise SystemExit(f"프레임을 읽지 못했습니다: {args.source}")
    s = min(1.0, 1280 / frame.shape[1])  # 화면보다 큰 프레임은 줄여서 보여 준다
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([round(x / s), round(y / s)])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow("zone")
    cv2.setMouseCallback("zone", on_mouse)
    while True:
        vis = frame.copy()
        if points:
            cv2.polylines(vis, [np.array(points)], True, (0, 255, 0), 3)
            for pt in points:
                cv2.circle(vis, pt, 8, (0, 255, 0), -1)
        cv2.imshow("zone", cv2.resize(vis, None, fx=s, fy=s))
        key = cv2.waitKey(30)
        if key == 13 and len(points) >= 3:
            json.dump({"points": points}, open(args.out, "w"))
            print(f"꼭짓점 {len(points)}개 저장: {args.out}")
            break
        if key == 27:
            print("저장하지 않고 종료합니다")
            break
    cv2.destroyAllWindows()
