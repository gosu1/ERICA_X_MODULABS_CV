# 03_모델 · 최종 판정 모델 (담당: 황예성)

| 파일 | 내용 |
|---|---|
| `알박기_yolo26s_v2.pt` | YOLO26s를 우리 스터디룸 데이터 727장(학습 637 + 검증 90)으로 파인튜닝한 최종 모델 (v2) |

- **성능** (학습에 안 쓴 영상 기준): 테이블 2% → 100%, 엎드린 사람 92.9% → 100%, 앉은 사람 100%. GPU(RTX 5060 노트북) 1장 18.1ms.
- **클래스 번호는 COCO 그대로**입니다(0 = 사람, 60 = 테이블, 가방·노트북 등 짐 클래스 포함). 기존 YOLO 코드에서 모델 파일만 바꿔 끼우면 됩니다.
- **천장 각도 전용**입니다. 천장에서 내려다본 사진으로 학습해서, 정면 웹캠 앞 사람은 14장 중 2장만 찾습니다. 정면 카메라에는 기본 `yolo26s.pt`를 쓰세요.
- 쓰는 곳: [06_통합실행](../06_통합실행)의 기본 판정 모델, [04_BASE_MODEL](../04_BASE_MODEL) 데모
- 학습 과정(데이터 제작, 1차 학습 실패와 v2, AdamW · lr0 0.001 · 25에포크): [2주차/1_모델비교·파인튜닝](https://github.com/hwang-ye-song/MODU_CV/tree/main/프로젝트_오픈스페이스_알박기감지/2주차/1_모델비교·파인튜닝)

```python
from ultralytics import YOLO

model = YOLO("알박기_yolo26s_v2.pt")
result = model.predict("frame.jpg", imgsz=960, conf=0.35)[0]
for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
    print(int(cls), box.tolist())   # 0 = 사람, 60 = 테이블
```
