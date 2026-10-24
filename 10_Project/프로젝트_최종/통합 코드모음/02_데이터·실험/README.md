# 02_데이터·실험 · 실험 기록, 데이터셋 조사, 생성형 데이터 실험 (담당: 박서준, 엎드린 사람 실험: 황예성)

| 파일·폴더 | 내용 |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | 실험 기록: ① 모델 비교(녹화 영상 36장, 엎드린 사람 사진 6장, 1차 룸 영상 659장) ② `server.py` 검증 ③ 추론 위치 비교 실험 설계 ④ 생성형 데이터 실험 ⑤ 파라미터 대비 성능(15종 전체) |
| [DATASETS.md](DATASETS.md) | 우리 방 카메라 조건에 맞는 공개 데이터셋 10종 조사와 결론 (Edinburgh Office Monitoring, Employee Sleeping, SCB-Dataset 등) |
| [synthetic](synthetic) | Stable Diffusion 인페인팅으로 빈 방 사진에 엎드린 사람을 그려 넣는 실험 |
| [image](image) | 실험 이미지와 결과 그래프 (실제 사람은 모자이크, 07~09는 Stable Diffusion이 그린 가상 인물). 설명은 [image/README.md](image/README.md) |

## 생성형 데이터 실험 (synthetic)

| 파일 | 하는 일 |
|---|---|
| `generate.py` | 빈 방 프레임의 자리마다 엎드린 사람을 그려 넣습니다. 자리 주변 512×512만 잘라 그리고, 마스크 안쪽만 원본에 다시 붙입니다 (`--hard`: 외투·후드·담요로 가린 어려운 장면) |
| `filter.py` | 그려 넣은 자리에 사람이 실제로 그려졌는지 CLIP으로 1차로 거릅니다. 남긴 이미지도 마지막에 사람이 눈으로 확인합니다 |
| `seats.json` | 자리 4곳(A·B·C·D)의 박스 좌표 |
| `base_0800.png` | 사람이 없는 빈 방 프레임 (1차 영상 8:00) |
| `eval15.csv` | 생성 이미지에 모델 15종을 돌린 결과 |
| `out_pilot/`, `out/`, `out_hard/` | 시험·2차·어려운 장면 회차의 판정 기록(`labels.json`). 생성 이미지 예시는 `image/07~09`에 있습니다 |

```bash
pip install diffusers transformers torch ultralytics
cd synthetic
python generate.py base_0800.png --seats seats.json --out out --device cuda   # 맥은 --device mps
python filter.py out
```

- 결과: 3회에 걸쳐 80장을 만들었고 그중 엎드린 장면은 40장입니다. 기존 모델이 40장을 모두 잡아 학습에는 쓰지 않았고, 약점 찾기에 썼습니다. 후드를 쓰고 엎드린 사람은 YOLO26l이 0.14로 놓쳤고, 의자에 외투만 걸린 사진은 어떤 모델도 사람으로 잡지 않았습니다.
- 1장 만드는 데 약 70초가 걸렸습니다(맥 MPS, 25~30스텝).

## 함께 쓴 부분

- **EXPERIMENTS.md 1절의 "엎드린 사람 사진 6장" 결과는 황예성의 실험**입니다. 1차 회의 중 직접 찍은 사진 6장으로 모델 6종을 돌리고 파라미터·연산량·속도까지 잰 것으로, 코드와 노트북은 [1주차/YOLO검증](https://github.com/hwang-ye-song/MODU_CV/tree/main/프로젝트_오픈스페이스_알박기감지/1주차/YOLO검증)에 있습니다.
- **모델 비교는 두 사람이 따로 했습니다.** 박서준은 15종 비교(5절)로 YOLO26s와 임계값 0.30을 기본으로 정했고, 황예성은 16종 비교(테이블·엎드린 사람 기준, [2주차/1_모델비교·파인튜닝](https://github.com/hwang-ye-song/MODU_CV/tree/main/프로젝트_오픈스페이스_알박기감지/2주차/1_모델비교·파인튜닝))로 파인튜닝이 필요하다는 것을 확인했습니다. 최종 모델은 YOLO26s를 파인튜닝한 v2입니다 → [03_모델](../03_모델)

> 이 README는 저장소를 정리하면서 코드의 설명(docstring, 옵션)과 팀 실험 기록을 모아 추가했습니다.
