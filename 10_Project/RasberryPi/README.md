# Raspberry Pi 카메라 송출 및 성능 평가

라즈베리파이에서 USB 카메라 영상을 H.264/RTSP로 송출하고, 녹화 영상으로 YOLO26s 파인튜닝 모델의 CPU 처리 성능을 측정한 자료입니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| [pi_camera_rtsp.py](pi_camera_rtsp.py) | USB 카메라 영상 RTSP 송출 프로그램 |
| [pi_team_benchmark_v3.zip](pi_team_benchmark_v3.zip) | 평가 프로그램, 설정, 모델 및 실행 안내 |
| [evaluation_20261021_000734_412577.zip](evaluation_20261021_000734_412577.zip) | 평가 보고서와 원시 측정 기록 |

## 카메라 실행

스크립트 상단에 안내된 GStreamer 패키지 설치 후 실행합니다.

```bash
/usr/bin/python3 pi_camera_rtsp.py
```

수신 주소: `rtsp://<라즈베리파이_IP>:8554/camera`

## 성능 평가

- 장치: Raspberry Pi 4 Model B, CPU 3스레드
- 모델: YOLO26s v2, 입력 크기 960, JPEG 입력
- 영상: 960×540, 12 FPS, 약 22분
- 표본: 전체 영상 2초 간격 659장, 보조 평가 포함 총 767장
- 속도 통계는 전체 영상 표본 659장 기준이며 워밍업 1회는 제외했습니다.

| 항목 | 측정값 |
|---|---:|
| 평균 predict 처리시간 | 2,831.8 ms/장 |
| P95 처리시간 | 2,863.8 ms/장 |
| 환산 처리량 | 0.353 장/초 |
| 최대 프로세스 메모리 | 620.5 MiB |
| 최고 관측 온도 | 81.3°C |

평균 처리시간은 약 2.83초/장으로, 팀의 2초 표본 간격보다 길었습니다.

벤치마크 실행 방법은 프로그램 ZIP의 `README.md`, 상세 결과는 결과 ZIP의 `report.md` 또는 `report.html`을 참고하세요. 원본 `evaluate.py`와 정답 라벨이 없어 탐지 평가 기준 일부를 재구성했으며, 관련 점수는 임시 참조 기준의 결과입니다.
