> 한 줄: (직접 씁니다)

# 공개 데이터셋 조사

조사일은 2026-10-19다. 기준은 우리 방 카메라다. C270 한 대가 방 모서리 높은 곳에서 30~45도로 테이블 1개와 의자 4개를 내려다본다. 모든 내용은 데이터셋 공식 페이지, 논문, 저장소, 라이선스 문서에서 확인했고, 확인하지 못한 것은 "확인 필요"로 적었다. 데이터 파일은 하나도 내려받지 않았다.

## 결론

이 조건에 가장 가까운 공개 데이터는 Edinburgh Office Monitoring이다. 사무실 위쪽 고정 카메라로 20일을 1초 1장 간격으로 찍었고, 빈 방과 앉은 사람에 프레임별 라벨이 있다. 그래서 높은 실내 각도와 사람 없는 방의 오검출을 한 데이터로 볼 수 있다. 엎드린 사람 재현율은 Roboflow의 Employee Sleeping(사무실 CCTV 각도, 책상에 엎드린 사람 3,042장)과 SCB-Dataset3-U의 leaning over the table 라벨이 가장 가깝다. 다만 Employee Sleeping은 이미지 출처 설명이 없고, SCB는 허락 없는 복사와 배포를 금지한다. 의자 옆에 가방만 남는 장면은 CAVIAR의 LeftBag 클립에 있지만, 로비를 거의 바로 위에서 찍은 저해상도 영상이다. 스터디룸 같은 작은 방에서 책상에 엎드린 사람을 방 모서리 각도로 찍은 공개 데이터는 찾지 못했다.

## 비교표

| 데이터셋 | 장면 | 카메라 각도 | 규모 | 라벨 | 라이선스 | 적합도 |
|---|---|---|---|---|---|---|
| Edinburgh Office Monitoring | 대학 사무실 4곳, 혼자 일하는 사람과 빈 방 | 방 위쪽 고정 카메라 (논문 샘플 기준) | 20일, 456,714장, 약 1 FPS | 프레임별 사람 박스와 상태 (빈 방, 서기, 앉기, 대화, 쓰러짐) | CC BY-NC-SA 3.0 | 높음 |
| Employee Sleeping (Roboflow) | 사무실 책상에 엎드려 자는 사람 | 사무실 위쪽 CCTV 각도 (미리보기 기준) | 이미지 3,042장 | 엎드린 사람 박스 1종 | 페이지 표기 CC BY 4.0, 출처 설명 없음 | 중간 |
| SCB-Dataset (3-U 포함) | 유치원부터 대학까지 교실 | 교실 정면, 측면, 후면 | 본편 13,330장, 3-U 671장 | 사람별 행동 박스 (bow the head, leaning over the table 등) | 저작권자 권리 보유, 비상업 학술용 | 중간 |
| CAVIAR (INRIA 로비) | 연구소 로비, 가방 두고 가기, 의자에서 쉬기 | 광각, 거의 바로 위 | 28개 클립, 384x288, 25 FPS | 사람 박스, 상황 라벨 (XML) | CC BY-SA | 중간 |
| PIROPO | 실내 방 2곳, 걷기, 서기, 앉기 | 벽과 천장 모서리의 일반 카메라, 천장 어안 카메라 | 주석 프레임 10만 장 이상, 약 10 FPS | 머리 중심 점 | 연구 목적 무료 | 중간 |
| Toyota Smarthome | 아파트 식당, 거실, 부엌의 노인 일상 | 방 모서리 위쪽 Kinect 7대 (논문 샘플 기준) | 16,115클립, 긴 영상 536개 | 활동 구간 (Enter, Leave, Sit down, Read, Use laptop 등), 관절 자세 | 비상업 연구와 교육, 재배포 금지 | 중간 |
| CEPDOF | 작은 강의실, 점심 모임, 모자와 후드 | 천장 어안 카메라, 수직 시점 | 영상 8개, 25,504장 | 회전 박스와 사람 ID | 비상업 또는 ODbL (페이지마다 다름) | 낮음 |
| EduNet | 인도 초중고 교실, YouTube 수업 영상 | 여러 각도 | 7,851클립, 약 12시간 | 클립 단위 행동 20종 (Sleeping 포함) | 표기 없음 | 낮음 |
| Classroom Behavior (Roboflow) | 교실 학생 | 교실 뒤쪽 위와 눈높이가 섞임 (미리보기 기준) | 이미지 4,881장 | hand, read, sleep, write 박스 | 페이지 표기 CC BY 4.0, 출처 설명 없음 | 낮음 |
| ABODA | 물건이 버려진 실내외 장면 | 확인 필요 | 영상 11개 | 저장소에 정답 파일 없음 | 표기 없음 | 낮음 |

## 데이터셋별 정리

### 1. Edinburgh Office Monitoring Video Dataset

- 공식 페이지: https://homepages.inf.ed.ac.uk/rbf/OFFICEDATA/
- 논문: T. Qasim, R. B. Fisher, N. Bhatti, "Ground-truthing Large Human Behavior Monitoring Datasets", ICPR 2020. http://homepages.inf.ed.ac.uk/rbf/PAPERS/TQICPR20.pdf
- 장면: 에든버러 대학 정보학부 사무실 4곳을 2016년 2월부터 4월까지 20일 동안 찍었다. 1번 사무실이 12일, 나머지 3곳이 2~3일씩이다. 대부분 한 사람이 책상에서 일하거나 방이 비어 있고, 가끔 최대 3명이 회의한다. 페이지는 찍힌 사람 모두 촬영에 동의했다고 쓴다.
- 카메라: 고정 카메라 1대다. 페이지와 논문 모두 설치 높이를 적지 않았다. 논문 Fig. 2의 샘플 프레임은 방 위쪽에서 책상과 의자를 비스듬히 내려다보는 화면이고, 논문은 이 영상을 사무실 CCTV 영상이라고 부른다. 조명 변화가 크고 처음 12일은 색 조명이 섞여 있다.
- 형태와 규모: 1280x720 컬러 이미지를 평균 약 1 FPS로 저장했다. 총 456,714장이다(논문은 456,715장). 1초 1장으로 치면 약 127시간이다(계산값). 하루치 AVI와 TAR가 각각 약 1.2GB다(1일차 기준).
- 라벨: 프레임마다 최대 3명의 박스(x, y, 너비, 높이)와 상태 번호가 있다. 0 빈 방, 1 서 있음 또는 걷기, 2 앉음, 3 두세 명 대화, 4 쓰러짐이다. 잠이나 엎드림 라벨은 없다. 페이지는 빈 방 134,110장, 1명 249,956장이라고 쓰고, 논문 Table 1은 같은 두 숫자를 반대로 적었다. 어느 쪽이 맞는지는 확인 필요다.
- 라이선스: CC BY-NC-SA 3.0이다(https://creativecommons.org/licenses/by-nc-sa/3.0/). 비상업 수업 과제에 쓸 수 있다. 공개해서 쓸 때는 페이지에 있는 감사 문구를 그대로 넣고 위 논문을 인용하라고 한다.
- 재배포: 라이선스상 같은 조건으로는 가능하다. 그래도 사람이 나오는 영상이므로 우리 저장소에는 올리지 않는다.
- 받는 방법: 페이지에서 날짜별 AVI, TAR, 정답 MAT 파일을 바로 받는다. 신청 절차는 없다. 정답은 MATLAB .mat 형식이고 읽는 예시 코드 traverse.m이 있다.
- 적합도: 높음. 방 위쪽 고정 카메라, 1초 1장 간격, 사람 없는 시간이 길게 이어지는 실제 사무실이라 server.py의 판정 규칙(2프레임 연속, 30분 타이머)을 그대로 돌려 볼 수 있다. 엎드린 사람이나 짐을 일부러 두고 나간 장면이 있는지는 확인 필요다.

### 2. Employee Sleeping (Roboflow Universe)

- 공식 페이지: https://universe.roboflow.com/data-annotation-16ruy/employee-sleeping
- 논문: 없다. 페이지에 설명이 없다.
- 장면: 미리보기 이미지로 보면 사무실 책상에 엎드려 자는 사람이다. 사무실 몇 곳을 여러 각도에서 찍은 것으로 보이지만, 몇 곳 몇 명인지는 확인 필요다.
- 카메라: 미리보기 기준으로 천장 근처 CCTV처럼 방을 내려다보는 각도다. 설치 정보는 없다.
- 형태와 규모: 이미지 3,042장이다. 공개된 데이터셋 버전은 0개, 학습된 모델은 1개다.
- 라벨: Employee_is_Sleeping 1종 박스다. 미리보기에서는 엎드린 상체와 머리 주변에 박스가 있다. 깨어 있는 사람이나 빈 책상 라벨은 없다.
- 라이선스: 페이지에 CC BY 4.0으로 표기되어 있다. 올린 사람이 이미지를 직접 찍었는지 설명이 없어서, 이 표기가 원본 이미지 권리까지 보장하는지는 확인 필요다.
- 재배포: CC BY 4.0이면 출처 표기 후 가능하지만, 출처가 불분명하고 사람이 나오므로 올리지 않는다.
- 받는 방법: Roboflow 계정이 필요할 것으로 보인다(확인 필요). 공개 버전이 0개라 Dataset 탭에서 바로 받을 수 있는지는 확인 필요다. 페이지에 Fork Dataset 버튼은 있다.
- 적합도: 중간. 높은 각도에서 책상에 엎드린 사람을 모은 공개 데이터 중 가장 가깝다. YOLO 사람 박스가 라벨 박스와 겹치는지로 엎드린 사람 재현율을 잴 수 있다. 출처와 장면 다양성은 받아 보고 확인해야 한다.

### 3. SCB-Dataset (SCB-Dataset3-U 포함)

- 공식 페이지: https://github.com/Whiffe/SCB-dataset
- 논문: SCB-Dataset https://arxiv.org/abs/2304.02488 (v7, 2025년 8월), SCB-Dataset3 https://arxiv.org/abs/2310.02522
- 장면: 유치원부터 대학까지 교실이다. 본편 이미지는 bilibili, TikTok 등 공개 사이트의 수업 영상에서 뽑았다(2304.02488 v7). SCB-Dataset3-U는 저자들이 대학 실제 수업 녹화에서 모은 671장이다(2310.02522).
- 카메라: 논문은 정면, 측면, 후면 등 여러 각도라고 쓴다. 설치 높이나 CCTV 여부는 적혀 있지 않다.
- 형태와 규모: 본편 v7은 검출용 13,330장, 라벨 122,977개, 분류용 21,019장, 19종이다. SCB-Dataset3는 5,686장, 라벨 45,578개, 6종이고, 그중 3-U가 671장, 라벨 19,768개다.
- 라벨: 사람마다 박스와 행동 이름이 있다. 본편 19종에는 bow the head, turn the head 등이 있다. SCB-Dataset3-U에는 bowing the head 947개, leaning over the table 1,029개가 있다(2310.02522 Table III). 논문은 leaning over the table을 따로 정의하지 않는다. 이름으로는 책상에 엎드린 자세에 가장 가깝지만 실제 이미지로 확인해야 한다. sleeping 라벨은 공개판에 없다.
- 라이선스: LICENSE 파일이 없다. README는 학술 연구, 개인 학습, 비상업 용도로만 쓰라고 하고 저작권자가 모든 권리를 갖는다고 쓴다. 같은 README가 허락 없는 복사, 배포, 수정, 사용을 금지한다고도 쓴다. 두 문구가 서로 부딪치므로 수업 과제에 쓰기 전에 저자에게 메일로 확인하는 것이 안전하다(확인 필요). 본편은 공개 사이트 영상에서 뽑았으므로 원 영상의 권리도 확인 필요다.
- 재배포: 허락 없이는 금지다.
- 받는 방법: Baidu Netdisk(추출 코드는 README에 있다)와 Hugging Face(https://huggingface.co/datasets/wintonYF/SCB-Dataset)다. Hugging Face에는 SCB5 폴더들과 SCB_BowTurnHead_20250509 폴더가 있고 전체 31.8GB다. SCB-Dataset3는 README의 Baidu 링크로만 확인했다. README는 SCB-Dataset3를 다시 만드는 중이라고 쓴다.
- 적합도: 중간. 실제 교실에서 고개 숙임과 책상에 엎드림에 가까운 자세에 사람 박스가 달린 공개 데이터는 이것이 거의 유일하다. 수십 명이 앉은 교실이라 우리 방(4인 테이블)과 사람 크기, 각도가 다르다.

### 4. CAVIAR (INRIA 로비 세트)

- 공식 페이지: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
- 논문: 페이지에 데이터셋 논문이 따로 없다(확인 필요). 출처는 EC 지원 CAVIAR 프로젝트(IST 2001 37540)로 표기하라고 한다.
- 장면: 프랑스 그르노블 INRIA 연구소 입구 로비에서 배우가 연기한 시나리오다. 걷기, 둘러보기, 의자에서 쉬기, 바닥에 쓰러지기, 가방 두고 가기, 만나기, 싸우기가 있다. 가방 두고 가기는 LeftBag, LeftBag_AtChair, LeftBag_BehindChair, LeftBag_PickedUp, LeftBox 5개 클립이다. 의자에서 쉬는 Rest_InChair 클립도 있다.
- 카메라: 광각 렌즈다. 페이지의 예시 프레임은 로비를 거의 바로 위에서 내려다보고 가장자리가 휘어 있다. 설치 높이는 적혀 있지 않다.
- 형태와 규모: 2003~2004년 촬영이다. INRIA 세트는 28개 클립이고 384x288, 25 FPS MPEG2다. 클립 파일은 대부분 6~12MB다. 리스본 쇼핑센터 세트도 있지만 복도 장면이라 이 조사에서는 로비 세트만 본다.
- 라벨: 프레임별 사람과 무리 박스, 역할과 상황 라벨이 CVML 기반 XML로 있다.
- 라이선스: 페이지에 Creative Commons BY-SA라고 적혀 있다. 버전 번호는 없다(확인 필요). 수업 과제에 쓸 수 있다.
- 재배포: BY-SA 조건으로 가능하지만 우리는 올리지 않는다.
- 받는 방법: 클립별 MPEG2, JPEG tar.gz, 정답 XML을 페이지에서 바로 받는다.
- 적합도: 중간. 의자 옆에 가방만 남기고 사람이 떠나는 장면이 있어서 짐만 남은 장면의 오검출을 볼 수 있다. 로비가 넓고 해상도가 낮으며 거의 위에서 찍어 우리 각도와는 다르다.

### 5. PIROPO

- 공식 페이지: https://www.gti.ssr.upm.es/data/piropodatabase
- 논문: C. R. del Blanco, P. Carballeira, F. Jaureguizar, N. García, Signal Processing: Image Communication 93, 116135, 2021. https://doi.org/10.1016/j.image.2021.116135
- 장면: 마드리드 공과대학 GTI 연구실의 실내 방 2곳이다. Room A는 약 15x10m 열린 공간이고, Room B는 약 8x5m에 테이블, 의자, 컴퓨터가 있는 방이다. 사람이 걷고, 서고, 앉는다. 시퀀스별 설명표가 이미지로만 되어 있어서 책상에 앉은 장면이 얼마나 되는지는 확인 필요다.
- 카메라: Room A는 천장 어안 카메라 3대와 벽과 천장이 만나는 모서리에 단 일반 카메라 5대다. Room B는 천장 어안 1대와 일반 카메라 1대(conv_2B)이고, conv_2B 위치는 적혀 있지 않다(확인 필요). 해상도는 800x600(conv_2B는 704x576), 약 10 FPS다.
- 형태와 규모: 주석이 달린 프레임이 10만 장 이상이고, 주석 없는 프레임도 있다.
- 라벨: 사람 머리 중심 점(x, y)을 CSV로 준다. 박스는 없다. 그래서 YOLO 박스 정확도는 못 재고, 프레임별 사람 수나 사람 있음 여부를 맞추는 데 쓴다.
- 라이선스: 연구 목적으로 무료라고만 쓴다. 수업 과제가 연구 목적에 들어가는지와 재배포 조건은 확인 필요다.
- 받는 방법: 카메라별 zip을 페이지에서 바로 받는다. 신청서는 없다. 위 논문 인용을 요구한다.
- 적합도: 중간. 방 모서리 높은 곳의 일반 카메라로 테이블과 의자가 있는 실내를 찍어 우리 각도와 비슷한 편이다. 라벨이 점이라 박스 평가는 못 한다.

### 6. Toyota Smarthome (Trimmed, Untrimmed)

- 공식 페이지: https://project.inria.fr/toyotasmarthome/
- 논문: Trimmed는 ICCV 2019(https://openaccess.thecvf.com/content_ICCV_2019/papers/Das_Toyota_Smarthome_Real-World_Activities_of_Daily_Living_ICCV_2019_paper.pdf), Untrimmed(TSU)는 TPAMI 2022(https://arxiv.org/abs/2010.14982)다.
- 장면: 아파트 한 곳의 식당, 거실, 부엌에서 60~80세 18명이 대본 없이 하루 8시간씩 생활한다. 식탁에서 책 읽기, 노트북 쓰기, 글쓰기, 앉기와 일어나기, 들어오기와 나가기가 있다.
- 카메라: Kinect v1 7대다. 1, 2번은 식당, 4, 5번은 거실, 3, 6, 7번은 부엌이다(TSU 논문 3.1절). ICCV 논문 Figure 1의 평면도에서 카메라는 벽과 모서리에 있고, 샘플 화면은 위쪽에서 비스듬히 내려다본다. 설치 높이는 적혀 있지 않다. 640x480, 20 FPS이고 얼굴은 흐리게 처리했다.
- 형태와 규모: Trimmed는 16,115클립, 31종이다. Untrimmed는 영상 536개, 평균 21분(합계 약 188시간, 계산값), 51종이다.
- 라벨: 활동 이름과 시작, 끝 시간이다. TSU에는 Enter, Leave, Sit down, Get up, Lay down, Read, Use laptop, Write 등이 있다. 사람 박스 대신 2D와 3D 관절 자세를 준다. TSU는 사람이 보이는 구간만 남겼다고 쓰므로 빈 방 구간은 거의 없을 것이다.
- 라이선스: License_v2(https://project.inria.fr/toyotasmarthome/files/2020/12/License_v2.pdf)다. 비상업 과학 연구, 비상업 교육, 비상업 예술 프로젝트에 한해 신청자 개인에게 사용권을 준다. 수업 과제는 비상업 교육에 해당한다.
- 재배포: 금지다. 보관용 사본 1개를 빼고 복사, 공유, 배포, 제3자 제공을 막는다. 사용권이 개인에게 주어지므로 팀원끼리 파일을 돌려 보는 것도 막힐 수 있다. 쓸 사람이 각자 신청하는 것이 안전하다(확인 필요).
- 받는 방법: 홈페이지 신청서에 이름, 소속, 이메일, 데이터 종류, 짧은 프로젝트 설명을 쓰고 라이선스에 동의한다. 학교나 회사 이메일이어야 하고, Gmail 같은 개인 메일은 지도교수 이름과 메일을 적지 않으면 무시한다. 3일 안에 답이 없으면 toyotasmarthome@inria.fr로 연락하라고 한다.
- 적합도: 중간. 방 모서리 위쪽에서 식탁에 앉아 읽고 노트북을 쓰는 사람, 드나드는 사람이 있다. 노인 가정집이고 사람 박스가 없으며 빈 방 구간이 거의 없다. Lay down은 책상이 아닌 곳일 가능성이 크다(확인 필요).

### 7. CEPDOF

- 공식 페이지: https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof
- 논문: Z. Duan 외, "RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images", CVPR 2020 Workshops(OmniCV). https://arxiv.org/abs/2005.11623
- 장면: 작은 강의실 하나다. 점심 모임(10명 넘게 앉아 식사), 극단적인 자세와 모자나 후드로 머리가 가려진 경우, 두 문으로 드나들기, 조명을 끈 상태가 있다.
- 카메라: 천장에 단 어안 카메라다. 위에서 수직으로 내려다보므로 비스듬히 보는 우리 카메라와 사람 모양이 다르다.
- 형태와 규모: 영상 8개, 주석 프레임 25,504장이다. 2048x2048 또는 1080x1080이고 1Hz 또는 10Hz다.
- 라벨: 회전 박스(중심, 너비, 높이, 각도)와 사람 ID다. 형식은 COCO JSON에 가깝다.
- 라이선스: 프로젝트 페이지는 비상업 목적으로 쓸 수 있다고 쓰고, 다운로드 페이지(https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/download)는 Open Database License와 Database Contents License라고 쓴다. 두 페이지가 다르므로 확인 필요다. 어느 쪽이든 수업 과제에는 쓸 수 있다.
- 재배포: ODbL은 조건부로 허용하지만 비상업 문구와 함께 보면 불분명하다(확인 필요).
- 받는 방법: 프로젝트 페이지는 이름, 소속, 이메일 신청서를 거쳐 다운로드 페이지로 간다고 쓴다. 다운로드 페이지에는 CEPDOF.zip(9GB) 직접 링크가 있다.
- 적합도: 낮음. 모자와 후드를 쓴 사람, 앉은 사람이 있지만 어안 수직 시점이라 결과를 우리 방에 옮겨 해석하기 어렵다.

### 8. EduNet

- 공식 페이지: https://github.com/vijetait/Classroom-Monitoring-Action-Dataset (README의 웹사이트 edunet-drsta.com은 2026-10-19에 접속되지 않았다)
- 논문: V. Sharma 외, "EduNet: A New Video Dataset for Understanding Human Activity in the Classroom Environment", Sensors 21(17), 5699, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8434086/
- 장면: 인도 북부 농촌 초중고 교실을 DSLR로 찍은 영상과 YouTube 수업 영상이다.
- 카메라: 논문은 여러 각도라고만 쓴다. 고정 카메라인지, 높이가 얼마인지는 확인 필요다.
- 형태와 규모: 7,851클립, 약 12시간, 1280x720, 30 FPS다. 클립 길이는 3~12초다.
- 라벨: 클립 단위 행동 20종이다. 학생 쪽에 Sleeping, Sitting_on_Desk, Reading_Book, Standing 등이 있다. 사람 박스는 없다. Sleeping이 책상에 엎드린 자세인지는 확인 필요다.
- 라이선스: 데이터 라이선스가 적혀 있지 않다(확인 필요). 논문은 CC BY 4.0이지만 데이터에는 해당하지 않는다.
- 재배포: 확인 필요.
- 받는 방법: README의 Google Form으로 신청한다.
- 적합도: 낮음. Sleeping 클립이 있어 엎드린 사람 재현율을 볼 수는 있지만, 박스가 없고 각도가 제각각이며 신청해도 받을 수 있을지 불확실하다.

### 9. Classroom Behavior (Roboflow Universe, CV lounge)

- 공식 페이지: https://universe.roboflow.com/cv-lounge/classroom-behavior-iz57f
- 논문: 없다. 페이지에 설명이 없다.
- 장면: 미리보기 기준으로 강의실 책상에 앉은 학생과 책상에 엎드린 학생이다.
- 카메라: 미리보기 기준으로 교실 뒤쪽 위에서 찍은 사진과 눈높이 사진이 섞여 있다(확인 필요).
- 형태와 규모: 이미지 4,881장이다. 공개 버전은 0개다.
- 라벨: hand, read, sleep, write 4종 박스다.
- 라이선스: 페이지에 CC BY 4.0으로 표기되어 있다. 이미지 출처 설명이 없으므로 다른 데이터셋 이미지를 다시 올린 것인지 확인 필요다.
- 재배포: 출처를 모르므로 올리지 않는다.
- 받는 방법: Employee Sleeping과 같다(확인 필요).
- 적합도: 낮음. sleep 박스는 있지만 출처와 각도를 알 수 없다.

### 10. ABODA (Abandoned Object Dataset)

- 공식 페이지: https://github.com/kevinlin311tw/ABODA
- 논문: README에 없다(확인 필요).
- 장면: 버려진 물체 탐지용 시퀀스 11개다. 붐비는 장면, 조명 변화, 야간, 실내와 실외가 섞여 있다. 영상별 실내외 구분은 적혀 있지 않다.
- 카메라: 확인 필요.
- 형태와 규모: AVI 11개, 합계 약 270MB다(GitHub 파일 목록 기준).
- 라벨: 저장소에는 영상 11개와 README만 있고 정답 파일이 없다.
- 라이선스: 표기가 없다. 표기가 없으면 저작권자가 모든 권리를 가지므로 사용 전에 확인 필요다.
- 재배포: 확인 필요.
- 받는 방법: 저장소에서 바로 받는다.
- 적합도: 낮음. 물건만 남는 장면은 있지만 실내 장면이 몇 개인지, 사람이 앉는 공간인지 알 수 없다.

## 찾지 못한 것

- 스터디룸이나 작은 회의실에서 방 모서리 높은 곳 카메라로 책상에 엎드려 자는 사람을 찍고 사람 박스를 단 공개 데이터는 찾지 못했다. 가장 가까운 Employee Sleeping은 출처 설명이 없고, SCB-Dataset3-U는 수십 명이 앉은 교실이다. 엎드린 사람 재현율은 우리 방에서 직접 찍어야 한다.
- 엎드림이나 수면 라벨이 있는 교실 데이터는 대부분 공개되지 않았다. 30개 학교에서 모은 수면 샘플 3천 개 데이터와 초등학교의 책상에 엎드림 1,000장 데이터는 SCB-Dataset3 논문 Table I(https://arxiv.org/abs/2310.02522)에만 나오고, SCB README 9장은 sleeping이 있는 SCB-E를 비공개 데이터로 분류한다. ARIC(https://arxiv.org/abs/2410.12337)는 실제 교실 감시 이미지에 Sleeping 라벨이 있지만 원본 이미지 대신 특징 벡터만 공개한다(https://ivipclab.github.io/publication_ARIC/ARIC).
- 사람은 없고 가방, 노트북, 외투만 의자와 책상에 남은 장면을 따로 라벨링한 공개 데이터는 찾지 못했다. Edinburgh의 빈 방 프레임은 짐을 일부러 둔 장면이 아니고, CAVIAR와 ABODA는 로비나 공공장소에 가방 하나를 두는 장면이다.
- 도서관 좌석 점유 데이터는 쓸 만한 것이 없었다. Roboflow의 Beyond(도서관 좌석, 233장, Empty Seat와 Occupied Seat, https://universe.roboflow.com/lyeonsl/beyond-sub9e)와 Seat occupancy detection(4장, https://universe.roboflow.com/project-ns798/seat-occupancy-detection)은 규모가 작고 촬영 정보가 없다. 짐만 있는 좌석을 따로 구분했는지도 알 수 없다. 도서관 좌석 논문 https://arxiv.org/abs/2306.16080 은 데이터를 공개하지 않았다. IEEE DataPort의 도서관 스터디 부스 데이터(https://ieee-dataport.org/documents/grid-eye-dataset-human-occupancy-detection-and-counting-study-pod)는 8x8 열화상 센서이고 유료 구독이 필요하다.
- 30분 넘는 자리 비움이 라벨로 표시된 데이터는 없다. Edinburgh에 긴 빈 방 구간은 있지만 짐을 두고 나갔는지는 라벨에 없다.
- 실내에서 패딩과 모자를 입은 사람만 모은 데이터는 찾지 못했다. CEPDOF에 모자와 후드가 있지만 어안 수직 시점이다.
- 유리벽 너머 복도의 사람이 보이는 장면은 찾지 못했다. 감지 영역 밖 사람을 잡는지는 우리 영상으로 확인해야 한다.
- 확인했지만 표에서 뺀 것은 다음과 같다.
  - Le2i 낙상 데이터(사무실, 강의실 장면, 프레임별 사람 박스): 페이지에 새 다운로드 링크를 기다리는 중이라고만 있고 라이선스 표기가 없다. https://imvia.ube.fr/?p=346
  - PETS 2006과 AVSS 2007 i-LIDS 버려진 가방 데이터: 기차역 같은 넓은 공간이고 공식 페이지(http://www.cvg.reading.ac.uk/PETS2006/data.html, https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html)에 접속되지 않았다.
  - Brainwash(카페 위쪽 카메라, 머리 박스): Stanford 저장소 페이지에 파일 목록이 보이지 않아 받을 수 없는 것으로 보인다(확인 필요). https://purl.stanford.edu/sx925dc9385
  - MEVA: CC BY 4.0이고 person_abandons_package, person_sits_down 라벨과 사람 박스가 있지만 대부분 야외 훈련장이고, 논문이 밝힌 실내는 체육관과 카페 로비다. https://mevadata.org/
  - SCUT-HEAD Part A: 대학 강의실 감시 영상에서 뽑은 2,000장이지만 머리 박스만 있고 라이선스 표기가 없다. https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
  - WiseNET: CC0이고 실내 카메라 6대에 앉기와 멈춤 라벨이 있지만, 공간이 s1~s6로만 표시되고 s1이 27.7m x 1.7m라 복도 위주로 보이며 영상이 1시간 13분뿐이다. https://data.4tu.nl/datasets/5f790109-8369-4207-a3ee-35174c144483
  - 천장 무지향 카메라 회의실 데이터(https://arxiv.org/abs/2007.04678): 논문은 공개하겠다고 쓰지만 링크를 찾지 못했다.
  - IndoorCrowd: 아트리움, 복도, 입구 홀의 군중 장면이라 뺐다. https://sheepseb.github.io/IndoorCrowd/
- 그래서 직접 찍어야 할 장면은 혼자 책상에 엎드린 사람, 패딩과 모자를 입은 사람, 의자 위 외투와 가방만 남은 30분 이상의 빈 방, 유리벽 너머 복도 통행이다.

## 라이선스 규칙

- 이 저장소는 공개라서 어떤 데이터셋도 파일 자체를 커밋하지 않는다. CC BY-SA나 CC BY-NC-SA처럼 재배포를 허용하는 데이터도 사람이 나오므로 올리지 않는다.
- Toyota Smarthome은 라이선스가 공유와 제3자 제공을 막고, SCB-Dataset은 허락 없는 복사와 배포를 막는다. 이 둘은 프레임 한 장도 올리지 않는다.
- 커밋할 수 있는 것은 데이터를 읽는 스크립트, 프레임별 판정 CSV처럼 숫자만 담은 결과, 그래프다.
- 지금 .gitignore는 *.mp4, frames/, snapshots/를 막는다. .avi, .tar, .mat, .jpg는 막지 않으므로 받은 데이터는 저장소 밖 폴더에 둔다.
- 발표 자료에 결과를 쓸 때는 각 페이지가 요구하는 출처 표기(Edinburgh 감사 문구, CAVIAR 프로젝트 표기, 논문 인용)를 넣는다.
