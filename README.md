# CCTV AI Intrusion Detection MVP

YOLO 및 Motion Detection 기반 CCTV 침입 탐지 프로젝트  
Python + OpenCV + YOLOv8 + PyTorch

---

## 📌 Project Overview

본 프로젝트는 CCTV 영상에서 침입 이벤트(intrusion)를 탐지하고,  
Ground Truth(GT)와 비교하여 F1-Score를 계산하는 AI 영상 분석 MVP입니다.

구현 단계:

- Level 1: 단순 Motion 기반 침입 탐지
- Level 2: Motion + Hit Frame + Cooldown 적용
- Level 3: YOLO 기반 사람(person) 탐지

---

## 📁 Current Project Structure

```
CCTV_MVP
│
├── src/
│    ├── run_detect_and_make_sa.py
│    ├── run_detect_and_make_sa_ver2.py
│    ├── run_detect_and_make_sa_ver3_yolo.py
│    ├── eval_f1.py
│    ├── eval_f1_ver2.py
├── data/
│   ├── gt.json
│   ├── gt_yolo_test.json
│   ├── sa.json
│   ├── sa_ver2.json
│   ├── sa_ver3.json
│   ├── video.mp4
│   ├── video_yolo_test.mp4
├── models    
│   ├── yolov8n.pt
│
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Test Video

Download a sample video from:
(https://pixabay.com/ko/videos/%ec%82%ac%eb%9e%8c%eb%93%a4-%ea%b1%b7%eb%8a%94-%ec%8b%a4%eb%a3%a8%ec%97%a3-%ec%a0%9c%eb%b0%a9-138564/)
(https://pixabay.com/ko/videos/%eb%82%a8%ec%84%b1-%eb%b3%b5%eb%8f%84-%ec%bb%a4%ed%94%bc-%ed%95%b8%eb%93%9c%ed%8f%b0-73531/)
(https://pixabay.com/ko/videos/%ed%9d%90%eb%a6%84-%ea%b0%95%eb%91%91-%eb%96%a8%ec%96%b4%ec%a7%80%eb%8b%a4-%eb%ac%bc-163198/)

---

## ⚙️ Installation

### 1️⃣ Python Version

Python 3.12 권장

```
python --version
```

---

### 2️⃣ 가상환경 생성

```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ 패키지 설치

```
pip install -r requirements.txt
```

---

## ▶️ Usage

---

### 🔹 Level 1 - Motion 기반 탐지

```
python run_detect_and_make_sa.py
```

출력:
```
sa.json
```

---

### 🔹 Level 2 - Motion + 안정화 로직

```
python run_detect_and_make_sa_ver2.py
```

출력:
```
sa_ver2.json
```

적용 기능:
- Background Subtraction (MOG2)
- ROI 기반 움직임 감지
- Hit Frame 조건
- Cooldown 적용

---

### 🔹 Level 3 - YOLO 기반 침입 탐지

```
python run_detect_and_make_sa_ver3_yolo.py
```

출력:
```
sa_ver3.json
```

적용 기능:
- YOLOv8n 모델 사용
- person(class=0) 탐지
- ROI 진입 시 intrusion 이벤트 기록
- Hit Frame + Cooldown 적용
- CUDA(GPU) 자동 사용

---

## 📊 Evaluation (F1 Score)

---

### 🔹 기본 평가

```
python eval_f1.py
```

GT: `gt.json`  
SA: `sa.json`

---

### 🔹 Level 2/3 평가

```
python eval_f1_ver2.py
```

GT: `gt_yolo_test.json` 또는 `gt.json`  
SA: `sa_ver2.json` 또는 `sa_ver3.json`

출력 항목:

- TP (True Positive): 맞춘 개수
- FP (False Positive): SA에 있지만 GT에 없는 경우 (오검출)
- FN (False Negative): GT에 있지만 SA에 없는 경우 (미검출)
- Precision (정밀도): 찍은 것 중에 맞춘 비율
- Recall (재현율): 정답 중 맞춘 비율
- F1 Score: Precision과 Recall의 조화 평균

---

## 🧠 Detection Logic Summary

### Motion Detection

1. 영상 프레임 읽기
2. Grayscale + Blur
3. Background Subtraction
4. ROI 영역 움직임 픽셀 계산
5. Threshold 초과 시 이벤트 기록

---

### YOLO Detection

1. YOLOv8n 모델 로드
2. 프레임별 객체 탐지
3. person(class=0)만 필터링
4. ROI 진입 여부 판단
5. Hit Frame 조건 만족 시 intrusion 이벤트 기록

---

## 📦 requirements.txt 역할

`requirements.txt`는 이 프로젝트 실행에 필요한 라이브러리 목록입니다.

다른 PC에서 아래 명령으로 동일한 환경을 구성할 수 있습니다:

```
pip install -r requirements.txt
```

---

## ⚠️ Note

- mp4 영상 파일 및 모델 가중치는 Git에 포함하지 않는 것을 권장합니다.
- YOLO 테스트 시 실제 사람 영상이 필요합니다.
- GPU 사용 시 torch + CUDA 버전이 일치해야 합니다.

---

## 👨‍💻 Author

CCTV AI Intrusion Detection MVP  
Python / OpenCV / YOLOv8 / PyTorch
