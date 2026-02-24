import cv2
import json
import numpy as np
import os

print("CWD:", os.getcwd())


# 0) 설정
VIDEO_PATH = "video.mp4"
print("exists: ", os.path.exists(VIDEO_PATH))
# VIDEO_PATH = r"C:\Users\79296\Downloads\video.mp4"
# OUT_SA_PATH = "sa.json"
OUT_SA_PATH = "sa_ver2.json" # 내 결과


EVENT_TYPE = "intrustion" # 이벤트 유형 : 침입
INTRUSION_REGION_X_RATIO = 0.7 # "침입 영역"을 임시로 화면 오른쪽 30%로 가정 (나중에 map파일로 교체)

MOTION_THRESHOLD = 2000 # motion 최대 3463 기준 (임계치)
HIT_FRAMES = 5 # 연속 5프레임 이상 기준 충족해야 이벤트 인정
COOLDOWN_SEC = 5.0 # 한번 찍고 5초 동안은 재기록 금지

WARMUP_FRAMES = 30 # 배경차분기 안정화(초반 학습) 위해 처음 30프레임은 스킵
LOG_EVERY_N_FRAMES = 10 # 로그 너무 많아지는 것 방지



# 1) 실행 전 점검
print("CWD:", os.getcwd())
print("VIDEO_PATH exists: ", os.path.exists(VIDEO_PATH))

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit("영상 열기 실패: 경로 확인 필요")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
print("영상 열기 성공")
print("FRAME_COUNT:", frame_count, "FPS:", fps)

# 2) 배경차분기
bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# 3) 상태 변수
frame_idx = 0
sa_events = []

hit = 0 # 연속 hit 카운터
last_event_time = -1e9 # 마지막 이벤트 기록 시간(초) - 아주 작은 값으로 초기화


# 4) 프레임 루프

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    t = frame_idx / fps

    # (A) 초반 워밍업: 배경모델 학습 시간
    if frame_idx <= WARMUP_FRAMES:
        # 그래도 bg.apply()는 해줘야 배경학습이 진행됨
        gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 컬러 -> 흑백
        gray0 = cv2.GaussianBlur(gray0, (7, 7), 0) # 노이즈 제거 위해 블러
        _ = bg.apply(gray0) # 배경 모델 업데이트
        continue

    # (B) 전처리: 흑백 + 블러
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 컬러 -> 흑백
    gray = cv2.GaussianBlur(gray, (7, 7), 0) # 노이즈 제거 위해 블러

    # (C) 배경 차분 적용 -> 움직이는 영역 마스크
    fg = bg.apply(gray)

    # (D) 이진화(흰/검) + 노이즈 제거
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY) # 픽셀이 200보다 크면 -> 255(움직임), 200 이하면 -> 0(배경/노이즈)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) # 작은 노이즈 제거 위해 열림 연산

    # (E) 침입 영역(오른쪽 30%) ROI 잘라내기
    h, w = fg.shape[:2]
    x_cut = int(w * INTRUSION_REGION_X_RATIO)
    roi = fg[:, x_cut:]

    # (F) 모션 측정: roi에서 흰색 픽셀 개수(움직임 픽셀 수)
    motion = int(np.sum(roi > 0))

    # (G) 로그
    if frame_idx % LOG_EVERY_N_FRAMES == 0:
        print(f"frame={frame_idx}, t={t:.2f}, motion={motion}, hit={hit}")

    
    # 5) 모션이 임계치 넘으면 hit 카운터 증가, 연속 hit 프레임 수로 이벤트 판단, 이벤트 기록 후 쿨다운 적용
    # 5-1) threshold 넘으면 hit 누적, 아니면 hit 초기화
    if motion > MOTION_THRESHOLD:
        hit += 1
    else:
        hit = 0

    # 5-2) 연속 HIT_FRAMES 충족 + 쿨다운 지난 경우만 이벤트 기록
    if hit >= HIT_FRAMES and (t - last_event_time) >= COOLDOWN_SEC:
        sa_events.append({
            "video_id": "video.mp4",
            "event_type": EVENT_TYPE,
            "event_time_sec": round(t, 3)
        })
        last_event_time = t
        hit = 0 # 다음 이벤트를 위해 hit 초기화

# 6) 정리 & SA 저장
cap.release()

with open(OUT_SA_PATH, "w", encoding="utf-8") as f:
    json.dump(sa_events, f, ensure_ascii=False, indent=2)

print(" SA saved: ", OUT_SA_PATH)
print("SA events: ", sa_events)

