import cv2
import json
import numpy as np
import os

print("CWD:", os.getcwd())


VIDEO_PATH = "video.mp4"
print("exists: ", os.path.exists(VIDEO_PATH))
# VIDEO_PATH = r"C:\Users\79296\Downloads\video.mp4"
OUT_SA_PATH = "sa.json"

# "침입 영역"을 임시로 화면 오른쪽 30%로 가정 (나중에 map파일로 교체)
INTRUSION_REGION_X_RATIO = 0.7

# 이벤트 중복 방지
event_written = False

cap = cv2.VideoCapture(VIDEO_PATH)

print("FRAME_COUNT:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not cap.isOpened():
    print("영상 열기 실패")
    exit()
else:
    print("영상 열기 성공")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# 배경 차분기
bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

frame_idx = 0
sa_events = []

while True:
    ref, frame = cap.read()
    if not ref:
        break
    frame_idx += 1
    t = frame_idx / fps

    # 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    fg = bg.apply(gray)
    # 노이즈 제거
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    h, w = fg.shape[:2]
    x_cut = int(w * INTRUSION_REGION_X_RATIO)

    # 침입영역(오른쪽 영역)에서 움직임 픽셀 수 측정
    roi = fg[:, x_cut:]
    motion = int(np.sum(roi > 0))

    print(f"frame={frame_idx}, t={t:.2f}, motion={motion}")
    # 임계치: 영상마다 조정 필요 (일단 3000으로 시작)
    if (motion > 3000) and (not event_written):
        sa_events.append({
            "video_id": "video.mp4",
            "event_type": "intrusion",
            "event_time_sec": round(t, 3)
        })
        event_written = True
        # 한 번만 찍고 끝내는 MVP
        break

cap.release()

with open(OUT_SA_PATH, "w", encoding="utf-8") as f:
    json.dump(sa_events, f, ensure_ascii=False, indent=2)

print("SA saved: ", OUT_SA_PATH, sa_events) 