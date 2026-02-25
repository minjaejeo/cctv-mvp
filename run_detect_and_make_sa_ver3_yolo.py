import os, json
import cv2
from ultralytics import YOLO
import torch

# GPU가 있으면 GPU, 없으면 CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("DEVICE = ", DEVICE)


VIDEO_PATH = "video.mp4"
OUT_SA_PATH = "sa_ver3.json" # 내 결과

# ROI (오른쪽 30%) 가정
INTRUSION_REGION_X_RATIO = 0.7

HIT_FRAMES = 5 # 사람이 ROI에 들어와서 연속 5프레임 나오면 진짜 이벤트로 측정을 위한 값
COOLDOWN_SEC = 5.0 # 이벤트 한 번 찍으면 5초 동안 다시 안 찍기 위한 값
CONF_THRES = 0.35 # YOLO 신뢰도 임계값 ("사람이 맞다"라고 확신하는 값이 0.35)

# 사람이 오른쪽 영역에 들어왔는지 판단하는 함수
def in_roi(box, x_cut):
    # YOLO가 찾은 사람 box: (x1, y1, x2, y2)
    x1,y1,x2,y2 = box
    cx = (x1 + x2) / 2
    return cx >= x_cut

def main():
    if not os.path.exists(VIDEO_PATH):
        raise SystemExit(f"영상 파일 없음: {VIDEO_PATH}") # 영상이 존재하지 않으면 종료
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("영상 열기 실패") # 영상 열기 실패하면 종료
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # 영상 프레임 가져오기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 영상 프레임의 가로 픽셀
    x_cut = int(w * INTRUSION_REGION_X_RATIO) # ROI 시작 x좌표

    # YOLO 모델 로드 
    model = YOLO("yolov8n.pt")

    frame_idx = 0
    sa_events = []

    hit = 0 # ROI에 사람이 몇번 있었는지 카운트
    last_event_time = -1e9 # 마지막으로 이벤트를 찍은 시간

    while True:
        ret, frame = cap.read() # 성공여부, 프레임이미지 전달
        if not ret:
            break
        frame_idx += 1
        t = frame_idx / fps # (프레임수 / fps) 초, 대략 현재 시각

        # YOLO 추론 ("이 프레임에서 물체를 찾아라")
        results = model.predict(frame, device=DEVICE, conf=CONF_THRES, verbose=False)

        # 사람 탐지 여부 + ROI 안인지 판단
        person_in_roi = False # 최종 판단 결과
        for r in results:
            if r.boxes is None:
                continue
            
            # 각 박스의 (클래스, 확신도, 좌표)를 하나씩 꺼냄
            for cls_id, conf, xyxy in zip(r.boxes.cls.tolist(),
                                          r.boxes.conf.tolist(),
                                          r.boxes.xyxy.tolist()):
                if int(cls_id) == 0 and conf >= CONF_THRES: # "사람"이고 "확신도가 기준 이상이면"
                    x1, y1, x2, y2 = xyxy
                    if in_roi((x1, y1, x2, y2), x_cut): # 좌표 뽑아서 ROI 안인지 확인
                        person_in_roi = True # 한 명이라도 ROI 안이면 TRUE
                        break
            if person_in_roi:
                break

        # ROI 안에 사람이 있다면 hit++ 아니면 hit 0으로 초기화
        if person_in_roi:
            hit += 1
        else:
            hit = 0

        # 연속 5프레임 ROI에 사람 존재하고 마지막 이벤트 이후 5초 지났으면
        if hit >= HIT_FRAMES and (t - last_event_time) >= COOLDOWN_SEC:
            sa_events.append({
                "video_id": os.path.basename(VIDEO_PATH),
                "event_type": "intrusion",
                "event_time_sec": round(t, 3)
            })
            last_event_time = t # 마지막 이벤트 시간 갱신
            hit = 0 # hit 초기화해서 같은 장면에서 연속 중복 기록을 줄임

    cap.release() # 영상 닫기

    with open(OUT_SA_PATH, "w", encoding="utf-8") as f:
        json.dump(sa_events, f, ensure_ascii=False, indent=2)

    print("SA saved: ", OUT_SA_PATH)
    print("SA events: ", sa_events)


if __name__ == "__main__":
    main()


