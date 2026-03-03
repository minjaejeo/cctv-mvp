import os, json, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional # 변수 지정

import cv2 # 영상 읽기/쓰기 + 선/글씨/박스 그리기
import torch # GPU(CUDA) 사용 가능 여부 확인
from ultralytics import YOLO # YOLOv8 모델 로드 + 탐지/추적

"""
Config (설정값)
"""
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("DEVICE = ", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

SOURCE = os.path.join(ROOT_DIR, "data", "138564-769988151_medium.mp4")
OUT_SA_PATH = os.path.join(ROOT_DIR, "data", "sa_ver4.json") # 이벤트 결과 저장
OUT_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "vis_ver4.mp4") # 시각화 결과 저장

MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov8n.pt") # YOLO 모델 파일
CONF_THRES = 0.25 # 25% 이상 사람일 확신이면 일단 사용
IOU_THRES = 0.5 # 중복 제거

# ROI: 오른쪽 30%
INTRUSION_REGION_X_RATIO = 0.7 # 화면 가로 100%라고 가정하면 왼쪽부터 70% 지점부터 오른쪽(나머지 30%)을 ROI로 잡는다

# 이벤트 파라미터
HIT_FRAMES = 2 # ROI 진입 안정화(2프레임 연속) e.g. 2프레임 연속 ROI 안에 있으면 이벤트 인정
COOLDOWN_SEC = 2.0 # 같은 이벤트 과다 방지 e.g. 이벤트 한 번 찍고 몇 초 동안은 또 안 찍기
LOITER_SEC = 3.0 # ROI 안에 3초 이상이면 Loitering 
LINE_CROSS_COOLDOWN = 1.0 # 라인 크로싱 연속 방지 (선 넘는 이벤트가 연속으로 너무 많이 찍히는 걸 막는 잠금 시간)

# Tracking
TRACKER_CFG = "bytetrack.yaml" # ultralytics 내장 (YOLO 탐지 결과를 "같은 사람은 같은 ID"로 붙이는 추적기 설정)


"""
Helpers (자주 쓰는 작은 함수들)
"""
# 사람 박스의 중심 x 좌표(cx)가 x_cut보다 오른쪽이면 ROI 안이라고 판단하는 함수
def in_roi(cx: float, x_cut: int) -> bool:
    return cx >= x_cut

# YOLO 박스는 왼쪽위(x1,y1), 오른쪽아래(x2,y2) => (cx,cy) 중심점 구하는 함수
def bbox_center_xyxy(xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

# 빨간 선을 x_cut 위치에 그려서 "여기부터 ROI"를 눈으로 보이게 하는 함수
def draw_roi(frame, x_cut):
    h, w = frame.shape[:2]
    cv2.line(frame, (x_cut, 0), (x_cut, h), (0, 0, 255), 2)
    cv2.putText(frame, "ROI", (x_cut + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# 박스를 그리고, ID 값을 붙이는 함수
def draw_box(frame, xyxy, track_id: int, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID={track_id}", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# 결과 이벤트들을 쌓이게 하는 리스트
def emit(sa_events: list, video_id: str, event_type: str, t: float, track_id: int, xyxy, extra=None):
    e = {
        "video_id": video_id,
        "event_type": event_type,
        "event_time_sec": round(t, 3),
        "track_id": int(track_id),
        "bbox_xyxy": [float(x) for x in xyxy],
    }
    if extra:
        e.update(extra)
    sa_events.append(e)

# 사람 1명마다 상태 저장
@dataclass
class TrackState:
    last_seen_t: float = 0.0
    last_seen_frame: int = 0
    in_roi_hit: int = 0 # ROI 안에 연속 몇 프레임 있었나
    roi_enter_t: Optional[float] = None # ROI 안에 처음 들어온 시간
    intrusion_emitted: bool = False # intrusion 이미 찍혔는지 확인
    loiter_emitted: bool = False # loitering 이미 찍었는지 확인
    last_event_t: float = -1e9 # 마지막 이벤트 찍은 시간 (쿨다운 체크)
    last_cx: Optional[float] = None # last_cx: 이전 프레임에서의 중심 x 좌표(선 넘었는지 확인용)
    last_line_cross_t: float = -1e9 # 마지막 line_crossing 찍은 시간

"""
Main (실제 실행)
"""

def main():
    video_id = os.path.basename(SOURCE)

    cap = cv2.VideoCapture(SOURCE)
    # 영상 열기 실패 시 종료
    if not cap.isOpened():
        raise SystemExit(f"영상/스트림 열기 실패: {SOURCE}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # 1초에 몇 프레임인지
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    x_cut = int(w * INTRUSION_REGION_X_RATIO) # ROI 시작 선 위치

    # 시각화 저장 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (w, h))

    model = YOLO(MODEL_PATH)

    frame_idx = 0
    sa_events: List[dict] = []
    tracks: Dict[int, TrackState] = {}

    start_wall = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = frame_idx / fps

        # Tracking inference (person only via classed=[0]) 탐지 + 추적(= ID 부여)
        results = model.track(
            frame,
            device=DEVICE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            persist=True, # 다음 프레임에도 ID를 이어간다.
            tracker=TRACKER_CFG,
            classes=[0],    # person
            verbose=False
        )

        draw_roi(frame, x_cut)

        # results 구조: 리스트지만 보통 1개
        r0 = results[0]
        if r0.boxes is not None and r0.boxes.id is not None:
            ids = r0.boxes.id.tolist() # 각 사람의 id
            xyxys = r0.boxes.xyxy.tolist() # 각 사람의 박스 좌표 리스트
            confs = r0.boxes.conf.tolist() # 각 박스의 confidence
            
            # 사람 한 명씩 처리 (zip은 "같은 인덱스끼리 묶어서" 처리하는 도구)
            for track_id, xyxy, conf in zip(ids, xyxys, confs):
                track_id = int(track_id)
                cx, cy = bbox_center_xyxy(xyxy)
                st = tracks.get(track_id, TrackState())
                st.last_seen_t = t
                st.last_seen_frame = frame_idx

                # line crossing (왼->오 / 오->왼)
                if st.last_cx is not None:
                    crossed = (st.last_cx < x_cut and cx >= x_cut) or (st.last_cx >= x_cut and cx < x_cut)
                    if crossed and (t - st.last_line_cross_t) >= LINE_CROSS_COOLDOWN:
                        emit(sa_events, video_id, "line_crossing", t, track_id, xyxy, extra={"line_x": x_cut})
                        st.last_line_cross_t = t
                st.last_cx = cx

                # ROI hit logic (ROI 안이면 hit 증가, 밖이면 hit 초기화)
                if in_roi(cx, x_cut):
                    st.in_roi_hit += 1
                    if st.roi_enter_t is None:
                        st.roi_enter_t = t

                else:
                    st.in_roi_hit = 0
                    st.roi_enter_t = None

                # intrusion (진입 이벤트 1회)
                if (not st.intrusion_emitted) and st.in_roi_hit >= HIT_FRAMES and (t - st.last_event_t) >= COOLDOWN_SEC:
                    emit(sa_events, video_id, "intrusion", t, track_id, xyxy, extra={"roi": "right_30"})
                    st.intrusion_emitted = True
                    st.last_event_t = t
                
                # loitering (ROI 체류시간) e.g. 3초 이상 머무르면 loitering 기록
                if st.roi_enter_t is not None and (not st.loiter_emitted):
                    dwell = t - st.roi_enter_t
                    if dwell >= LOITER_SEC and (t - st.last_event_t) >= COOLDOWN_SEC:
                        emit(sa_events, video_id, "loitering", t, track_id, xyxy, extra={"dwell_sec": round(dwell, 3), "roi": "right_30"})
                        st.loiter_emitted = True
                        st.last_event_t = t
                # 상태 저장
                tracks[track_id] = st

                # 박스와 ID 그리기
                draw_box(frame, xyxy, track_id, color=(0, 255, 0))

            # 최근 이벤트 1개를 화면에 텍스트로 보여주고 결과 영상을 파일로 저장
            if sa_events:
                last = sa_events[-1]
                cv2.putText(frame, f"EVENT: {last['event_type']} ID={last['track_id']} t={last['event_time_sec']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            writer.write(frame)

    cap.release()
    writer.release()

    with open(OUT_SA_PATH, "w", encoding="utf-8") as f:
        json.dump(sa_events, f, ensure_ascii=False, indent=2)

    print("SA saved: ", OUT_SA_PATH, "events= ", len(sa_events))
    print("VIS saved: ", OUT_VIDEO_PATH)
    print("elapsed(sec)=", round(time.time() - start_wall, 2))

        

if __name__ == "__main__":
    main()