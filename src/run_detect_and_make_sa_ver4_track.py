import os, json, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import torch
from ultralytics import YOLO

"""
Config
"""
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("DEVICE = ", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

SOURCE = os.path.join(ROOT_DIR, "data", "138564-769988151_medium.mp4")
OUT_SA_PATH = os.path.join(ROOT_DIR, "data", "sa_ver4.json")
OUT_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "vis_ver4.mp4") # 시각화 결과 저장

MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov8n.pt")
CONF_THRES = 0.25
IOU_THRES = 0.5

# ROI: 오른쪽 30%
INTRUSION_REGION_X_RATIO = 0.7

# 이벤트 파라미터
HIT_FRAMES = 2 # ROI 진입 안정화(2프레임 연속)
COOLDOWN_SEC = 2.0 # 같은 이벤트 과다 방지
LOITER_SEC = 3.0 # ROI 안에 3초 이상이면 Loitering
LINE_CROSS_COOLDOWN = 1.0 # 라인 크로싱 연속 방지

# Tracking
TRACKER_CFG = "bytetrack.yaml" # ultralytics 내장


"""
Helpers
"""
def in_roi(cx: float, x_cut: int) -> bool:
    return cx >= x_cut

def bbox_center_xyxy(xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def draw_roi(frame, x_cut):
    h, w = frame.shape[:2]
    cv2.line(frame, (x_cut, 0), (x_cut, h), (0, 0, 255), 2)
    cv2.putText(frame, "ROI", (x_cut + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

def draw_box(frame, xyxy, track_id: int, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID={track_id}", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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


@dataclass
class TrackState:
    last_seen_t: float = 0.0
    last_seen_frame: int = 0
    in_roi_hit: int = 0
    roi_enter_t: Optional[float] = None
    intrusion_emitted: bool = False
    loiter_emitted: bool = False
    last_event_t: float = -1e9
    last_cx: Optional[float] = None
    last_line_cross_t: float = -1e9

"""
Main
"""

def main():
    video_id = os.path.basename(SOURCE)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"영상/스트림 열기 실패: {SOURCE}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    x_cut = int(w * INTRUSION_REGION_X_RATIO)

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

        # Tracking inference (person only via classed=[0])
        results = model.track(
            frame,
            device=DEVICE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            persist=True,
            tracker=TRACKER_CFG,
            classes=[0],    # person
            verbose=False
        )

        draw_roi(frame, x_cut)

        # results 구조: 리스트지만 보통 1개
        r0 = results[0]
        if r0.boxes is not None and r0.boxes.id is not None:
            ids = r0.boxes.id.tolist()
            xyxys = r0.boxes.xyxy.tolist()
            confs = r0.boxes.conf.tolist()

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

                # ROI hit logic
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
                
                # loitering (ROI 체류시간)
                if st.roi_enter_t is not None and (not st.loiter_emitted):
                    dwell = t - st.roi_enter_t
                    if dwell >= LOITER_SEC and (t - st.last_event_t) >= COOLDOWN_SEC:
                        emit(sa_events, video_id, "loitering", t, track_id, xyxy, extra={"dwell_sec": round(dwell, 3), "roi": "right_30"})
                        st.loiter_emitted = True
                        st.last_event_t = t

                tracks[track_id] = st

                # draw
                draw_box(frame, xyxy, track_id, color=(0, 255, 0))

            # 이벤트 텍스트 overlay (최근 1개만)
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