import os, json, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional # 변수 지정
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import cv2 # 영상 읽기/쓰기 + 선/글씨/박스 그리기
import torch # GPU(CUDA) 사용 가능 여부 확인
from ultralytics import YOLO # YOLOv8 모델 로드 + 탐지/추적
from shapely.geometry import Point, Polygon

"""
Config (설정값)
"""
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("DEVICE = ", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

SOURCE = os.path.join(ROOT_DIR, "data", "138564-769988151_medium.mp4")
OUT_SA_JSON_PATH = os.path.join(ROOT_DIR, "data", "sa_ver4.json") # 이벤트 결과 저장
OUT_SA_XML_PATH = os.path.join(ROOT_DIR, "data", "sa_ver4.xml") # 이벤트 결과 저장
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
def in_roi_full_body(x1: float, x_cut: int) -> bool:
    """
    중심점 기준이 아니라 전신 기준으로 수정
    사람 박스의 왼쪽 끝(x1)이 경계선을 완전히 넘었을 때
    """
    return x1 >= x_cut

# 사람 박스의 중심 x 좌표(cx)가 x_cut보다 오른쪽이면 ROI 안이라고 판단하는 함수
# def in_roi(cx: float, x_cut: int) -> bool:
#     return cx >= x_cut

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

def sec_to_hhmmss(sec: float) -> str:
    """초 -> HH:MM:SS 문자열로 변환"""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def save_sa_xml(sa_events: list, out_path: str, video_filename: str):
    """KISA XML 형식으로 SA 파일 저장"""
    EVENT_NAME_MAP = {
        "intrusion": "Intrusion",
        "loitering": "Loitering",
        "line_crossing": "Intrusion"
    }

    root = ET.Element("KisaLibraryIndex")
    library = ET.SubElement(root, "Library")
    clip = ET.SubElement(library, "Clip")

    header = ET.SubElement(clip, "Header")
    ET.SubElement(header, "AlarmEvents").text = str(len(sa_events))
    ET.SubElement(header, "Filename").text = video_filename

    alarms = ET.SubElement(clip, "Alarms")
    for event in sa_events:
        alarm = ET.SubElement(alarms, "Alarm")
        ET.SubElement(alarm, "StartTime").text = sec_to_hhmmss(event["event_time_sec"])
        kisa_name = EVENT_NAME_MAP.get(event["event_type"], event["event_type"])
        ET.SubElement(alarm, "AlarmDescription").text = kisa_name
        ET.SubElement(alarm, "AlarmDuration").text = "00:00:00"

    raw_str = ET.tostring(root, encoding="unicode")
    pretty_str = minidom.parseString(raw_str).toprettyxml(indent="  ")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty_str)
    print(f"SA XML 저장 완료: {out_path}")

def load_map_polygons(map_path: str) -> dict:
    """
    KISA Map XML 파일을 읽어서
    { 태그이름: Polygon 객체 } 딕셔너리로 반환

    반환 예시: 
    {
        "DetectArea": Polygon([(6,475), (958,155), ...]),
        "Intrusion": Polygon([(200, 300), (800, 300), ...]),
        ...
    }
    """

    # Map 파일이 없으면 빈 딕셔너리 반환 (fallback용)
    if not os.path.exists(map_path):
        print(f"[경고] Map 파일 없음: {map_path} -> 기본 ROI 사용")
        return {}
    tree = ET.parse(map_path)
    root = tree.getroot()

    # KISA에서 정의한 영역 태그 목록
    AREA_TAGS = [
        "DetectArea", # 전체 감지 영역
        "Intrusion", # 침입 영역
        "Loitering", # 배회 영역
        "Abandonment", # 유기 영역
        "Fight", # 싸움 영역
    ]
    
    polygons = {}

    for tag in AREA_TAGS:
        # XML에서 해당 태그 찾기
        # find()는 없으면 None 반환
        element = root.find(f".//{tag}") # // = 하위 어디서든 찾기
        if element is Nons:
            continue # 이 태그가 없으면 건너뜀

        points = []
        for point_elem in element.findall("Point"):
            # "958, 155" -> ["958", "155"] -> (958, 155)
            x_str, y_str = point_elem.text.strip().split(",")
            points.append((int(x_str), int(y_str)))

        if len(points) < 3:
            print(f"[Map] {tag} 로드 완료: {len(points)}개 꼭짓점")
    return polygons

def person_in_polygon(xyxy, polygon: Polygon) -> bool:

    """
    KISA 기준: '사람의 몸 전체가 영역 안에 있을 때'
    -> 박스의 네 꼭짓점이 모두 다각형 안에 있으면 True

    [박스 꼭짓점 4개]
    (x1, y1) ---- (x2, y1)
        :           :
        :           :
    (x1, y2) ---- (x2, y2)

    4개 중 하나라도 밖에 있으면 False
    """

    x1, y1, x2, y2 = xyxy

    corners = [
        (x1, y1), # 왼쪽 위
        (x2, y1), # 오른쪽 위
        (x1, y2), # 왼쪽 위
        (x2, y2), # 오른쪽 아래
    ]
    return all(polygon.contains(Point(px, py)) for px, py in corners)

def person_entering_polygon(xyxy, polygon: Polygon) -> bool:
    """
    '진입 중' 판정
    -> 사람 중심점이 다각형 안에 있으면 True
    
    차이
    persona_in_polygon: 완전히 들어왔을 때
    person_entering_polygon: 들어오기 시작했을 때
    """

    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return polygon.contains(Point(cx, cy))


def draw_polygon_roi(frame, polygons: dict):
    """
    Map에서 읽은 다각형들을 화면에 그려줌

    색상 구분:
    - DetectArea: 파란색
    - Intrusion: 빨간색
    - Loitering: 노란색
    - 기타: 흰색
    """
    COLOR_MAP = {
        "DetectArea": (255, 100, 0),
        "Intrusion": (0, 0, 255),
        "Loitering": (0, 255, 255)
    }
    for tag, polygon in polygons.items():
        color = COLOR_MAP.get(tag, (255, 255, 255)) # 색상 결정

        pts = np.array(list(polygon.exterior.coords[:-1]), dtype=np.int32) # 좌표 변환

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2) # 선 그리기
        label_x, label_y = pts[0] # 첫 번째 꼭짓점
        cv2.putText(frame, tag, (int(label_x), int(label_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) # 태그 이름 표시


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
                x1_f = xyxy[0] # 바운딩박스 왼쪽 끝 (float)
                # st = tracks.get(track_id, TrackState()) # 새 사람이 있든 없든 매번 새로 만드는 오류 (메모리 낭비)
                # 새 사람일 때만 TrackState() 생성
                if track_id not in tracks:
                    tracks[track_id] = TrackState()
                st = tracks[track_id]
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
                # if in_roi(cx, x_cut):
                #     st.in_roi_hit += 1
                #     if st.roi_enter_t is None:
                #         st.roi_enter_t = t
                # 왼쪽 끝 경계선을 기준으로 완전히 넘었을 경우로 수정
                if in_roi_full_body(x1_f, x_cut):
                    st.in_roi_hit += 1
                    if st.roi_enter_t is None:
                        st.roi_enter_t = t
                else:
                    st.in_roi_hit = 0
                    st.roi_enter_t = None
                    st.intrusion_emitted = False # ROI 나가면 다시 감지 가능하게
                    st.loiter_emitted = False # 배회도 ROI 나가면 다시 감지 가능하게

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
                # tracks[track_id] = st

                # 박스와 ID 그리기
                draw_box(frame, xyxy, track_id, color=(0, 255, 0))

            # 최근 이벤트 1개를 화면에 텍스트로 보여주고 결과 영상을 파일로 저장
            if sa_events:
                last = sa_events[-1]
                cv2.putText(frame, f"EVENT: {last['event_type']} ID={last['track_id']} t={last['event_time_sec']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    with open(OUT_SA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sa_events, f, ensure_ascii=False, indent=2)

    save_sa_xml(sa_events, OUT_SA_XML_PATH, video_id)

    print("SA saved: ", OUT_SA_XML_PATH, "events= ", len(sa_events))
    print("VIS saved: ", OUT_VIDEO_PATH)
    print("elapsed(sec)=", round(time.time() - start_wall, 2))

        

if __name__ == "__main__":
    main()