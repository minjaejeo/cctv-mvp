import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
import os


"""
Config
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 이 파이썬 파일의 경로
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))


GT_PATH = os.path.join(ROOT_DIR, "data", "gt.json")
# GT_PATH = 'gt.json' # 정답 (json 또는 xml)
# SA_PATH = 'sa.json' # 내 결과
# SA_PATH = 'sa_ver2.json' # 내 결과
SA_PATH = os.path.join(ROOT_DIR, "data", "sa_ver4.json")
# REPORT_PAHT = 'report.txt' # 보고서 저장 경로
REPORT_PAHT = os.path.join(ROOT_DIR, "data", "report.txt")

# KISA 기준: 시간 오차 허용 범위 -2초 ~ +10초
EARLY = 2.0
LATE = 10.0

# KISA 합격 기준
PASS_F1 = 90.0 # F1 90점 이상


"""
파일 로드
"""
def load_json(path: str) -> list:
    """JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}는 list(JSON 배열)여야 합니다.")
    return data

def load_kisa_xml(path: str) -> list:
    """
    KISA GT XML 파일을 읽어서 JSON 형식 리스트로 변환

    XML 형식:
    <Alarm>
        <StartTime>00:03:00</StartTime>
        <AlarmDescription>Intrusion</AlarmDescription>
    </Alarm>
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # 파일명 추출 (video_id로 사용)
    filename_elem = root.find(".//Filename")
    video_id = filename_elem.text if filename_elem is not None else "unknown"

    # 이벤트 파싱
    EVENT_NAME_MAP = {
        "Intrusion": "intrusion",
        "Loitering": "loitering",
    }

    events = []
    for alarm in root.findall(".//Alarm"):
        start_time = alarm.find("StartTime").text # "00:03:00"
        description = alarm.find("AlarmDescription").text

        # HH:MM:SS -> 초 변환
        h, m, s = map(int, start_time.split(':'))
        t_sec = h * 3600 + m * 60 + s

        event_type = EVENT_NAME_MAP.get(description, description.lower())
        
        events.append({
            "video_id": video_id,
            "event_type": event_type,
            "event_time_sec": float(t_sec),
        })

    return events

def load_events(path: str) -> list:
    """
    확장자에 따라 자동으로 JSON / XML 선택해서 로드
    """
    ext = os.path.splitext(path)[1].lower() # ".json" 또는 ".xml"
    if ext == ".json":
        return load_json(path)
    elif ext == ".xml":
        return load_kisa_xml(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

def load_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}는 list(JSON 배열)여야 합니다")
    return data

"""
F1 계산
"""
def f1_score(tp: int, fp: int, fn: int):
    """
    Precision, Recall, F1 계산
    Precision (정밀도): 내가 찍은 것 중 맞춘 비율
                        TP / (TP + FP)
    Recall (재현율): 정답 중 내가 맞춘 비율
                    TP / (TP + FN)
    F1: Precision과 Recall의 조화 평균
                2 * P * R / (P + R)
    """
    p = tp / (tp + fp) if (tp + fp) else 0.0 # Precision(정밀도): 찍은 것 중에 맞춘 비율
    r = tp / (tp + fn) if (tp + fn) else 0.0 # Recall (재현율): 정답 중 맞춘 비율
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0 # F1 Score: Precision과 Recall의 조화 평균
    return p, r, f1

def match_events(gt_events, sa_events, early_sec, late_sec):

    """
    GT 이벤트와 SA 이벤트를 매칭해서 TP/FP/FN 분류

    매칭 기준:
    - 같은 video_id + event_type
    - 시간 오차가 -early_sec ~ +late_sec 범위 안
    - GT 1개당 SA 1개만 매칭 (가장 가까운 것)
    """
    gt_by_key = defaultdict(list)
    sa_by_key = defaultdict(list)

    # 1) (video_id, event_type)로 그룹핑
    for g in gt_events:
        gt_by_key[(g["video_id"], g["event_type"])].append(g)
    for s in sa_events:
        sa_by_key[(s["video_id"], s["event_type"])].append(s)

    tp_pairs = []
    fp = []
    fn = []

    # 2) GT가 있는 그룹부터 하나씩 평가
    for key, g_list in gt_by_key.items():
        s_list = sa_by_key.get(key, [])

        # 시간순 정렬
        g_list = sorted(g_list, key=lambda x: x["event_time_sec"])
        s_list = sorted(s_list, key=lambda x: x["event_time_sec"])

        used_sa = set()

        # 3) GT 하나당 SA 하나만 매칭(가장 가까운 것)
        for g in g_list:
            best_j = None
            best_abs_dt = None

            for j, s in enumerate(s_list):
                if j in used_sa:
                    continue

                dt = s["event_time_sec"] - g["event_time_sec"]
                if -early_sec <= dt <= late_sec:
                    abs_dt = abs(dt)
                    if best_abs_dt is None or abs_dt < best_abs_dt:
                        best_abs_dt = abs_dt
                        best_j = j

            if best_j is not None:
                used_sa.add(best_j)
                tp_pairs.append((g, s_list[best_j]))
            else:
                fn.append(g)

        # 4) 매칭되지 못한 SA는 FP
        for j, s in enumerate(s_list):
            if j not in used_sa:
                fp.append(s)

    # 5) GT에 없는 그룹에서 나온 SA는 전부 FP
    for key, s_list in sa_by_key.items():
        if key not in gt_by_key:
            fp.extend(s_list)

    return tp_pairs, fp, fn

"""
테스트용 GT 자동 생성
"""
def generate_test_gt(sa_events: list, output_path: str = "gt_test.json"):
    """
    SA 결과를 기반으로 테스트용 GT 자동 생성
    SA 이벤트 시간을 정답으로 쓰되, 약간 다르게 만들어서 테스트

    사용 목적: 
    - 실제 GT 가 없을 때 코드 동작 확인용
    - F1 계산 로직이 올바른지 검증용

    [생성 규칙]
    - SA 이벤트를 그대로 GT로 씀 -> F1=100점 나와야 정상
    - 이후 GT를 수동으로 조금 수정해서 테스트
    """
    gt_events = []
    for e in sa_events:
        gt_events.append({
            "video_id": e["video_id"],
            "event_type": e["event_type"],
            "event_time_sec": e["event_time_sec"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gt_events, f, ensure_ascii=False, indent=2)

    print(f"테스트 GT 생성 완료: {output_path} ({len(gt_events)}개 이벤트)")
    print(" SA와 동일한 내용이므로 F1=100점이 나와야 정상입니다.")
    return gt_events

"""
보고서 저장
"""
def save_report(lines: list, path: str):
    """
    결과를 txt 파일로 저장
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".json(lines))
    print(f"\n보고서 저장 완료: {path}")

"""
Main
"""
def main():
    # gt = load_list(GT_PATH)
    # sa = load_list(SA_PATH)
    gt = load_events(GT_PATH)
    sa = load_events(SA_PATH)

    print(f"GT 이벤트 수: {len(gt)}개")
    print(f"SA 이벤트 수: {len(sa)}개")

    # 매칭
    tp_pairs, fp_list, fn_list = match_events(gt, sa, EARLY, LATE)

    tp = len(tp_pairs)
    fp = len(fp_list)
    fn = len(fn_list)

    p, r, f1 = f1_score(tp, fp, fn)
    f1_100 = f1 * 100
    passed = " 합격" if f1_100 >= PASS_F1 else "불합격"

    print("==== Overall ====")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision={p:.3f}, Recall={r:.3f}, F1={f1*100:.2f}")

    # 이벤트 타입별 리포트
    # (TP는 pair의 GT event_type 기준으로 카운트)
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for g, s in tp_pairs:
        per_type[g["event_type"]]["tp"] += 1
    for s in fp_list:
        per_type[s["event_type"]]["fp"] += 1
    for g in fn_list:
        per_type[g["event_type"]]["fn"] += 1

    # print("\n===== By event_type =====")
    # for et, c in sorted(per_type.items()):
    #     p2, r2, f12 = f1_score(c["tp"], c["fp"], c["fn"])
    #     print(f"- {et}: TP={c['tp']}, FP={c['fp']}, FN={c['fn']} | "
    #           f"P={p2:.3f}, R={r2:.3f}, F1={f12*100:.2f}")

    # --- 보고서 내용 구성 ---
    lines = []
    lines.append("=" * 55)
    lines.append(" KISA 지능형 CCTV 침입 감지 성능 평가 보고서 ")
    lines.append(f" 평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 55)

    lines.append("")
    lines.append("[1] 파일 정보")
    lines.append(f" GT 파일 : {os.path.basename(GT_PATH)}  ({len(gt)}개 이벤트)")
    lines.append(f" SA 파일 : {os.path.basename(SA_PATH)} ({len(sa)}개 이벤트)")
    lines.append(f" 허용오차: GT 기준 -{EARLY}초 ~ + {LATE}초")
    lines.append(f" 합격기준: F1 {PASS_F1}점 이상")

    lines.append("")
    lines.append("[2] 전체 결과")
    lines.append(f" TP (정확히 감지) : {tp}개")
    lines.append(f" FP (오탐, 잘못감지): {fp}개 <- GT에 없는데 찍힌 것")
    lines.append(f" FN (미탐, 놓침) : {fn}개 <- GT에 있는데 못 찍은 것")
    lines.append(f" Precision : {p:.3f} (찍은 것 중 맞춘 비율)")
    lines.append(f" Recall : {r:.3f} (정답 중 맞춘 비율)")
    lines.append(f" F1 Score : {f1_100:.2f}점 / 100점")
    lines.append(f" 최종 결과 : {passed}")

    lines.append("")
    lines.append("[3] 이벤트 타입별 결과")
    for et, c in sorted(per_type.items()):
        p2, r2, f12 = f1_score(c["tp"], c["fp"], c["fn"])
        passed2 = "pass" if f12 * 100 >= PASS_F1 else "fail"
        lines.append(f" [{et}]")
        lines.append(f" TP={c['tp']} FP={c['fp']} FN={c['fn']}")
        lines.append(f" Precision={p2:.3f} Recall={r2:.3f}"
                     f"F1={f12*100:.2f}점 {passed2}")
        
    lines.append("")
    lines.append("[4] 매칭 상세 내역")

    lines.append(" TP - 정확히 감지한 이벤트")
    if tp_pairs:
        for g, s in tp_pairs:
            dt = s["event_time_sec"] - g["event_time_sec"]
            lines.append(f" [{g['event_type']}]"
                         f"GT={g['event_time_sec']:.3f}s"
                         f"SA={s['event_time_sec']:.3f}s"
                         f"오차={dt:+.3f}s")
    else:
        lines.append("없음")

    lines.append(" FP - 잘못 감지한 이벤트 (없어야 했는데 찍힌 것)")
    if fp_list:
        for s in fp_list:
            lines.append(f" [{s['event_type']}]"
                         f"SA={s['event_time_sec']:.3f}"
                         f"video={s['video_id']}")
    else:
        lines.append("없음")

    lines.append(" FN - 놓친 이벤트 (찍어야 했는데 못 찍은 것)")
    if fn_list:
        for g in fn_list:
            lines.append(f" [{g['event_type']}]"
                         f"GT={g['event_time_sec']:.3f}s"
                         f"video={g['video_id']}")
    else:
        lines.append("없음")

    lines.append("")
    lines.append("[5] 개선 방향 제안")
    if fp > 0:
        lines.append(f" - FP {fp}개: 오탐이 있습니다.")
        lines.append(f" -> COOLDOWN_SEC 값을 높이거나")
        lines.append(f" HIT_FRAMES 값을 높여서 민감도를 낮춰보세요.")
    if fn > 0:
        lines.append(f" - FN {fn}개: 미탐이 있습니다.")
        lines.append(f" -> CONF_THRES 값을 낮추거나")
        lines.append(f" HIT_FRAMES 값을 낮춰서 민감도를 높여보세요.")
    if fp == 0 and fn == 0:
        lines.append(" - FP/FN 모두 없음. 현재 설정값이 최적입니다!")

    lines.append("")
    lines.append("=" * 50)

    # 터미널 출력
    for line in lines:
        print(line)



    

        

if __name__ == "__main__":
    main()




