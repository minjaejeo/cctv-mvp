import json
from collections import defaultdict

GT_PATH = 'gt.json' # 정답
# SA_PATH = 'sa.json' # 내 결과
SA_PATH = 'sa_ver2.json' # 내 결과

# 시간 오차 허용 범위 -2초 ~ +10초
EARLY = 2.0
LATE = 10.0


def load_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}는 list(JSON 배열)여야 합니다")
    return data

def f1_score(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0 # Precision(정밀도): 찍은 것 중에 맞춘 비율
    r = tp / (tp + fn) if (tp + fn) else 0.0 # Recall (재현율): 정답 중 맞춘 비율
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0 # F1 Score: Precision과 Recall의 조화 평균
    return p, r, f1

def match_events(gt_events, sa_events, early_sec, late_sec):
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

def main():
    gt = load_list(GT_PATH)
    sa = load_list(SA_PATH)

    tp_pairs, fp_list, fn_list = match_events(gt, sa, EARLY, LATE)

    tp = len(tp_pairs)
    fp = len(fp_list)
    fn = len(fn_list)

    p, r, f1 = f1_score(tp, fp, fn)

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

    print("\n===== By event_type =====")
    for et, c in sorted(per_type.items()):
        p2, r2, f12 = f1_score(c["tp"], c["fp"], c["fn"])
        print(f"- {et}: TP={c['tp']}, FP={c['fp']}, FN={c['fn']} | "
              f"P={p2:.3f}, R={r2:.3f}, F1={f12*100:.2f}")
        

if __name__ == "__main__":
    main()




