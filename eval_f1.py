import json

GT_PATH = 'gt.json'
SA_PATH = 'sa.json'

# 정상검출 시간 윈도우 (예: GT 기준으로 -2초에서 +10초 허용) 나중에 변경 가능
EARLY = 2.0
LATE = 10.0

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
gt = load(GT_PATH)
sa = load(SA_PATH)

# 단순 매칭: 같은 event_type/video_id에 대해 시간 윈도우 내면 TP
tp = 0 # True Positive (맞춘 개수)
fp = 0 # False Positive: SA에 있지만 GT에 없는 경우 (오검출)
fn = 0 # False Negative: GT에 있지만 SA에 없는 경우 (미검출)

matched_sa = set() # 이미 매칭된 SA 인덱스 기록 (중복 매칭 방지)

for i, g in enumerate(gt):
    found = False
    for j, s in enumerate(sa):
        if j in matched_sa:
            continue
        if s["video_id"] != g["video_id"]:
            continue
        if s["event_type"] != g["event_type"]:
            continue

        dt = s["event_time_sec"] - g["event_time_sec"]
        if (-EARLY <= dt <= LATE):
            tp += 1
            matched_sa.add(j)
            found = True
            break
        
    if not found:
        fn += 1
    
# 남은 SA는 FP
fp = len(sa) - len(matched_sa)

p = tp / (tp + fp) if (tp + fp) else 0.0 # Precision(정밀도): 찍은 것 중에 맞춘 비율
r = tp / (tp + fn) if (tp + fn) else 0.0 # Recall (재현율): 정답 중 맞춘 비율
f1 = (2*p*r/(p+r)) if (p+r) else 0.0 # F1 Score: Precision과 Recall의 조화 평균

print(f"TP: {tp}, FP: {fp}, FN: {fn}")
print(f"Precision={p: .3f}, Recall={r:.3f}, F1={f1*100:.2f}")