import json, numpy as np
from nowdance.io import load_sequence
from nowdance.normalize import normalize_frame
from nowdance.schema import POSE_LANDMARK_NAMES, LANDMARK_INDEX

seq = load_sequence('charts/jljt.json')
ts = np.array([f.timestamp for f in seq.frames])
data = json.load(open('charts/jljt_chart.json', encoding='utf-8'))
up = [LANDMARK_INDEX[n] for n in ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']]
for s in data['steps']:
    m = (ts >= s['start_time']) & (ts <= s['end_time'])
    idx = np.where(m)[0]
    fr = [seq.frames[i] for i in idx]
    if fr:
        nrm = np.stack([normalize_frame(f) for f in fr])
        tpl = np.median(nrm, axis=0)
        s['template'] = {POSE_LANDMARK_NAMES[i]: {'x':round(float(tpl[i,0]),6),'y':round(float(tpl[i,1]),6),'z':round(float(tpl[i,2]),6),'visibility':round(float(tpl[i,3]),6)} for i in range(33)}
        if len(fr) >= 4:
            vs = np.diff(np.stack(nrm)[:, up, :2], axis=0)
            sm = np.linalg.norm(vs, axis=2)
            jk = np.std(sm, axis=0)
            s['expected_jerk'] = round(float(np.mean(jk)), 4)
json.dump(data, open('charts/jljt_chart.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
print(str(len(data['steps'])) + ' steps done')
