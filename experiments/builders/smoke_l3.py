"""L3 노트북의 입력 구성·측정 함수를 맥 CPU에서 실제 판정기 가중치로 돌려 보는 smoke test (지연 수치는 의미 없음).
가중치 폴더: F_DIR(F1 base), I_DIR(I2). 없으면 건너뛴다.
실행: python experiments/builders/smoke_l3.py [F_DIR] [I_DIR]"""
import importlib.util, os, sys, json, time, random
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('build_l3', os.path.join(HERE, 'build_l3.py'))
B = importlib.util.module_from_spec(spec); _argv = sys.argv; spec.loader.exec_module(B); sys.argv = _argv
from hallugate.judge import StudentJudge, read_config, input_mode

g = {'np': np, 'torch': torch, 'json': json, 'time': time, 'random': random, 'COMBO': B.COMBO,
     'StudentJudge': StudentJudge, 'read_config': read_config, 'input_mode': input_mode}
exec(B.L3_INPUTS, g); exec(B.L3_CORE, g)

# 1) 지점: 문장마다 8토큰째 + 문장 끝, 지점 텍스트는 문장 안
text = ('The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889 for the World Fair. '
        'It is 330 metres tall and was the tallest structure in the world for 41 years. Short one.')
pts = g['points_of'](text)
sents = [s for s in text.split('. ')]
assert pts[-1][1] == len(text) and all(text[a] != ' ' for a, _ in pts)
assert sum(1 for a, c in pts if c == len(text)) == 1
print('지점', [(text[a:c][:25], c - a) for a, c in pts])
fake_resp = [{'split': 'train', 'source_id': 's1', 'response': text}, {'split': 'test', 'source_id': 's1', 'response': 'NO'}]
fake_src = {'s1': {'task_type': 'QA', 'source_info': {'question': 'How tall?', 'passages': 'The Eiffel Tower is 330 metres tall. ' * 30}}}
inputs = g['make_inputs'](fake_resp, fake_src, n_resp=5, n_pts=50)
assert inputs and all('NO' not in x['text'] for x in inputs), 'test 분할 응답이 섞임'
assert all(x['prior'] + x['text'] in text for x in inputs)

# 2) 조합식: I3 표준화 평균 (상수 그대로)
c = B.COMBO
assert abs(g['combine'](c['F_mean'], c['I_mean'])) < 1e-9
assert abs(g['combine'](c['F_mean'] + c['F_std'], c['I_mean'] + c['I_std']) - 1.0) < 1e-9

# 3) 실제 가중치로 차례로·동시 방식 점수 일치
F_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Downloads/f1_final_judges/judge_base')
I_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/Downloads/i2_judge/judge_i2')
if os.path.exists(F_DIR) and os.path.exists(I_DIR):
    jF, jI = g['make_judge'](F_DIR, 'cpu'), g['make_judge'](I_DIR, 'cpu')
    assert jF.mode == 'retrieval' and jI.mode == 'full'
    refs = {x['ref']: (jF.prepare(x['ref']), jI.prepare(x['ref'])) for x in inputs}
    xs = inputs[:4]
    ms_s, sc_s = g['bench'](g['judge_seq'], jF, jI, xs, refs, warm=1)
    ms_p, sc_p = g['bench'](g['judge_par'], jF, jI, xs, refs, warm=1)
    d = float(np.abs(sc_s[:, :2] - sc_p[:, :2]).max())
    assert d < 1e-3, d
    one = g['bench_one'](jF, xs, refs, 0, warm=1)
    assert len(one) == len(xs)
    print('점수(F1, I2, 조합)', np.round(sc_s, 2).tolist(), '| 차례로·동시 최대 차이', d)
else:
    print('가중치 폴더 없음 — 실제 모델 확인은 건너뜀')
print('smoke_l3 통과')
