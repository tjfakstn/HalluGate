"""I3 노트북의 새 셀(지점 번호 맞추기·표준화·조합·비교·보고)을 가짜 데이터로 끝까지 돌려 보는 smoke test.
노트북과 같은 코드 문자열을 build_i3.py에서 가져와 exec한다. 네트워크·모델 없음. 실행: python experiments/builders/smoke_i3.py"""
import importlib.util, os, random, copy, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('build_i3', os.path.join(HERE, 'build_i3.py'))
B = importlib.util.module_from_spec(spec); _argv = sys.argv; spec.loader.exec_module(B); sys.argv = _argv

rng = random.Random(0)
WORDS = 'river city bloom india museum bridge north train school price year market festival temple'.split()
TASKS = ['QA', 'Summary', 'Data2txt']
g = {'np': np, 'random': random, 'copy': copy, 'pd': pd, 'Z0': {}, 'TASKS': TASKS}
exec(B.X1.SERIES, g); exec(B.X1.EVAL_DEFS, g); exec(B.X1.CONF, g); exec(B.I2_EVAL + B.I3_CORE, g)

def fake_item(i, hall):
    sents_txt = [' '.join(rng.choice(WORDS) for _ in range(n)) + '.' for n in (9, 14, 11)]
    text = ' '.join(sents_txt)
    ts, te, pos = [], [], 0
    for w in text.split(' '):
        s = text.index(w, pos); ts.append(s); te.append(s + len(w)); pos = s + len(w)
    sents, t = [], 0
    for st in sents_txt:
        n = len(st.split(' ')); sents.append({'a': ts[t], 'b': te[t + n - 1], 't0': t, 't1': t + n - 1, 'y': 0}); t += n
    onset = None
    if hall: onset = sents[1]['t0'] + 3; sents[1]['y'] = 1
    sent_of = {tt: sn['t0'] for sn in sents for tt in range(sn['t0'], sn['t1'] + 1)}
    return {'id': f'x{i}', 'task_type': TASKS[i % 3], 'ref_key': f'r{i % 4}', 'ts': ts, 'te': te, 'n_tok': len(ts),
            'onset': onset, 'sents': sents, 'sent_of': sent_of, 'ends': {sn['t1'] for sn in sents}}, text

items, RESP = [], {}
for i in range(240):
    x, text = fake_item(i, i % 2 == 0); items.append(x); RESP[x['id']] = text
# F1 base(질의 번호)는 Summary 이외에서, I2(평가 지점 번호)는 Summary에서 환각을 잘 본다
fb = []
for x in items:
    x['pts2'] = []
    for sn in x['sents']:
        for t in range(sn['t0'], sn['t1'] + 1):
            if (t - sn['t0'] + 1) % 2 == 0 or t == sn['t1']:
                hot = x['onset'] is not None and sn['y'] and t >= x['onset']
                x['pts2'].append((t, len(fb))); fb.append(rng.gauss(3 if hot and x['task_type'] != 'Summary' else 0, 1))
g['SC']['F_base'] = np.array(fb, dtype=np.float32)
CAL, DEV = items[:120], items[120:]
E_KEYS, E_TEXTS, E_NEED, E_MAP = g['build_eval_points'](CAL + DEV, RESP)
i2 = np.full(len(E_KEYS), np.nan, dtype=np.float32)
for x in CAL + DEV:
    for t, _ in x['pts2']:
        e = E_MAP[(x['id'], t)]
        if E_NEED[e]:
            sn = next(s for s in x['sents'] if s['t0'] <= t <= s['t1'])
            hot = x['onset'] is not None and sn['y'] and t >= x['onset']
            i2[e] = 10 + 4 * rng.gauss(3 if hot and x['task_type'] != 'Data2txt' else 0, 1)   # 크기가 다른 점수
g['SC']['I2'] = i2
g['SC']['Fb'] = g['to_E'](CAL + DEV, 'F_base', E_MAP, len(E_KEYS))
LS = {'orig': (g['remap'](CAL, E_MAP), g['remap'](DEV, E_MAP))}
CALo = LS['orig'][0]
e8 = [e for x in CALo for t, e in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % 8 == 0]
assert np.isfinite(g['SC']['Fb'][e8]).all() and np.isfinite(g['SC']['I2'][e8]).all()
x0 = CAL[0]; t0, q0 = x0['pts2'][3]
assert g['SC']['Fb'][E_MAP[(x0['id'], t0)]] == g['SC']['F_base'][q0]

STATS = {m: g['clean_stats'](CALo, f'{m}|8') for m in ['Fb', 'I2']}
for m, (mu, sd) in STATS.items():
    z = (g['SC'][m] - mu) / sd
    g['SC'][f'z{m}'] = z
    mu2, sd2 = g['clean_stats'](CALo, f'z{m}|8')
    assert abs(mu2) < 1e-4 and abs(sd2 - 1) < 1e-4, '표준화 후 멀쩡한 응답 점수가 평균 0·표준편차 1이 아님'
g['SC']['MEAN'] = (g['SC']['zFb'] + g['SC']['zI2']) / 2
g['SC']['MAX'] = np.maximum(g['SC']['zFb'], g['SC']['zI2'])
for task in [None] + TASKS:
    r = g['compare_e'](CALo, 'MEAN', 'Fb', task=task, n_boot=300)
    assert r['ci_lo'] <= r['delta'] <= r['ci_hi'] and r['n_hall'] > 0
    print(task or 'all', {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
rs = g['compare_e'](CALo, 'MEAN', 'Fb', task='Summary', n_boot=300)
assert rs['delta'] > 0.1, '가짜 데이터에서 조합이 Summary를 올려야 함'
rep = g['report'](*LS['orig'], 'MEAN')
assert [r['task'] for r in rep] == ['all'] + TASKS and all(0 <= r['clean_flagged'] <= 1 for r in rep)
print('smoke_i3 통과')
