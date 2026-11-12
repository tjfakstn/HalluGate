"""K2 smoke test (맥 CPU).
1) K2a·K2b 노트북의 모든 코드 셀 문법 확인
2) K2a 요청 복제·정답 요청·K2-0 집계를 가짜 응답으로 실행
3) K2b 최신 LLM 학습 응답 추가를 F1과 같은 LABELS 셀로 실행 (H 문장은 정답 문구부터 환각, 상한 적용)
4) K2b 임계값 B·판정 함수를 실제 M1 파일로 실행 — 새 점수 = 기존 점수면 차이 0, 임계값 B 같음
   python experiments/builders/smoke_k2.py [M1 드라이브 파일 폴더]"""
import ast, json, os, random, sys, copy, pickle, re, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, pysbd, tiktoken
import build_k2a as KA, build_k2b as KB
M1, L3 = KA.M1, KA.L3

# 1) 문법
for name, cells in [('k2a', KA.cells), ('k2b', KB.cells)]:
    n = 0
    for c in cells:
        if c['cell_type'] != 'code': continue
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        ast.parse('\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('!'))); n += 1
    print(f'[1] {name}: 코드 셀 {n}개 문법 통과')

# 2) K2a
G = {'np': np, 'pd': pd, 'json': json, 'os': os, 'random': random}
exec(L3.L3_INPUTS + M1.M1_FLOW + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + KA.K2A_PREP.split('def prepare_split')[0] + KA.K2A_CHECK, G)
base = {'train-haiku-s1': {'sid': 's1', 'task': 'QA', 'gen': 'haiku', 'flow': 'collection', 'gold_in': True, 'ref_label': 'Q', 'ref_judge': 'R'}}
ex = G['expand'](base, 2)
assert list(ex) == ['train-haiku-s1-0', 'train-haiku-s1-1'] and ex['train-haiku-s1-0'] is not ex['train-haiku-s1-1']
p = G['label_params_k2'](dict(base['train-haiku-s1'], task='QA'), 'One. Two.', 'claude-sonnet-5-5')
assert p['model'] == 'claude-sonnet-5-5' and '[2] Two.' in p['messages'][0]['content']
G['THR_C'], G['THR_F'] = 1.0, 1.0
def fake_split(sp, n, hal_every):
    cases, gens, labels, scores = {}, {}, {}, {}
    for k in range(n):
        cid = f'{sp}-haiku-s{k}-0'
        cases[cid] = {'sid': f's{k}', 'task': 'Data2txt', 'gen': 'haiku', 'flow': 'direct', 'gold_in': None, 'ref_label': '', 'ref_judge': ''}
        gens[cid] = {'text': 'A is good. B is bad.', 'stop': 'end_turn'}
        h = k % hal_every == 0
        labels[cid] = {1: {'label': 'H' if h else 'S', 'phrase': 'good'}, 2: {'label': 'S', 'phrase': ''}}
        scores[cid] = [[0, 9, 0.0, 0.0, 2.0 if h else 0.0], [11, 20, 0.0, 0.0, 0.0]]
    return cases, (gens, labels, scores)
CASES, RUN = {}, {}
for sp, n, he in [('train', 600, 1), ('calM', 500, 50), ('devM', 400, 2)]:
    CASES[sp], RUN[sp] = fake_split(sp, n, he)
CHK, ROWS = G['k2a_check'](CASES, RUN)
assert CHK['train_H_sent'] == 600 and CHK['calM_clean'] == 490 and CHK['devM_hal'] == 200 and CHK['K2_0'], CHK
# 파일럿 예상치·예산 상한
G['cost'] = lambda res: sum(r['c'] for r in res.values())
exec(KA.K2A_PILOT, G)
pg = {f'p{k}': {'c': 0.002} for k in range(10)}
pr = {f'p{k}': {'c': 0.004} for k in range(10)}
plab = {f'p{k}': {1: {'label': 'H' if k < 2 else 'S'}, 2: {'label': 'S'}} for k in range(10)}
PJ = G['pilot_projection'](None, pg, pr, plab, {'train': 4000, 'calM': 1000, 'devM': 1000})
assert abs(PJ['proj_cost']['train'] - 4000 * 0.006) < 1e-9 and abs(PJ['proj_cost']['calM'] - 1000 * 0.010) < 1e-9, PJ
assert abs(PJ['proj_train_H'] - 800) < 1e-9 and abs(PJ['proj_total'] - 44) < 1e-9, PJ
tmpd = tempfile.mkdtemp(); G['K2_DIR'] = tmpd
G['load_json'] = lambda p: json.load(open(p)) if os.path.exists(p) else None
exec(KA.K2A_RUN.split('def run_split')[0], G)
json.dump({'a': {'c': 30.0}}, open(os.path.join(tmpd, 'gen_train.json'), 'w'))
G['guard'](1000, 0.01)                                   # 30 + 10 ≤ 50
try:
    G['guard'](3000, 0.01); raise AssertionError('상한 초과인데 멈추지 않음')
except SystemExit as e:
    print('    예산 상한 작동:', str(e)[:60])
print('[2] K2a 요청 복제·정답 요청·K2-0 집계 통과', {k: CHK[k] for k in ['train_H_sent', 'calM_clean', 'devM_hal', 'K2_0']})

# 3) K2b 학습 지점 추가 — F1 LABELS 셀 그대로 (smoke_k1과 같은 가짜 학습 응답)
NS = {'json': json, 'random': random, 'np': np, 'pd': pd, 'os': os, 'copy': copy, 'tiktoken': tiktoken, 'pysbd': pysbd}
src = {'s1': {'source_id': 's1', 'task_type': 'Summary', 'source_info': 'Kim is the editor. Lee directed the service. They met in 2020.'},
       's2': {'source_id': 's2', 'task_type': 'QA', 'source_info': {'question': 'How to brew tea?', 'passages': 'Pour hot water, add leaves, steep three minutes.'}}}
NS['sources'] = src
F1 = KB.F1
exec('def build_reference(src):' + F1.DATA.split('def build_reference(src):')[1].split("TASKS = ['QA'")[0], NS)
exec(F1.PREP.split('dev_items, val_items')[0], NS)
resp = 'Kim works as the editor of the paper here. Lee directed the foreign service for many years.'
k = resp.find('foreign')
train = [{'id': 'r0', 'source_id': 's1', 'task_type': 'Summary', 'model': 'm', 'response': resp, 'labels': [{'start': k, 'end': k + 15}]}]
NS['train'] = train
NS['TRAIN_ITEMS'] = NS['to_items'](train)
NS['ITEMS'] = {f'tr-{x["id"]}': x for x in NS['TRAIN_ITEMS']}
NS['LABELS'] = {}
exec(F1.ADD, NS)
exec(F1.LABELS, NS)
n_orig = len(NS['P_TXT'])
tmp = tempfile.mkdtemp()
NS['K2_DIR'] = NS['R0_DIR'] = tmp
pickle.dump({'items': {'calL': [{'ref_key': 'zz'}], 'devL': [{'ref_key': 'yy'}]}}, open(os.path.join(tmp, 'r0_items.pkl'), 'wb'))
k2c = {'train-haiku-s1-0': {'sid': 's1', 'task': 'Summary', 'gen': 'haiku', 'flow': 'direct', 'ref_judge': NS['REFS']['s1']},
       'train-sonnet-s2-0': {'sid': 's2', 'task': 'QA', 'gen': 'sonnet', 'flow': 'collection', 'ref_judge': 'Pour hot water ... steep three minutes.'},
       'train-haiku-s2-0': {'sid': 's2', 'task': 'QA', 'gen': 'haiku', 'flow': 'collection', 'ref_judge': 'x'}}
k2g = {'train-haiku-s1-0': {'text': 'Lee is the editor of the paper. Kim directed the service.', 'stop': 'end_turn'},
       'train-sonnet-s2-0': {'text': 'Pour hot water into the pot. Then steep the leaves for ten minutes.', 'stop': 'end_turn'},
       'train-haiku-s2-0': {'text': '', 'stop': 'end_turn'}}
lab = lambda xs: {'text': json.dumps({'sentences': [{'i': i + 1, 'label': l, 'type': 'x' if l == 'H' else '', 'phrase': ph, 'reason': ''} for i, (l, ph) in enumerate(xs)]})}
k2l = {'train-haiku-s1-0': lab([('H', 'Lee is the editor'), ('S', '')]),
       'train-sonnet-s2-0': lab([('S', ''), ('H', 'ten minutes')])}
for n_, d in [('cases', k2c), ('gen', k2g), ('label', k2l)]:
    json.dump(d, open(os.path.join(tmp, f'{n_}_train.json'), 'w'))
NS['seg'] = NS['seg']
add_src = KB.K2B_ADD.replace('K2_MAX_RATIO = 1.0', 'K2_MAX_RATIO = 100.0')
exec('K2_LABEL_SRC = ' + repr(KB.K2_LABEL_SRC) + '\n' + add_src, NS)
st = NS['K2_STATS']
assert st['valid'] == 2 and st['parsed'] == 2 and st['H_sent'] == 2 and st['resp_used'] == 2, st
cats = list(NS['P_CAT'][n_orig:]); keys = NS['P_KEY'][n_orig:]; txts = NS['P_TXT'][n_orig:]
assert 'k2q:train-sonnet-s2-0' in NS['REFS'] and NS['REFS']['k2q:train-sonnet-s2-0'].startswith('Pour hot water')
# sonnet 응답: 첫 문장은 음성, 둘째 문장은 'ten minutes' 앞은 pre, 거기부터 win
s2 = [(t, c) for t, c, kk in zip(txts, cats, keys) if kk == 'k2q:train-sonnet-s2-0']
assert all(c == 'neg' for t, c in s2 if 'Then' not in t), s2
first = next((t, c) for t, c in s2 if c in ('win', 'post'))
assert 'ten' in first[0] and any(c == 'pre' for t, c in s2), s2
print('    sonnet 응답 지점:', [(t[-22:], c) for t, c in s2])
# 상한: 기존 지점 수보다 많이 못 더함
NS2 = dict(NS)
for kk in ['P_KEY', 'P_TXT', 'TR']: NS2[kk] = list(NS[kk][:n_orig]) if kk != 'TR' else list(NS['TR'][:len(train)])
NS2['P_CAT'] = np.array(NS['P_CAT'][:n_orig])
exec('K2_LABEL_SRC = ' + repr(KB.K2_LABEL_SRC) + '\n' + KB.K2B_ADD.replace('K2_MAX_RATIO = 1.0', 'K2_MAX_RATIO = 0.6'), NS2)
assert NS2['K2_STATS']['pts_used'] <= 0.6 * n_orig and NS2['K2_STATS']['resp_used'] < 2, NS2['K2_STATS']
print(f"[3] K2b 학습 지점 추가 통과: +{len(cats)}개", pd.Series(cats).value_counts().to_dict(), '| 상한 0.6 →', NS2['K2_STATS']['resp_used'], '응답')

# 4) K2b 판정 함수 — 실제 M1 파일
mdir = sys.argv[1] if len(sys.argv) > 1 else None
if not (mdir and os.path.isdir(mdir)):
    print('[4] 건너뜀 (M1 파일 폴더 필요)'); print('smoke_k2 통과'); sys.exit(0)
import time
E = {'np': np, 'pd': pd, 'json': json, 'os': os, 'time': time, 'random': random}
exec(f'COMBO = {M1.COMBO!r}\n' + L3.L3_INPUTS + L3.L3_CORE + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + KB.KB.K1B_M1 + KB.K2B_EVAL, E)
ld = lambda n: json.load(open(os.path.join(mdir, f'{n}_devL.json')))
cases, gens, raw, old = ld('cases'), ld('gen'), ld('label'), ld('scores')
labels = E['parse_labels'](gens, raw)
C = dict(M1.COMBO)
R = {'cases': cases, 'gens': gens, 'labels': labels, 'old': E['with_combo'](old, C), 'new': E['with_combo'](old, C)}
RES = {'devM': R, 'calM': R}
E['THR_F'] = {'old': 3.968751, 'new': 3.968751}
tb = E['conformal'](E['clean_max'](R, 'old'))
E['THR'] = {'old': {'A': C['threshold'], 'B': tb}, 'new': {'A': C['threshold'], 'B': E['conformal'](E['clean_max'](R, 'new'))}}
assert abs(E['THR']['old']['B'] - E['THR']['new']['B']) < 1e-9
P, rep, k21, k22 = E['k2_decide'](RES)
assert P['delta'] == 0 and not k21, P
ra = E['report'](E['rows_of'](R, 'old', 'A'))
assert abs(ra['clean_combo']['rate'] - 0.0449) < 0.002, ra['clean_combo']           # M1 기록 4.5% 재현
rb = E['report'](E['rows_of'](R, 'old', 'B'))
print(f"[4] K2b 판정 통과: 임계값 A {C['threshold']:.3f} → 표시율 {ra['clean_combo']['rate']:.3f}, 탐지 {ra['hal_any']['rate']:.3f} | "
      f"같은 세트로 맞춘 B {tb:.3f} → 표시율 {rb['clean_combo']['rate']:.3f}, 탐지 {rb['hal_any']['rate']:.3f} | 쌍체 Δ {P['delta']}")
print('smoke_k2 통과')
