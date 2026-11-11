"""K1 smoke test (맥 CPU).
1) K1a·K1b·m1_fresh 노트북의 모든 코드 셀 문법 확인
2) K1a 문장 고르기·생성 파싱·검증·저장을 가짜 학습 응답과 가짜 API 결과로 실행
3) K1b 합성 지점 추가를 F1과 같은 LABELS 셀로 실행 (판정 지점 범주 확인)
4) K1b 최신 LLM 다시 판정을 실제 M1 파일(있으면)과 실제 F1 base 가중치로 실행 — 같은 가중치면 저장 점수가 재현돼야 함
   python experiments/builders/smoke_k1.py [M1 드라이브 파일 폴더] [F1 base 판정기 폴더]"""
import ast, json, os, random, sys, copy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, pysbd, tiktoken
import build_k1a as KA, build_k1b as KB, build_m1 as M1

# 1) 문법
for name, cells in [('k1a', KA.cells), ('k1b', KB.cells), ('m1_fresh', M1.fresh_cells())]:
    n = 0
    for c in cells:
        if c['cell_type'] != 'code': continue
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        ast.parse('\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('!'))); n += 1
    print(f'[1] {name}: 코드 셀 {n}개 문법 통과')

# 2) K1a — 가짜 학습 응답
NS = {'json': json, 'random': random, 'np': np, 'pd': pd, 'os': os, 'copy': copy, 'tiktoken': tiktoken, 'pysbd': pysbd}
src = {'s1': {'source_id': 's1', 'task_type': 'Data2txt', 'source_info': {'name': 'Cafe', 'review_info': [
    {'review_text': 'Alice: the latte was great and the waitress forgot our bill.', 'review_stars': 4},
    {'review_text': 'Bob: slow service, cold eggs.', 'review_stars': 2}]}},
       's2': {'source_id': 's2', 'task_type': 'Summary', 'source_info': 'Kim is the editor. Lee directed the service. They met in 2020.'},
       's3': {'source_id': 's3', 'task_type': 'QA', 'source_info': {'question': 'How to brew tea?', 'passages': 'Pour hot water into the teapot to warm it, then add leaves and steep three minutes.'}}}
resp = {'s1': 'The cafe gets mixed reviews from its customers overall. One reviewer said the latte was great and service was friendly enough. Another reviewer complained about slow service and cold eggs at breakfast.',
        's2': 'Kim works as the editor of the paper here. Lee directed the foreign service for many years. They met in the year 2020 in town.',
        's3': 'First pour hot water into the teapot to warm it up. Then add the tea leaves and steep for about three minutes.'}
NS['sources'] = src
exec('def build_reference(src):' + KA.DATA.split('def build_reference(src):')[1].split("TASKS = ['QA'")[0], NS)
exec("TASKS = ['QA', 'Summary', 'Data2txt']", NS)
exec(KA.PREP.split('dev_items, val_items')[0], NS)
train = [{'id': f'r{k}', 'source_id': s, 'task_type': src[s]['task_type'], 'model': 'm', 'response': resp[s], 'labels': []}
         for k, s in enumerate(['s1', 's2', 's3'])]
NS['TRAIN_ITEMS'] = TI = NS['to_items'](train)
NS['LABELS'] = {f'tr-{x["id"]}': {i + 1: {'label': 'S', 'phrase': ''} for i in range(len(x['sents']))} for x in TI}
NS['ADD_ONSET'] = {}
NS['R0P'] = {'items': {'calL': [{'ref_key': 'zz'}], 'devL': [{'ref_key': 'yy'}]}}
exec(KA.K1A_SELECT.replace('N_PER_TASK, MAX_PER_RESP, MIN_TOK, MAX_TOK = 500, 2, 8, 60', 'N_PER_TASK, MAX_PER_RESP, MIN_TOK, MAX_TOK = 5, 2, 4, 60'), NS)
assert len(NS['PICK']) == 3, NS['PICK']
exec(M1.LABEL_PROMPT + KA.K1A_GEN + KA.K1A_VERIFY + KA.K1A_EXPORT, NS)
x = NS['BY_ID']['r0']; t = NS['PICK']['r0']
p = NS['syn_params'](x, t)
assert 'TARGET sentences' in p['messages'][0]['content'] and p['system'].startswith('You create')
o1 = NS['sent_text'](x, x['sents'][t[0]])
fake = {'items': [{'i': t[0] + 1, 'contradiction': o1.replace('One reviewer', 'Two reviewers').replace('Another reviewer', 'The same reviewer').replace('The cafe', 'The diner'),
                   'changed': 'Two reviewers' if 'One reviewer' in o1 else ('The same reviewer' if 'Another' in o1 else 'The diner'), 'kind': 'attribution',
                   'paraphrase': 'In other words, ' + o1}]}
out, st = NS['parse_syn']('blah ' + json.dumps(fake), x, t)
assert st == 'ok' and {k for _, k, *_ in out} == {'contra', 'para'}, out
assert NS['parse_syn']('{"items": [{"i": 1, "contradiction": "Totally different words here now", "changed": "Totally", "paraphrase": ""}]}', x, [0])[0] == [], '최소 수정 아닌 모순을 걸러야 함'
cands, vres = {}, {}
for k, (i, kind, s, ch, kd) in enumerate(out):
    vid = f'v-r0-{i}-{kind}'
    cands[vid] = {'rid': 'r0', 'ref_key': x['ref_key'], 'task': x['task_type'], 'i': i, 'kind': kind, 'orig': o1, 'sentence': s, 'phrase': ch, 'gen_kind': kd}
    vp = NS['verify_params'](x, i, s)
    assert f'label ONLY sentence [{i + 1}]' in vp['messages'][0]['content'] and s in vp['messages'][0]['content']
    vres[vid] = {'text': json.dumps({'sentences': [{'i': i + 1, 'label': 'H' if kind == 'contra' else 'S'}]})}
assert NS['parse_verify']('{"sentences":[{"i":9,"label":"H"}]}', 0) is None
SYN, passed, n = NS['build_synth'](cands, vres)
assert passed == {'contra': 1, 'para': 1} and n == 1 and len(SYN) == 2
chk = NS['check_sample'](SYN)
assert len(chk) == 1 and chk[0]['rewritten'] != chk[0]['original']
print('[2] K1a 고르기·파싱·검증·저장 통과', {k: v[:60] for k, v in chk[0].items() if k in ('original', 'rewritten', 'changed')})

# 3) K1b 합성 지점 — F1 LABELS 셀 그대로
_k = resp['s2'].find('foreign'); train[1]['labels'] = [{'start': _k, 'end': _k + 15}]   # LABELS 셀은 환각 문장이 하나는 있어야 함
NS['train'] = train
exec(KB.F1.LABELS, NS)
n0 = len(NS['P_TXT'])
assert set(NS['P_CAT']) >= {'neg'}
NS['K1_DIR'] = '/tmp/_k1smoke'; os.makedirs(NS['K1_DIR'], exist_ok=True)
json.dump(SYN, open(os.path.join(NS['K1_DIR'], 'synth_train.json'), 'w'))
import pickle
NS['R0_DIR'] = NS['K1_DIR']; pickle.dump(NS['R0P'], open(os.path.join(NS['K1_DIR'], 'r0_items.pkl'), 'wb'))
NS['REFS'] = NS['REFS']
exec(KB.K1B_SYN, NS)
cats = list(NS['P_CAT'][n0:])
con = [c for c in SYN if c['kind'] == 'contra'][0]
pts = NS['syn_points'](con)
print('    모순 지점:', [(tx[-25:], ct) for tx, ct in pts])
assert 'win' in cats and cats.count('neg') == len(NS['syn_points']([c for c in SYN if c['kind'] == 'para'][0]))
k = next(j for j, (_, ct) in enumerate(pts) if ct != 'pre')
assert con['phrase'].split()[0] in pts[k][0], '경계가 바뀐 구절에서 시작해야 함'
print(f'[3] K1b 합성 지점 통과: +{len(cats)}개', pd.Series(cats).value_counts().to_dict())

# 4) K1b 다시 판정 — 실제 M1 파일·가중치
mdir = sys.argv[1] if len(sys.argv) > 1 else None
jdir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/Downloads/f1_final_judges/judge_base')
if not (mdir and os.path.isdir(mdir) and os.path.isdir(jdir)):
    print('[4] 건너뜀 (M1 파일 폴더·F1 base 가중치 필요)'); sys.exit(0)
import torch, time
from hallugate.judge import StudentJudge, read_config, input_mode
G = {'np': np, 'pd': pd, 'json': json, 'os': os, 'time': time, 'torch': torch, 'random': random,
     'StudentJudge': StudentJudge, 'read_config': read_config, 'input_mode': input_mode}
L3 = M1.L3
exec(f'COMBO = {M1.COMBO!r}\n' + L3.L3_INPUTS + L3.L3_CORE + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + KB.K1B_M1, G)
G['save_json'] = lambda o, p: json.dump(o, open(p, 'w'))
G['load_json'] = lambda p: json.load(open(p)) if os.path.exists(p) else None
ld = lambda n: json.load(open(os.path.join(mdir, f'{n}_devL.json')))
cases, gens, raw, old = ld('cases'), ld('gen'), ld('label'), ld('scores')
keep = [c for c in old if c in [m[0] for m in KB.M1A]][:2] + [c for c in old][:2]
old = {c: old[c] for c in keep}
labels = G['parse_labels'](gens, raw)
j = G['make_judge'](jdir, 'cpu')
d = G['parity'](j, cases, gens, old, n=2)
print(f'    기존 판정기 재현 최대 차이 {d:.5f}'); assert d < 0.1
new = G['rescore'](j, cases, gens, old, '/tmp/_k1smoke/scores_k1.json', every=2)
C = dict(M1.COMBO)
newc = G['with_combo'](new, C)
mx = max(abs(a[2] - b[2]) for c in old for a, b in zip(old[c], newc[c]))
mc = max(abs(a[4] - b[4]) for c in old for a, b in zip(old[c], newc[c]))
print(f'    같은 가중치 다시 판정: F1 점수 최대 차이 {mx:.5f}, 조합 최대 차이 {mc:.5f}'); assert mx < 0.1 and mc < 0.05
ro = G['response_rows']({c: cases[c] for c in old}, gens, labels, old, C['threshold'], 3.968751)
rn = G['response_rows']({c: cases[c] for c in old}, gens, labels, newc, C['threshold'], 3.968751)
pr = G['paired'](ro, rn)
assert (ro.flag_c.values == rn.flag_c.values).all(), '같은 가중치인데 표시가 달라짐'
sf = G['sentence_flags'](gens, newc, [m for m in KB.M1A if m[0] in old], C['threshold'])
print(f'[4] K1b 다시 판정 통과: 응답 {len(old)}, 쌍체 {pr}, M1a 문장 {len(sf)}개')
print('smoke_k1 통과')
