"""J2 smoke test (맥 CPU, API·GPU·실제 데이터 없음).
1) 노트북 코드 셀 문법
2) 가짜 LLM-AggreFact parquet(RAGTruth 포함 11개 데이터셋) → RAGTruth가 읽히지 않는지, 표본·dev/test 분리
3) 가짜 API 결과로 점수·임계값·선택·판정까지 노트북과 같은 J2_* 코드 실행
   - opus_atom = 정답을 거의 맞히는 가짜 → 선택·채택
   - haiku = 모두 H(점수 1) → 오경보 20%를 못 맞춰 임계값 inf, 탐지 0
   - 파싱 실패는 표시 안 함
python experiments/builders/smoke_j2.py"""
import ast, json, os, random, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import build_j2 as J

n = 0
for c in J.cells:
    if c['cell_type'] != 'code': continue
    ast.parse(c['source']); n += 1
print(f'[1] j2: 코드 셀 {n}개 문법 통과')

tmp = tempfile.mkdtemp(); rng = np.random.default_rng(0)
names = ['AggreFact-CNN', 'AggreFact-XSum', 'TofuEval-MediaS', 'TofuEval-MeetB', 'Wice', 'Reveal', 'ClaimVerify', 'FactCheck-GPT', 'ExpertQA', 'Lfqa', 'RAGTruth']
for split, m in [('dev', 80), ('test', 130)]:
    rows = [{'dataset': ds, 'doc': f'{split} doc {ds} {k}. It says things.', 'claim': f'{split} claim {ds} {k}. Second sentence.',
             'label': int(rng.random() < 0.6), 'contamination_identifier': 'x'} for ds in names for k in range(m)]
    pd.DataFrame(rows).to_parquet(os.path.join(tmp, f'{split}.parquet'))
G = {'np': np, 'json': json, 'os': os, 'random': random}
exec(J.J2_DATA, G)
dv, ts = G['load_split'](os.path.join(tmp, 'dev.parquet')), G['load_split'](os.path.join(tmp, 'test.parquet'))
assert 'RAGTruth' not in set(dv.dataset) | set(ts.dataset) and dv.dataset.nunique() == 10
P = G['make_parts'](dv, ts)
assert len(P[P.part == 'dev']) == 600 and len(P[P.part == 'test']) == 1000
assert G['make_parts'](dv, ts).id.tolist() == P.id.tolist() and (G['make_parts'](dv, ts).claim == P.claim).all(), '표본이 재현되지 않음'
print(f'[2] RAGTruth 0행, dev {len(P[P.part == "dev"])}, test {len(P[P.part == "test"])}, 환각 비율 {P.hal.mean():.2f}')

G['PRICE'] = {'claude-haiku-5-5': (0.05, 0.25), 'claude-sonnet-5-5': (1.0, 5.0), 'claude-opus-5-5': (2.0, 10.0)}
exec(J.J2_PROMPTS, G)
REQ = G['requests_all'](P)
assert len(REQ['haiku']) == 5 * len(P) and len(REQ['sonnet']) == 3 * len(P) and len(REQ['opus_atom']) == len(P)
r0 = REQ['opus_atom'][f'opus_atom-{P.id.iloc[0]}']
assert r0['model'] == 'claude-opus-5-5' and 'REFERENCE:' in r0['messages'][0]['content'] and r0['system'] == G['SYS_ATOM']
print('    예상 비용(가짜 짧은 문서) $', {m: round(G['est_cost'](r, G['OUT_TOK'][m]), 3) for m, r in REQ.items()})

def sent_json(n, lab): return json.dumps({'sentences': [{'i': j + 1, 'label': lab} for j in range(n)]})
RES = {'haiku': {}, 'sonnet': {}, 'opus_atom': {}}
for r in P.itertuples():
    k = len(G['sents'](r.claim))
    for v in range(5): RES['haiku'][f'haiku-{r.id}-{v}'] = {'text': sent_json(k, 'H'), 'model': 'x', 'in': 1, 'out': 1}
    for v in range(3):                                  # sonnet: 환각이면 3표 중 2표 H, 멀쩡하면 1표 H
        lab = 'H' if (v < (2 if r.hal else 1)) else 'S'
        RES['sonnet'][f'sonnet-{r.id}-{v}'] = {'text': sent_json(k, lab), 'model': 'x', 'in': 1, 'out': 1}
    p = 90 if r.hal else 10
    RES['opus_atom'][f'opus_atom-{r.id}'] = {'text': json.dumps({'facts': [{'fact': 'f', 'verdict': 'not_found' if r.hal else 'supported'}], 'p_unsupported': p}), 'model': 'x', 'in': 1, 'out': 1}
bad = P[P.hal & (P.part == 'test')].id.iloc[0]; RES['opus_atom'][f'opus_atom-{bad}']['text'] = 'not json'
P = G['claude_scores'](P, RES)
row = P[P.id == bad].iloc[0]
assert row.s_opus_atom == 0.0 and row.ok_opus_atom == 0.0 and not row.opus_atom_anybad, '파싱 실패는 표시 안 함'
assert (P.s_haiku_H == 1.0).all() and set(P.s_sonnet_H.round(3)) == {0.333, 0.667}
P['s_minicheck'] = np.where(P.hal, rng.uniform(0.3, 1, len(P)), rng.uniform(0, 0.7, len(P)))

exec(J.J2_METRICS, G)
T_ = G['thr_at']
assert T_([1.0] * 10, [1.0] * 12) == float('inf') and T_([0] * 8 + [1, 1], [0, 1]) == 1.0 and T_([0, 0, 0, 1, 1], [0, 1]) == float('inf') and T_([0.1] * 5, [0.1, 0.9]) == 0.9
P['s_ens'] = P[['s_sonnet_H', 's_opus_atom', 's_minicheck']].mean(axis=1)
CANDS = ['haiku_H', 'haiku_HA', 'minicheck', 'sonnet_H', 'sonnet_HA', 'opus_atom', 'ens']
SEL, THR = G['select'](P, CANDS)
assert THR['haiku_H'] == float('inf') and THR['sonnet_H'] > 0.6 and SEL == 'sonnet_H', (SEL, THR)                  # sonnet_H·sonnet_HA·opus_atom·ens 모두 dev 탐지 1 → 동점 규칙상 가장 싼 sonnet_H
R = G['judge_j2'](P, SEL, THR)
assert R['J2'] and R['det'] > 0.95 and R['flag'] <= 0.20 and R['det_ci'][0] > 0.9 and len(R['by_dataset']) == 10, R
print(f"[3] 선택 {SEL} | test 탐지 {R['det']:.3f} [{R['det_ci'][0]:.3f}, {R['det_ci'][1]:.3f}] 표시 {R['flag']:.3f} → J2 {R['J2']}")
assert G['judge_j2'](P, None, THR)['J2'] is False
print('[4] 후보 없음 → J2 미달 확인')
print('smoke_j2 통과')
