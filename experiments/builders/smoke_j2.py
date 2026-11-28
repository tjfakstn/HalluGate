"""J2 smoke test (11/27 재설계, 맥 CPU, API·GPU·드라이브 없음).
1) 노트북 코드 셀 문법, 저장소 K2 표에서 1차 재현 수치(devM 36/191, 58/881) — build_j2.py import 때 assert
2) 가짜 K2 표·응답으로 표 만들기 → Haiku 요청 → 가짜 투표 결과 → 점수 → calM 임계값·선택 → devM 판정
   - Haiku가 환각을 잘 잡는 가짜 → 채택
   - Haiku가 모두 H(점수 1) → 2차 임계값 inf(1차만 남음) → J2-1 미달
   - 파싱 실패는 표시 안 함
python experiments/builders/smoke_j2.py"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import build_j2 as J

n = sum(1 for c in J.cells if c['cell_type'] == 'code' and ast.parse(c['source']))
print(f"[1] j2: 코드 셀 {n}개 문법 통과 | 저장소 K2 표 재현 devM {J.REPRO['devM']}, calM {J.REPRO['calM']}")
assert len(J.K2T['calM']) == 1072 and len(J.K2T['devM']) == 1072
_cell = next(c['source'] for c in J.cells if c['cell_type'] == 'code' and c['source'].startswith('K2T = '))
_g = {}; exec(_cell.split('\nimport pandas')[0], _g)          # 노트북 셀의 K2T·THR_B 줄을 실제로 실행 (11/28 false/true 오류 재발 방지)
assert _g['K2T'] == J.K2T and _g['THR_B'] == J.THR_B

rng = np.random.default_rng(0)
K2T, CASES, GENS = {}, {}, {}
for sp in ['calM', 'devM']:
    K2T[sp], CASES[sp], GENS[sp] = {}, {}, {}
    for k in range(600):
        cid = f'{sp}-{"haiku" if k % 2 else "sonnet"}-{k}-0'
        hal = rng.random() < 0.18
        mx = (3.5 if rng.random() < 0.19 else 1.0) if hal else (3.0 if rng.random() < 0.08 else 0.5)    # 1차: 환각 19%, 오경보 8%
        K2T[sp][cid] = [not hal, mx, ['QA', 'Summary', 'Data2txt'][k % 3], 'haiku' if k % 2 else 'sonnet']
        CASES[sp][cid] = {'ref_label': f'Reference {k}. It says facts.', 'ref_judge': 'x'}
        GENS[sp][cid] = {'text': f'Answer {k} first sentence. {"WRONG claim here." if hal else "Supported claim here."}'}
G = {'np': np, 'json': json, 'os': os, 'random': random}
exec(J.J2_DATA, G)
P = G['make_table'](K2T, CASES, GENS, J.THR_B)
assert len(P) == 1200 and P.ref.iloc[0].startswith('Reference') and P.flag1.dtype == bool
print(f"[2] 표 {len(P)}행 | 1차 탐지 {P[P.hal].flag1.mean():.3f}, 오경보 {P[~P.hal].flag1.mean():.3f}")

G['PRICE'] = {'claude-haiku-5-5': (0.05, 0.25), 'claude-sonnet-5-5': (1.0, 5.0), 'claude-opus-5-5': (2.0, 10.0)}
exec(J.J2_PROMPTS, G)
REQ = G['requests_all'](P)
assert len(REQ) == 5 * len(P) and all(r['model'] == 'claude-haiku-5-5' for r in REQ.values())
r0 = REQ[f'haiku-{P.id.iloc[0]}-0']
assert 'REFERENCE:\n<<<\nReference' in r0['messages'][0]['content'] and '[2] ' in r0['messages'][0]['content']
print(f"    요청 {len(REQ)}건, 예상 비용(가짜 짧은 글) ${G['est_cost'](REQ):.3f}")

def sent_json(n, labs): return json.dumps({'sentences': [{'i': j + 1, 'label': labs[j]} for j in range(n)]})
def fake_res(fn):
    RES = {}
    for r in P.itertuples():
        k = len(G['sents'](r.text))
        for v in range(5): RES[f'haiku-{r.id}-{v}'] = {'text': fn(r, v, k), 'model': 'x', 'in': 1, 'out': 1}
    return RES

good = lambda r, v, k: sent_json(k, ['S'] + ['H' if (r.hal and v < 4) or (not r.hal and v < 1 and r.Index % 7 == 0) else 'S'] * (k - 1))
RES = fake_res(good)
bad_id = P[P.hal & (P.part == 'devM')].id.iloc[0]
for v in range(5): RES[f'haiku-{bad_id}-{v}']['text'] = 'not json'
P = G['haiku_scores'](P, RES)
row = P[P.id == bad_id].iloc[0]
assert row.s_haiku_H == 0.0 and row.ok_haiku == 0.0, '파싱 실패는 표시 안 함'
P['s_minicheck'] = np.where(P.hal, rng.uniform(0.2, 1, len(P)), rng.uniform(0, 0.8, len(P)))
P['s_ens'] = P[['s_haiku_H', 's_minicheck']].mean(axis=1)

exec(J.J2_METRICS, G)
CANDS = ['haiku_H', 'haiku_HA', 'minicheck', 'ens']
SEL, THR = G['select'](P, CANDS)
R = G['judge_j2'](P, SEL, THR)
assert SEL in ('haiku_H', 'haiku_HA') and R['J2'] and R['flag'] <= 0.20 and R['det'] > R['det1'] + 0.3, (SEL, R)
print(f"[3] 선택 {SEL} | devM 1차 {R['det1']:.3f} → 결합 {R['det']:.3f} [{R['det_ci'][0]:.3f}, {R['det_ci'][1]:.3f}], 오경보 {R['flag']:.3f} → J2 {R['J2']}")

P2 = G['haiku_scores'](P.copy(), fake_res(lambda r, v, k: sent_json(k, ['H'] * k)))
SEL2, THR2 = G['select'](P2, ['haiku_H'])
R2 = G['judge_j2'](P2, SEL2, THR2)
assert THR2['haiku_H'] == float('inf') and abs(R2['det'] - R2['det1']) < 1e-12 and not R2['J2_1'] and not R2['J2']
assert G['judge_j2'](P, None, THR)['J2'] is False
print(f"[4] 모두 H → 2차 임계값 inf, 결합 = 1차 {R2['det']:.3f} → J2-1 {R2['J2_1']} | 후보 없음 → J2 미달")
print('smoke_j2 통과')
