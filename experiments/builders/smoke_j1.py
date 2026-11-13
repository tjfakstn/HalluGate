"""J1 smoke test (맥 CPU, API·GPU 없음).
1) 노트북 코드 셀 문법
2) 실제 FaithBench 데이터 + 가짜 X2 점수 + 가짜 API 결과로 노트북과 같은 J1_* 코드를 끝까지 실행
   - haiku = 정답을 그대로 맞히는 가짜(탐지 1, 표시 0) → 선택·J1 채택이 나와야 함
   - sonnet = 모든 문장 H(표시율 1) → J-cal에서 탈락해야 함
   - 파싱 실패 응답은 표시 안 함으로 세는지, J-cal·J-test 근거가 겹치지 않는지 확인
   python experiments/builders/smoke_j1.py <FaithBench 폴더>"""
import ast, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import build_j1 as J
X2, M1, L3 = J.X2, J.M1, J.L3

n = 0
for c in J.cells:
    if c['cell_type'] != 'code': continue
    ast.parse(c['source']); n += 1
print(f'[1] j1: 코드 셀 {n}개 문법 통과')

fbdir = sys.argv[1]
G = {'np': np, 'json': json, 'os': os, 'random': random}       # pd는 셀이 직접 import해야 함
exec(L3.L3_INPUTS + X2.X2_DATA.replace("glob.glob('FaithBench/", f"glob.glob('{fbdir}/") + X2.X2_METRICS + J.J1_SPLIT, G)
FB = G['FB']; assert len(FB) == 750
rng = np.random.default_rng(0)
SC = {}
for i, s in enumerate(FB):                                     # 가짜 조합 점수: 환각이면 조금 높게 (판별력 약함)
    y = G['fb_label'](s) == 3
    SC[str(i)] = [[0, 5, 0.0, 0.0, float(rng.normal(0.5 if y else 0.0)), 0.0]]
X2R = G['x2_rows'](FB, SC, {'c': 1.5}, {})
X2R['flag_cB'] = [p[0][4] >= 1.0 for p in (SC[str(i)] for i in X2R.i)]
P = G['j1_split'](X2R)
cal, test = P[P.part == 'cal'], P[P.part == 'test']
assert not set(cal.src) & set(test.src) and len(cal) + len(test) == len(X2R[X2R.primary]) == 750 - 72
assert G['j1_split'](X2R).part.tolist() == P.part.tolist(), '나누기가 재현되지 않음'
print(f'[2] 나누기: J-cal {len(cal)} (근거 {cal.src.nunique()}), J-test {len(test)} (근거 {test.src.nunique()})')

G['PRICE'] = {'claude-haiku-5-5': (0.05, 0.25), 'claude-sonnet-5-5': (1.0, 5.0), 'claude-opus-5-5': (2.0, 10.0)}
exec(M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + J.J1_VERIFY + J.J1_METRICS, G)
REQ = G['fb_requests'](P, FB)
assert set(REQ) == {'haiku', 'sonnet'} and len(REQ['haiku']) == len(P)
r0 = REQ['sonnet'][f'sonnet-{P.i.iloc[0]}']
assert r0['model'] == 'claude-sonnet-5-5' and 'Summary — summarize' in r0['messages'][0]['content'] and r0['system'] == G['SYSTEM']
est = G['est_cost'](REQ['haiku']) + G['est_cost'](REQ['sonnet'])
assert 0 < est < G['BUDGET_USD'], est

def fake(i, mode):
    k = len(G['sentences'](FB[i]['summary']))
    if mode == 'truth':
        lab = ['H' if (G['fb_label'](FB[i]) == 3 and j == 0) else 'S' for j in range(k)]
    else:
        lab = ['H'] * k
    return {'text': json.dumps({'sentences': [{'i': j + 1, 'label': l} for j, l in enumerate(lab)]}), 'stop': 'end_turn',
            'model': 'x', 'in': 1000, 'out': 100}
RES = {'haiku': {f'haiku-{i}': fake(i, 'truth') for i in P.i}, 'sonnet': {f'sonnet-{i}': fake(i, 'allH') for i in P.i}}
bad = int(P[P.hal & (P.part == 'test')].i.iloc[0]); RES['haiku'][f'haiku-{bad}']['text'] = 'not json'     # 파싱 실패 1건
P = G['claude_cols'](P, FB, RES)
row = P[P.i == bad].iloc[0]
assert not row.ok_haiku and not row.v_haiku, '파싱 실패는 표시 안 함이어야 함'
MC = {str(i): {'p': [0.1 if G['fb_label'](FB[i]) == 3 else 0.9], 'ms': 100.0} for i in P.i}
_cal = P[P.part == 'cal'].copy(); _cal['s'] = [max(1 - p for p in MC[str(i)]['p']) for i in _cal.i]
thr = G['conformal'](_cal[~_cal.hal].s.values, 0.10)
P = G['minicheck_cols'](P, MC, thr)

CANDS = ['haiku', 'minicheck', 'sonnet']
tau = G['gate_threshold'](P)
assert abs((P[(P.part == 'cal') & P.hal].max_c >= tau).mean() - 0.95) < 0.01
P['gate'] = P.max_c >= tau
for c in CANDS: P[f'k_{c}'] = P.gate & P[f'v_{c}']
sel = G['select'](P, CANDS)
assert sel == 'haiku', sel                                   # J-cal에서 haiku·minicheck 모두 탐지 1 → 동점 규칙으로 haiku
R = G['judge_j1'](P, sel)
assert R['J1'] and R['J1_1'] and R['J1_2'] and R['det'] > 0.99 and R['flag'] == 0.0, R
assert R['diff_vs_cB']['ci'][0] > 0 and 0.5 < R['cascade']['escalate'] <= 1.0
print(f"[3] 선택 {sel} | J-test 탐지 {R['det']:.3f} 표시 {R['flag']:.3f} | 배포 조합 B 대비 {R['diff_vs_cB']['delta']:+.3f} "
      f"[{R['diff_vs_cB']['ci'][0]:+.3f}, {R['diff_vs_cB']['ci'][1]:+.3f}] | 2단 보내는 비율 {R['cascade']['escalate']:.3f}, 거르기 탐지 {R['cascade']['det']:.3f} → J1-3 {R['J1_3']}")

# 후보가 없으면 미달
P2 = P.copy()
for c in CANDS: P2[f'v_{c}'] = True
assert G['select'](P2, CANDS) is None and G['judge_j1'](P2, None)['J1'] is False
print('[4] 후보 없음 → J1 미달 확인')
print('smoke_j1 통과')
