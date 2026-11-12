"""X2 smoke test (맥 CPU).
1) 노트북 코드 셀 문법
2) 실제 FaithBench 데이터로 정답 등급·겹침 판정·지표 함수 (점수는 정답에서 만든 가짜 → AUROC 1, 무작위 → 약 0.5)
3) 실제 판정기(F1 base·I2)로 FaithBench 요약 몇 개를 노트북과 같은 코드로 판정, K2 자리에 F1 base를 넣으면 새 조합 = 기존 조합
   python experiments/builders/smoke_x2.py <FaithBench 폴더> [F1 base 폴더] [I2 폴더]"""
import ast, glob, json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import build_x2 as X
M1, L3 = X.M1, X.L3

n = 0
for c in X.cells:
    if c['cell_type'] != 'code': continue
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    ast.parse('\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('!'))); n += 1
print(f'[1] x2: 코드 셀 {n}개 문법 통과')

fbdir = sys.argv[1]
G = {'np': np, 'pd': pd, 'json': json, 'os': os, 'random': random}
exec(L3.L3_INPUTS + X.X2_DATA.replace("glob.glob('FaithBench/", f"glob.glob('{fbdir}/") + X.X2_METRICS, G)
FB = G['FB']; assert len(FB) == 750
labs = [G['fb_label'](s) for s in FB]
assert labs.count(3) == 439 and labs.count(0) == 175, (labs.count(3), labs.count(0))
ov = G['overlap_flags'](sorted({s['source'] for s in FB}), [FB[0]['source'] + ' extra words here', 'unrelated text ' * 30])
assert ov[FB[0]['source']] and sum(ov.values()) == 1
rng = np.random.default_rng(0)
SC = {}
for i, s in enumerate(FB):
    y = G['fb_label'](s) == 3; on = G['fb_onset'](s)
    a = 0 if on is None else max(0, on - 5)
    SC[str(i)] = [[0, 5, 0.0, 0.0, 5.0 if y else -5.0, float(rng.normal())], [a, a + 5, 0.0, 0.0, -5.0, float(rng.normal())]]
DF = G['x2_rows'](FB, SC, {'c': 0.0, 'n': 0.0}, ov)
S = G['x2_summary'](DF)
assert abs(S['auroc']['max_c'] - 1.0) < 1e-9 and 0.35 < S['auroc']['max_n'] < 0.65, S['auroc']
assert S['n_primary'] == 750 - 72 - int(DF[(DF.worst != 2) & DF.overlap].shape[0]) and S['bacc']['flag_c'] == 1.0
print(f"[2] 지표 통과: 주 분석 {S['n_primary']} (환각 {S['n_hal']}, 멀쩡 {S['n_clean']}) | 정답 점수 AUROC {S['auroc']['max_c']:.3f}, 무작위 {S['auroc']['max_n']:.3f} | "
      f"HHEM-2.1 AUROC {S['auroc']['hhem21']:.3f} | 균형 정확도 GPT-4o {S['bacc'].get('pred_gpt_4o', float('nan')):.3f}")

fdir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/Downloads/f1_final_judges/judge_base')
idir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser('~/Downloads/i2_judge/judge_i2')
if not (os.path.isdir(fdir) and os.path.isdir(idir)):
    print('[3] 건너뜀 (판정기 폴더 필요)'); print('smoke_x2 통과'); sys.exit(0)
import torch
from hallugate.judge import StudentJudge, read_config, input_mode
E = {'np': np, 'json': json, 'os': os, 'time': time, 'torch': torch, 'StudentJudge': StudentJudge, 'read_config': read_config, 'input_mode': input_mode}
exec(f'COMBO = {M1.COMBO!r}\n' + L3.L3_INPUTS + L3.L3_CORE + M1.M1_SCORE, E)
jF, jI, jK = E['make_judge'](fdir, 'cpu'), E['make_judge'](idir, 'cpu'), E['make_judge'](fdir, 'cpu')
CN = dict(M1.COMBO)
idx = [k for k, s in enumerate(FB) if len(s['source'].split()) < 120][:4]
for i in idx:
    s = FB[i]; src, summ = s['source'], s['summary']
    pts = [list(p) for p in E['score_response'](jF, jI, (jF.prepare(src), jI.prepare(src)), summ, E['judge_seq'])]
    rk = jK.prepare(src)
    for p in pts:
        sk = jK.judge(rk, summ[p[0]:p[1]].strip(), summ[:p[0]]).score
        p.append(((sk - CN['F_mean']) / CN['F_std'] + (p[3] - CN['I_mean']) / CN['I_std']) / 2)
    assert pts and max(abs(p[4] - p[5]) for p in pts) < 0.02, pts
    print(f"    요약 {i} ({s['metadata']['summarizer'].split('/')[-1]}, 등급 {G['fb_label'](s)}): 지점 {len(pts)}, 최대 조합 {max(p[4] for p in pts):.2f}")
print('[3] 실제 판정기 판정 통과 (K2 자리에 같은 판정기 → 새 조합 = 기존 조합)')
print('smoke_x2 통과')
