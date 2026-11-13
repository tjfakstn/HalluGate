"""J1 노트북 생성: 강한 검증기 2단 구조 — 경량 판정기(I3 조합)가 놓치는 환각을 응답 끝의 강한 검증기로 잡을 수 있는지 잰다.
X2에서 배포 조합이 FaithBench 환각을 서비스 임계값 B로 11.8%밖에 못 잡은 것의 후속(11/13 설계·고정). 사전 기준은 SSOT §6 J1.
평가는 사람 정답(FaithBench)으로만 판정한다. FaithBench 근거를 시드 2035로 둘로 나눠 J-cal(후보 선택·보정)과 J-test(판정)로 쓴다.
devM(최신 Claude 응답, Opus 정답)은 같은 지시문을 쓰는 Claude 후보에게 순환이 생기므로 보고만 한다.
FaithBench 데이터(CC BY-NC-SA 4.0)는 노트북 안에서 받아 쓰고 저장소에 넣지 않는다. RAGTruth는 열지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
_spec = importlib.util.spec_from_file_location('build_x2', os.path.join(HERE, 'build_x2.py'))
X2 = importlib.util.module_from_spec(_spec); _argv = sys.argv; sys.argv = ['import']; _spec.loader.exec_module(X2); sys.argv = _argv
M1, L3 = X2.M1, X2.L3

# 배치 API 도구(get_key, A, PRICE, run_batch, cost)는 M1과 같은 코드
J1_API = M1.M1_API
assert 'def run_batch(' in J1_API and "'claude-sonnet-5-5'" in J1_API

# ---- 아래 J1_* 문자열은 smoke_j1.py가 실제 FaithBench·가짜 API 결과로 그대로 실행해 확인한다 ----
J1_SPLIT = r"""
import pandas as pd
CANDS_CLAUDE = {'haiku': 'claude-haiku-5-5', 'sonnet': 'claude-sonnet-5-5'}
ORDER = ['haiku', 'minicheck', 'sonnet']          # 동점이면 싼 쪽 (사전 고정)
BUDGET_USD = 25.0                                  # 이 노트북 전체 API 상한 (사전 고정)

def j1_split(X2R, seed=2035):
    # 주 분석 행(X2와 같은 정의)만. 근거 단위로 섞어 앞 절반 = J-cal, 나머지 = J-test
    P = X2R[X2R.primary].copy()
    srcs = sorted(P.src.unique()); random.Random(seed).shuffle(srcs)
    cal = set(srcs[:len(srcs) // 2])
    P['part'] = ['cal' if s in cal else 'test' for s in P.src]
    assert not set(P[P.part == 'cal'].src) & set(P[P.part == 'test'].src), 'J-cal·J-test 근거가 겹침'
    return P
"""

J1_VERIFY = r"""
LABELER = 'claude-opus-5-5'          # label_params 기본값. v_params가 후보 모델로 바꾼다

def fb_case(s):
    return {'task': 'Summary', 'ref_label': s['source']}

def v_params(c, text, model):
    # R4·M1과 같은 문장 단위 지시문, 모델만 바꾼다
    p = label_params(c, text); p['model'] = model
    return p

def est_cost(reqs, out_tokens=400):
    # 요청 글자 수 / 3.5 ≈ 입력 토큰 (추정). 배치 단가
    tot = 0.0
    for p in reqs.values():
        n_in = (len(p.get('system', '')) + sum(len(m['content']) for m in p['messages'])) / 3.5
        pin, pout = PRICE[p['model']]
        tot += n_in / 1e6 * pin + out_tokens / 1e6 * pout
    return tot

def fb_requests(P, FB):
    return {m: {f'{m}-{i}': v_params(fb_case(FB[i]), FB[i]['summary'], model) for i in P.i} for m, model in CANDS_CLAUDE.items()}

def claude_cols(P, FB, RES):
    # 응답마다: 파싱 성공, H 문장 하나라도(주), H 또는 A(보고). 파싱 실패 = 표시 안 함(탐지에 불리한 쪽)
    for m in CANDS_CLAUDE:
        ok_l, h_l, ha_l = [], [], []
        for i in P.i:
            r = RES[m].get(f'{m}-{i}'); n = len(sentences(FB[i]['summary']))
            labs, ok = parse(r['text'], n) if r else ({}, False)
            ls = [v['label'] for v in labs.values()] if ok else []
            ok_l.append(ok); h_l.append('H' in ls); ha_l.append('H' in ls or 'A' in ls)
        P[f'ok_{m}'], P[f'v_{m}'], P[f'v_{m}_HA'] = ok_l, h_l, ha_l
    return P

def minicheck_cols(P, MC, thr):
    # 응답 점수 = 문장마다 (1 − 근거 지지 확률)의 최대. 임계값은 J-cal 멀쩡한 응답 등각 10%
    P['s_minicheck'] = [max([1 - p for p in MC[str(i)]['p']], default=0.0) for i in P.i]
    P['v_minicheck'] = P.s_minicheck >= thr
    return P
"""

J1_METRICS = r"""
def rates(d, col):
    return float(d[d.hal][col].astype(bool).mean()), float(d[~d.hal][col].astype(bool).mean())

def select(P, cands):
    # J-cal에서: 멀쩡한 응답 표시율 ≤ 15%인 후보 중 탐지율 최대 (동점이면 ORDER 순)
    C = P[P.part == 'cal']; ok = []
    for c in cands:
        det, fl = rates(C, f'v_{c}')
        print(f'  J-cal {c:10} 탐지 {det:.3f} | 멀쩡한 응답 표시 {fl:.3f}')
        if fl <= 0.15: ok.append((det, -ORDER.index(c), c))
    return max(ok)[2] if ok else None

def gate_threshold(P, keep=0.95):
    # 1단(배포 조합) 통과 기준: J-cal 환각 응답의 95%가 2단으로 넘어가는 점수
    h = np.sort(P[(P.part == 'cal') & P.hal].max_c.values)
    return float(h[int(np.floor((1 - keep) * len(h)))])

def judge_j1(P, sel, seed=2035):
    T = P[P.part == 'test']
    out = {'n_test': len(T), 'n_test_hal': int(T.hal.sum()), 'n_test_clean': int((~T.hal).sum()), 'selected': sel}
    out['cB'] = dict(zip(['det', 'flag'], rates(T, 'flag_cB')))
    if sel is None:
        out.update(J1_1=False, J1_2=False, J1=False); return out
    v = f'v_{sel}'
    det, fl = rates(T, v)
    out['det'], out['flag'] = det, fl
    out['det_ci'] = cluster_boot(T, lambda d: d[d.hal][v].astype(bool).mean(), seed=seed)
    out['flag_ci'] = cluster_boot(T, lambda d: d[~d.hal][v].astype(bool).mean(), seed=seed)
    diff = lambda d: d[d.hal][v].astype(bool).mean() - d[d.hal].flag_cB.astype(bool).mean()
    out['diff_vs_cB'] = {'delta': det - out['cB']['det'], 'ci': cluster_boot(T, diff, seed=seed)}
    out['J1_1'] = bool(det >= 0.50 and out['diff_vs_cB']['ci'][0] > 0)
    out['J1_2'] = bool(fl <= 0.15)
    out['J1'] = out['J1_1'] and out['J1_2']
    k = f'k_{sel}'
    out['cascade'] = {'det': rates(T, k)[0], 'flag': rates(T, k)[1], 'escalate': float(T.gate.mean())}
    out['J1_3'] = bool(out['cascade']['det'] >= det - 0.03)
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# J1 — 강한 검증기 2단 구조 (FaithBench 사람 정답)

## 왜 하나

X2(11/13)에서 배포 판정기(F1 base + I2 조합)는 FaithBench 사람 정답 기준 AUROC 0.629로 판별력은 있었지만, **서비스 임계값 B에서 환각 응답의 11.8%만 잡았다**.
K1·K2에서 경량 판정기를 다시 학습해도 최신 LLM 탐지율은 +0.021에 그쳤다. 경량 판정기 하나로는 환각을 잡는 서비스가 되지 않는다.
그래서 **경량 판정기는 스트리밍 중 1단으로 두고, 응답이 끝나면 강한 검증기가 다시 판정하는 2단 구조**가 환각을 절반 이상 잡을 수 있는지 잰다.

## 무엇을 하나 (학습 없음)

- **후보(2단)**: `claude-haiku-5-5`, `claude-sonnet-5-5`(R4·M1과 같은 문장 단위 지시문, 문장마다 S/H/A, **H 문장이 하나라도 있으면 표시**),
  `Bespoke-MiniCheck-7B`(공개, 문장마다 근거 지지 확률, 응답 점수 = 최대 (1 − 확률), 임계값 = J-cal 멀쩡한 응답 등각 10%)
- **데이터**: FaithBench 주 분석 661개(X2와 같은 정답·제외 규칙). 근거를 `random.Random(2035)`로 섞어 절반은 **J-cal**(후보 선택·보정), 절반은 **J-test**(판정)
- **1단 거르기(보고·J1-3)**: 배포 조합 점수가 J-cal 환각 응답의 95%를 통과시키는 기준 이상인 응답만 2단으로 보낸다
- **devM**(K2a의 최신 Claude 응답, Opus 정답): 정답과 같은 지시문이라 Claude 후보에게 순환이 생겨 **보고만**

## 사전 정의 기준 (SSOT §6 J1, 실행 전 고정)

- **J1-0(유효성)**: Claude 후보마다 J-cal·J-test 응답의 JSON 파싱 성공 ≥ 95%. 미달 후보는 뺀다. MiniCheck가 설치·실행되지 않으면 후보에서 빠진다
- **선택(J-cal)**: 멀쩡한 응답 표시율 ≤ 15%인 후보 중 환각 탐지율 최대(동점이면 haiku → minicheck → sonnet). 없으면 J1 미달
- **J1-1(주, J-test)**: 선택 후보의 환각 응답 탐지율 ≥ 0.50, **그리고** 서비스 임계값 B 배포 조합 대비 탐지율 차이의 근거 단위 군집 부트스트랩 95% 하한 > 0 (2,000회, 시드 2035)
- **J1-2(J-test)**: 선택 후보의 멀쩡한 응답 표시율 ≤ 15%
- **J1 채택 = J1-1 그리고 J1-2** → 게이트웨이에 2단 검증을 넣는다(SPEC부터). 미달이면 "FaithBench급 환각은 현재 검증기로 절반을 못 잡는다"로 기록하고 서비스 주장을 줄인다
- **J1-3(구조)**: 1단 거르기 + 2단의 J-test 탐지율 ≥ 2단 단독 − 0.03이면 거르기를 쓰고, 아니면 모든 응답을 2단으로 보낸다
- **J1-4(지연)**: 선택 후보의 응답 1개 검증 p95 ≤ 5초(J-cal 40개, 실시간 요청)면 "응답을 잡아 두고 검증(hold)" 가능, 넘으면 응답 뒤 경고(비동기)로만 쓴다
- 보고만: 모든 후보의 J-cal·J-test 수치, H 또는 A 기준, 생성 모델별, devM, 비용. **API 비용 상한 $25**(제출 전 추정이 넘으면 멈춤)

> 런타임: **GPU**(MiniCheck-7B용, L4·A100 권장). 먼저 X2를 돌려 드라이브 `x2/x2_rows.csv`가 있어야 한다. Colab 비밀 값 `ANTHROPIC_API_KEY` 필요.
> 끊기면 다시 모두 실행(배치·점수는 드라이브 `j1/`에서 이어서). 끝나면 `j1_results.zip`(원문 없음)이 내려받아진다.
""")

md("## 1. 환경 준비")
code(r"""
import os, sys, json, time, random, subprocess, shutil
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'anthropic', 'tiktoken', 'pysbd', 'scikit-learn'], check=True)
subprocess.run('git clone --depth 1 https://github.com/vectara/FaithBench.git 2>/dev/null || true', shell=True)
import numpy as np, torch
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
X2_DIR, K2_DIR, J1_DIR = os.path.join(OUT, 'x2'), os.path.join(OUT, 'k2'), os.path.join(OUT, 'j1')
os.makedirs(J1_DIR, exist_ok=True)
assert os.path.exists(os.path.join(X2_DIR, 'x2_rows.csv')), 'x2/x2_rows.csv 없음 — X2 노트북을 먼저 실행'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
print('device', GPU)
""")

md("## 2. FaithBench, X2 점수, J-cal·J-test 나누기")
code(L3.L3_INPUTS + X2.X2_DATA + X2.X2_METRICS + J1_SPLIT + r"""
assert len(FB) == 750, len(FB)
X2R = pd.read_csv(os.path.join(X2_DIR, 'x2_rows.csv'))
assert len(X2R) == 750 and 'flag_cB' in X2R, 'X2 결과에 임계값 B 표시(flag_cB)가 없음'
P = j1_split(X2R)
for part, d in P.groupby('part'):
    print(f'J-{part}: 응답 {len(d)} (환각 {int(d.hal.sum())}, 멀쩡 {int((~d.hal).sum())}), 근거 {d.src.nunique()}')
""")

md("## 3. Claude 후보 — 비용 추정, 배치 제출·결과 받기")
code(M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + J1_VERIFY + "\n" + J1_API + r"""
REQ = fb_requests(P, FB)
dm_files = [os.path.join(K2_DIR, f'{k}_devM.json') for k in ['cases', 'gen', 'label']]
HAS_DM = all(os.path.exists(f) for f in dm_files)
if HAS_DM:
    CM, GM, LM = (load_json(f) for f in dm_files)
    DM_VALID = [c for c in CM if gen_status(GM.get(c)) == 'valid' and c in LM]
    REQ_DM = {m: {f'{m}-{c}': v_params(CM[c], GM[c]['text'], model) for c in DM_VALID} for m, model in CANDS_CLAUDE.items()}
else:
    REQ_DM = {m: {} for m in CANDS_CLAUDE}
EST = sum(est_cost(r) for r in REQ.values()) + sum(est_cost(r) for r in REQ_DM.values())
print(f'요청: FaithBench {sum(map(len, REQ.values()))}, devM {sum(map(len, REQ_DM.values()))} | 예상 비용 ${EST:.2f} (상한 ${BUDGET_USD})')
assert EST <= BUDGET_USD, '예상 비용이 상한을 넘음 — 멈춤'
_i = int(P.i.iloc[0])                                  # 형식 확인 1건 (실시간, 1센트 미만)
for m, model in CANDS_CLAUDE.items():
    _r = rec(A.messages.create(**v_params(fb_case(FB[_i]), FB[_i]['summary'], model)))
    _labs, _ok = parse(_r['text'], len(sentences(FB[_i]['summary'])))
    print(f"  형식 확인 {m}: 파싱 {'성공' if _ok else '실패'} {[v['label'] for v in _labs.values()]} | 입력 {_r['in']} 출력 {_r['out']}")
RES = {m: run_batch(REQ[m], os.path.join(J1_DIR, f'v_{m}_fb.json')) for m in CANDS_CLAUDE}
P = claude_cols(P, FB, RES)
for m in CANDS_CLAUDE:
    print(f'  {m}: 파싱 성공 J-cal {P[P.part == "cal"][f"ok_{m}"].mean():.3f}, J-test {P[P.part == "test"][f"ok_{m}"].mean():.3f}')
""")

md("## 4. MiniCheck-7B 후보 (설치·실행이 안 되면 후보에서 빠진다)")
code(r"""
MC = None
try:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main'], check=True)
    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False, cache_dir='/content/ckpts')
    mp = os.path.join(J1_DIR, 'minicheck_fb.json'); MC = load_json(mp) or {}
    t0 = time.time()
    for k, i in enumerate(P.i):
        if str(i) in MC: continue
        ss = [t for _, _, t in sentences(FB[i]['summary'])]
        t1 = time.perf_counter()
        _, prob, _, _ = mc.score(docs=[FB[i]['source']] * len(ss), claims=ss)
        MC[str(i)] = {'p': [float(x) for x in prob], 'ms': (time.perf_counter() - t1) * 1000}
        if (k + 1) % 50 == 0: save_json(MC, mp); print(f'  {k + 1}/{len(P)} ({time.time() - t0:.0f}s)', flush=True)
    save_json(MC, mp)
except Exception as e:
    print('MiniCheck 실행 안 됨 → 후보에서 뺀다:', type(e).__name__, str(e)[:300]); MC = None
if MC is not None:
    _cal = P[P.part == 'cal'].copy()
    _cal['s'] = [max([1 - p for p in MC[str(i)]['p']], default=0.0) for i in _cal.i]
    THR_MC = conformal(_cal[~_cal.hal].s.values, 0.10)
    P = minicheck_cols(P, MC, THR_MC); print(f'MiniCheck 임계값(J-cal 멀쩡한 응답 등각 10%) {THR_MC:.4f}')
""")

md("## 5. 선택(J-cal) → 판정(J-test)")
code(J1_METRICS + r"""
CANDS = [m for m in CANDS_CLAUDE
         if P[P.part == 'cal'][f'ok_{m}'].mean() >= 0.95 and P[P.part == 'test'][f'ok_{m}'].mean() >= 0.95]
if MC is not None: CANDS.append('minicheck')
CANDS = sorted(CANDS, key=ORDER.index)
print('J1-0 통과 후보:', CANDS)
TAU1 = gate_threshold(P)
P['gate'] = P.max_c >= TAU1
for c in CANDS: P[f'k_{c}'] = P.gate & P[f'v_{c}']
SEL = select(P, CANDS)
R = judge_j1(P, SEL)
T = P[P.part == 'test']
print(f"\nJ-test: 응답 {R['n_test']} (환각 {R['n_test_hal']}, 멀쩡 {R['n_test_clean']}) | 배포 조합 B: 탐지 {R['cB']['det']:.3f}, 표시 {R['cB']['flag']:.3f}")
for c in CANDS:
    d, f = rates(T, f'v_{c}'); print(f'  (보고) J-test {c:10} 탐지 {d:.3f} | 표시 {f:.3f}' + (f" | H 또는 A: {rates(T, f'v_{c}_HA')[0]:.3f} / {rates(T, f'v_{c}_HA')[1]:.3f}" if c in CANDS_CLAUDE else ''))
if SEL is None:
    print('\n=== J1 미달: J-cal에서 멀쩡한 응답 표시율 ≤ 15%인 후보가 없음 ===')
else:
    dc = R['diff_vs_cB']
    print(f"\n=== 선택: {SEL} ===")
    print(f"=== J1-1(주): 탐지율 {R['det']:.3f} [{R['det_ci'][0]:.3f}, {R['det_ci'][1]:.3f}] (≥ 0.50), 배포 조합 B 대비 {dc['delta']:+.3f} [{dc['ci'][0]:+.3f}, {dc['ci'][1]:+.3f}] (하한 > 0) → {'충족' if R['J1_1'] else '미달'} ===")
    print(f"=== J1-2: 멀쩡한 응답 표시율 {R['flag']:.3f} [{R['flag_ci'][0]:.3f}, {R['flag_ci'][1]:.3f}] (≤ 0.15) → {'충족' if R['J1_2'] else '미달'} ===")
    print(f"=== J1 {'채택' if R['J1'] else '미채택'} ===")
    cz = R['cascade']
    print(f"=== J1-3: 1단 거르기(조합 ≥ {TAU1:.3f}, 2단으로 보내는 비율 {cz['escalate']:.3f}) + 2단 탐지 {cz['det']:.3f} vs 2단 단독 {R['det']:.3f} → {'거르기 사용' if R['J1_3'] else '모든 응답 2단'} ===")
print('(보고) 생성 모델별 J-test 탐지율:', {m.split('/')[-1]: round(float(d[d.hal][f'v_{SEL}'].mean()), 3) for m, d in T.groupby('summarizer')} if SEL else '-')
""")

md("## 6. J1-4 지연 — 선택 후보의 응답 1개 검증 시간")
code(r"""
LAT = None
if SEL in CANDS_CLAUDE:
    ids = list(P[P.part == 'cal'].i)[:40]; ms = []
    for i in ids:                                     # 실시간 요청 (배치 아님, 표준 단가)
        t1 = time.perf_counter(); A.messages.create(**v_params(fb_case(FB[i]), FB[i]['summary'], CANDS_CLAUDE[SEL]))
        ms.append((time.perf_counter() - t1) * 1000)
    LAT = ms
elif SEL == 'minicheck':
    LAT = [MC[str(i)]['ms'] for i in P[P.part == 'cal'].i]
if LAT:
    p50, p95 = np.percentile(LAT, 50), np.percentile(LAT, 95)
    R['latency_ms'] = {'n': len(LAT), 'p50': float(p50), 'p95': float(p95)}; R['J1_4'] = bool(p95 <= 5000)
    print(f"=== J1-4: {SEL} 응답 1개 검증 p50 {p50:.0f}ms, p95 {p95:.0f}ms (≤ 5000) → {'hold 가능' if R['J1_4'] else '응답 뒤 경고(비동기)만'} ===")
""")

md("## 7. (보고) devM — 최신 Claude 응답, Opus 정답 (Claude 후보는 순환이므로 판정에 쓰지 않음)")
code(r"""
DM = None
if HAS_DM:
    gold = {}
    for c in DM_VALID:
        labs, ok = parse(LM[c]['text'], len(sentences(GM[c]['text'])))
        if ok: gold[c] = 'H' in [v['label'] for v in labs.values()]
    DM = {'n': len(gold), 'n_hal': sum(gold.values())}
    for m, model in CANDS_CLAUDE.items():
        res = run_batch(REQ_DM[m], os.path.join(J1_DIR, f'v_{m}_devM.json'))
        fl = {}
        for c in gold:
            r = res.get(f'{m}-{c}'); labs, ok = parse(r['text'], len(sentences(GM[c]['text']))) if r else ({}, False)
            fl[c] = ok and 'H' in [v['label'] for v in labs.values()]
        h = [fl[c] for c in gold if gold[c]]; k = [fl[c] for c in gold if not gold[c]]
        DM[m] = {'det': float(np.mean(h)), 'flag': float(np.mean(k))}
        print(f'  devM {m:7} 탐지 {DM[m]["det"]:.3f} | 멀쩡한 응답 표시 {DM[m]["flag"]:.3f}  (환각 {len(h)}, 멀쩡 {len(k)})')
else:
    print('k2/의 cases·gen·label_devM.json이 없어 건너뜀')
""")

md("## 8. 저장")
code(r"""
usage = {f: round(cost(load_json(os.path.join(J1_DIR, f))), 2) for f in os.listdir(J1_DIR) if f.startswith('v_') and f.endswith('.json') and '_batches' not in f}
print('API 사용량(배치 단가 기준, 지연 측정 제외) $', usage, '합계', round(sum(usage.values()), 2))
LOCAL = '/content/j1_local'; os.makedirs(LOCAL, exist_ok=True)
cols = [c for c in P.columns if c in ('i', 'src', 'summarizer', 'hal', 'part', 'max_c', 'flag_cB', 'gate', 's_minicheck') or c.startswith(('ok_', 'v_', 'k_'))]
P[cols].to_csv(os.path.join(LOCAL, 'j1_rows.csv'), index=False)        # 원문 없음
json.dump(dict(R, cands=CANDS, tau1=TAU1, thr_minicheck=(THR_MC if MC is not None else None), devM=DM, cost_usd=usage, gpu=GPU),
          open(os.path.join(LOCAL, 'j1_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(J1_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/j1_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'j1_strong_verifier.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
