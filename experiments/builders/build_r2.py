"""R2 노트북 생성: 점수를 경보로 바꾸는 순차 판정 규칙(누적합, 연속 초과, 평활, 학습 규칙)과
오경보 보장 보정(R3), 가변 재판정 간격을 R0의 점수 궤적으로 평가한다. GPU 불필요."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R2 — 점수를 경보로 바꾸는 규칙: 순차 판정, 오경보 보장 보정(R3), 가변 재판정 간격

## 왜 하나

R0(dev-L, 환각 응답 759개)에서 두 가지가 드러났다.
1. **실패의 대부분은 발생 전 경보다.** 기준선(B, 문장 끝)은 환각 응답의 41.5%에서 환각이 시작되기 **전에** 경보를 울려 실패했고, 끝까지 못 잡은 것은 8.4%뿐이다.
   점수기는 환각 응답 대부분에 반응하지만 너무 일찍 반응한다.
2. 지금 경보 규칙은 "판정 지점 하나라도 임계값을 넘으면 경보"다. 판정을 자주 할수록 오경보 기회가 늘어서, 같은 오경보 예산에서는 임계값이 올라가고 재현율이 떨어진다.

점수기는 그대로 두고 **여러 판정 지점의 점수 이력을 보고 경보를 정하는 규칙**으로 이 둘을 줄일 수 있는지 본다.
GPU가 필요 없다. R0가 저장한 점수 궤적(`selfproject/r0/`)만 쓴다.

## 무엇을 하나

| 규칙 | 내용 | 조절값 |
|---|---|---|
| 임계값 (기준) | 판정 지점 하나가 h 이상이면 경보 (E1·R0와 같음) | 판정 간격 |
| 누적합 (CUSUM) | S ← max(0, S + 점수 − k), S ≥ h이면 경보 후 S = 0. 점수가 꾸준히 높아야 울림 | 기준값 k, 문장마다 S 초기화 여부 |
| 연속 초과 | m번 연속 h 이상이면 경보 | m = 2, 3 |
| 평활 (EWMA) | 지수 평균 점수에 임계값 | λ = 0.3, 0.5 |
| 학습 규칙 | 판정 지점마다 점수 이력 특징(두 판정기 점수, 직전 점수, 문장 안 최고, 앞 문장들 점수, 위치)으로 "이미 환각이 시작됐나"를 예측하는 HistGradientBoosting. 선행 연구(Itkin 2026)의 학습형 규칙과 같은 계열이지만 입력이 근거를 읽는 판정기 점수 | 특징 집합(B+E2, E2만, B만) |

- **규칙 학습·조절값 선택·임계값 보정은 모두 cal-L로만** 한다. 학습 규칙은 cal-L 안에서 근거 문서 단위 5겹 교차 예측으로 보정·선택한다.
- dev-L에는 모든 후보를 보고하되, **사전 기준은 cal-L에서 고른 대표 규칙 하나로만 판정**한다.
- 같은 노트북에서 R3(오경보 보장 보정), 가변 재판정 간격, 발생 전 경보 분석, 서비스 조건에서 E2 대 B 비교도 한다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **R2-0 재현 점검**: 이 노트북의 일반 보정 함수로 임계값 규칙을 다시 계산하면 R0 dev-L 수치와 같다 (B 문장 끝 0.501, B s=4 0.481, E2 s=4 0.451, ±0.002). 어긋나면 이후 수치는 해석하지 않는다.
- **대표 규칙 선택(결과 보기 전 고정)**: cal-L에서 지연 중앙값 ≤ 9토큰인 후보 중 cal-L 엄격 재현율(ARL₀=100)이 가장 높은 것. 동점이면 판정 횟수가 적은 것.
- **R2 주 기준**: 대표 규칙이 dev-L ARL₀=100에서 **엄격 재현율 ≥ 0.531**(B 문장 끝 0.501 + 0.03), **B 문장 끝 대비 쌍체 부트스트랩 95% 하한 > 0**, **지연 중앙값 ≤ 9토큰**(B s=4 수준)을 모두 만족.
- **R2 보조(판정 없이 보고)**: 대표 규칙의 서비스 조건(표시 10%) 재현율과 지연 — 참고선 B 문장 끝 0.395 / 21토큰, B s=4 0.331 / 15토큰.
- **R3**: 등각 보정(α = 10%, δ = 10%, Clopper–Pearson 상한)으로 정한 임계값에서 dev-L 실제 표시율 ≤ 10% — 대표 규칙, B 문장 끝, E2 s=4 **모두**.
- **게이트웨이 판정기 결정 규칙**: 서비스 조건에서 E2 s=4 vs B s=4 쌍체 부트스트랩으로, 재현율 차이 하한 > 0 이거나 (재현율 차이 구간이 0을 포함하고 지연 차이 상한 < 0)이면 student_prefix 유지, 아니면 기본 판정기를 student_vark(+ 재판정)로 바꾼다.
- **보고(판정 없음)**: 발생 전 경보의 위치 분포(cal-L), 정답 경계 허용 폭별 재현율, 가변 재판정 간격의 판정 횟수 대비 재현율·지연, 대표 규칙 계열을 시드 판정기에 적용한 결과.

> 런타임 **CPU로 충분**(GPU 필요 없음). 약 15~30분. 드라이브에 R0 결과(`selfproject/r0/r0_items.pkl`, `r0_scores.npz`)가 있어야 한다.
""")

md("## 1. 준비 — R0 점수 궤적 불러오기")
code(r"""
import os, json, pickle, time, math, shutil
import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)

USE_DRIVE = True
OUT_DIR = '/content/drive/MyDrive/selfproject' if USE_DRIVE else '.'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
R0_DIR, R2_DIR = os.path.join(OUT_DIR, 'r0'), os.path.join(OUT_DIR, 'r2')
os.makedirs(R2_DIR, exist_ok=True)
P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
Z = dict(np.load(os.path.join(R0_DIR, 'r0_scores.npz')))
CAL, DEV = P['items']['calL'], P['items']['devL']
Q_TEXTS = P['q_texts']
SEEDS = [m for m in Z if m.startswith('S')]
for name, its in [('cal-L', CAL), ('dev-L', DEV)]:
    print(f'{name}: 응답 {len(its)} (환각 {sum(x["onset"] is not None for x in its)}) | 판정기 {list(Z)}')

try:
    from numba import njit
except Exception:                       # numba가 없으면 순수 파이썬으로 (느리지만 동작)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
""")

md(r"""## 2. 판정 지점 — 문장 끝 / 8 / 4 / 2토큰 간격 (E1과 같은 규칙)""")
code(r"""
KINDS = ['end', '8', '4', '2']
TASK_ID = {'QA': 0, 'Summary': 1, 'Data2txt': 2}

def build_checks(x):
    sidx = {sn['t0']: i for i, sn in enumerate(x['sents'])}
    out = {}
    for kind in KINDS:
        T, Q, E, K, S_ = [], [], [], [], []
        for t, q in x['pts2']:
            t0 = x['sent_of'][t]; k = t - t0 + 1; end = t in x['ends']
            if kind == 'end' and not end: continue
            if kind != 'end' and not (end or k % int(kind) == 0): continue
            T.append(t); Q.append(q); E.append(end); K.append(k); S_.append(sidx[t0])
        S_ = np.array(S_, dtype=np.int64)
        out[kind] = {'t': np.array(T, dtype=np.int64), 'q': np.array(Q, dtype=np.int64), 'end': np.array(E, dtype=bool),
                     'k': np.array(K, dtype=np.int64), 's': S_,
                     'new': np.r_[True, S_[1:] != S_[:-1]] if len(S_) else np.zeros(0, bool)}
    x['chk'] = out
    if x['onset'] is not None:
        x['onset_sent'] = next(i for i, sn in enumerate(x['sents']) if sn['t0'] <= x['onset'] <= sn['t1']) \
            if any(sn['t0'] <= x['onset'] <= sn['t1'] for sn in x['sents']) else None

for x in CAL + DEV:
    build_checks(x)
print('판정 횟수(응답당 평균):', {k: round(np.mean([len(x['chk'][k]['t']) for x in DEV]), 1) for k in KINDS})
""")

md(r"""## 3. 규칙 엔진 — 보정(ARL₀, 서비스, 등각)과 엄격 평가

모든 규칙은 "판정 지점마다 통계량 → h 이상이면 경보"로 표현한다. 오경보 횟수는 E1처럼 **경보가 울릴 때마다 센다**(누적합·연속 규칙은 경보 후 초기화).
임계값 규칙에서는 이 정의가 E1·R0와 정확히 같다.""")
code(r"""
@njit(cache=True)
def _cusum_path(c, new, reset_sent):
    S = np.zeros(len(c)); s = 0.0
    for i in range(len(c)):
        if reset_sent and new[i]: s = 0.0
        s = max(0.0, s + c[i]); S[i] = s
    return S

@njit(cache=True)
def _cusum_fires(c, new, reset_sent, h):
    s = 0.0; n = 0
    for i in range(len(c)):
        if reset_sent and new[i]: s = 0.0
        s = max(0.0, s + c[i])
        if s >= h:
            n += 1; s = 0.0
    return n

@njit(cache=True)
def _consec_fires(z, m, h):
    run = 0; n = 0
    for i in range(len(z)):
        run = run + 1 if z[i] >= h else 0
        if run >= m:
            n += 1; run = 0
    return n

def ewma(z, lam):
    out = np.empty_like(z); s = None
    for i, v in enumerate(z):
        s = v if s is None else lam * v + (1 - lam) * s; out[i] = s
    return out

LR = {}          # 학습 규칙 통계량: LR[name][item id] = 판정 지점별 로짓

def base(x, rule):
    c = x['chk'][rule['kind']]
    if rule['type'] == 'learned':
        return LR[rule['lr']][x['id']]
    z = Z[rule['model']][c['q']].astype(np.float64)
    if rule['type'] == 'ewma':
        return ewma(z, rule['lam'])
    if rule['type'] == 'cusum':
        return z - rule['kref']
    return z

CACHE = {}
def prep(x, rule):
    key = (rule['name'], x['id'])
    if key not in CACHE:
        b = base(x, rule); c = x['chk'][rule['kind']]
        if rule['type'] == 'cusum':
            path = _cusum_path(b, c['new'], rule['reset_sent'])
            CACHE[key] = (b, path, path.max() if len(path) else -np.inf)
        elif rule['type'] == 'consec':
            m = rule['m']
            lv = max((b[i:i + m].min() for i in range(len(b) - m + 1)), default=-np.inf)
            CACHE[key] = (b, None, lv)
        else:
            CACHE[key] = (b, b, b.max() if len(b) else -np.inf)
    return CACHE[key]

def n_fires(items, rule, h):
    n = 0
    for x in items:
        b, path, _ = prep(x, rule)
        if rule['type'] == 'cusum':
            n += _cusum_fires(b, x['chk'][rule['kind']]['new'], rule['reset_sent'], h)
        elif rule['type'] == 'consec':
            n += _consec_fires(b, rule['m'], h)
        else:
            n += int((b >= h).sum())
    return n

def first_idx(x, rule, h):
    b, path, lv = prep(x, rule)
    if lv < h: return None
    if rule['type'] == 'consec':
        run = 0
        for i, v in enumerate(b):
            run = run + 1 if v >= h else 0
            if run >= rule['m']: return i
        return None
    return int(np.argmax(path >= h))

def level(x, rule):
    return prep(x, rule)[2]

def calibrate_arl(items, rule, arl0=100):
    clean = [x for x in items if x['onset'] is None]
    n_alarm = max(1, int(sum(x['n_tok'] for x in clean) / arl0))
    target = n_alarm - 1
    allv = np.concatenate([prep(x, rule)[0] for x in clean])
    if rule['type'] in ('thr', 'ewma', 'learned'):       # 경보 후 초기화가 없는 규칙: E1과 같은 공식 (+1e-6 포함)
        sv = np.sort(allv)[::-1]
        return float(sv[n_alarm - 1]) + 1e-6 if n_alarm <= len(sv) else float(sv[-1])
    lo = (0.0 if rule['type'] == 'cusum' else float(allv.min())) - 1.0
    hi = max(level(x, rule) for x in clean) + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if n_fires(clean, rule, mid) <= target: hi = mid
        else: lo = mid
    return hi

def calibrate_flag(items, rule, rate):
    mx = np.array([level(x, rule) for x in items if x['onset'] is None])
    return float(np.quantile(mx, 1 - rate)) + 1e-6

def calibrate_conformal(items, rule, alpha=0.10, delta=0.10):
    # Clopper–Pearson 상한: 표시되는 멀쩡한 응답 비율의 (1−δ) 신뢰 상한이 α 이하가 되는 가장 낮은 임계값
    mx = np.sort(np.array([level(x, rule) for x in items if x['onset'] is None]))[::-1]
    n = len(mx); best_k = 0
    for k in range(n):
        if beta.ppf(1 - delta, k + 1, n - k) <= alpha: best_k = k
        else: break
    return float(mx[best_k]) + 1e-6          # 이 값 이상인 응답 수 = best_k

def evaluate(items, rule, h):
    kind = rule['kind']
    clean = [x for x in items if x['onset'] is None]
    hall = [x for x in items if x['onset'] is not None]
    fires = n_fires(clean, rule, h)
    det, dl, dch, pre = 0, [], [], 0
    for x in hall:
        i = first_idx(x, rule, h)
        if i is None: continue
        t = int(x['chk'][kind]['t'][i])
        if t >= x['onset']:
            det += 1; dl.append(t - x['onset']); dch.append(x['te'][t] - x['ts'][x['onset']])
        else:
            pre += 1
    return {'recall': det / len(hall), 'delay_med': float(np.median(dl)) if dl else np.nan,
            'delay_chars_med': float(np.median(dch)) if dch else np.nan,
            'pre_onset': pre / len(hall), 'missed': 1 - (det + pre) / len(hall),
            'realized_ARL0': sum(x['n_tok'] for x in clean) / max(fires, 1),
            'clean_flag': float(np.mean([level(x, rule) >= h for x in clean])),
            'calls': float(np.mean([len(x['chk'][kind]['t']) for x in items]))}

def per_resp(items, rule, h):
    d, l = [], []
    for x in items:
        if x['onset'] is None: continue
        i = first_idx(x, rule, h)
        t = None if i is None else int(x['chk'][rule['kind']]['t'][i])
        ok = t is not None and t >= x['onset']
        d.append(float(ok)); l.append(t - x['onset'] if ok else np.nan)
    return np.array(d), np.array(l)

def R(rtype, kind, **kw):
    name = f"{rtype}|{kind}|" + ','.join(f'{k}={v}' for k, v in sorted(kw.items()))
    return dict(type=rtype, kind=kind, name=name, **kw)
""")

md(r"""## 4. R2-0 재현 점검 — 임계값 규칙이 R0 dev-L 수치와 같은가""")
code(r"""
REF = {('B', 'end'): 0.5007, ('B', '4'): 0.4809, ('E2', '4'): 0.4506}
ok_all = True
for (m, k), ref in REF.items():
    r = R('thr', k, model=m)
    got = evaluate(DEV, r, calibrate_arl(CAL, r))['recall']
    ok = abs(got - ref) <= 0.002; ok_all &= ok
    print(f'{m}|{k:3}  R0 {ref:.4f} | 지금 {got:.4f} | {"일치" if ok else "불일치"}')
R20 = ok_all
print('R2-0 재현:', '충족' if R20 else '미달 — 이후 수치를 해석하지 말 것')
""")

md(r"""## 5. 발생 전 경보 분석 (cal-L) — 이른 경보는 어디서 울리나

B 문장 끝과 E2 s=4의 ARL₀=100 임계값에서, 환각 시작 전에 첫 경보가 울린 응답을 본다.
경보가 **환각이 시작되는 바로 그 문장 안**(시작 토큰 직전)인지, **바로 앞 문장**인지, **그보다 앞**인지 나눈다.
바로 앞에 몰려 있다면 정답 구간 경계가 모호한 문제일 수 있다. 허용 폭별 재현율은 참고용이다(주 지표는 엄격 기준 그대로).""")
code(r"""
pre_rows = []
for m, k in [('B', 'end'), ('E2', '4'), ('B', '4')]:
    r = R('thr', k, model=m); h = calibrate_arl(CAL, r)
    gaps, where = [], {'같은 문장': 0, '바로 앞 문장': 0, '2문장 이상 앞': 0}
    win = {5: 0, 10: 0, 20: 0}; n_h = 0; det = 0
    for x in CAL:
        if x['onset'] is None: continue
        n_h += 1
        i = first_idx(x, r, h)
        if i is None: continue
        t = int(x['chk'][k]['t'][i]); s = int(x['chk'][k]['s'][i])
        if t >= x['onset']:
            det += 1; continue
        g = x['onset'] - t; gaps.append(g)
        os_ = x.get('onset_sent')
        if os_ is None: continue
        where['같은 문장' if s == os_ else '바로 앞 문장' if s == os_ - 1 else '2문장 이상 앞'] += 1
        for w in win:
            if g <= w: win[w] += 1
    n_pre = len(gaps)
    row = {'scorer': f'{m}|{k}', 'n_hall': n_h, 'recall': det / n_h, 'pre_onset': n_pre / n_h,
           'gap_tok_p25': np.percentile(gaps, 25), 'gap_tok_med': np.median(gaps), 'gap_tok_p75': np.percentile(gaps, 75)}
    row.update({f'pre_{kk}': v / max(n_pre, 1) for kk, v in where.items()})
    row.update({f'recall_if_tol{w}': (det + v) / n_h for w, v in win.items()})
    pre_rows.append(row)
pre_df = pd.DataFrame(pre_rows)
print(pre_df.round(3).to_string(index=False))

# 예시 5개 (B 문장 끝): 경보가 울린 문장과 환각이 시작된 문장
r = R('thr', 'end', model='B'); h = calibrate_arl(CAL, r); shown = 0
for x in CAL:
    if x['onset'] is None or shown >= 5: continue
    i = first_idx(x, r, h)
    if i is None or x['chk']['end']['t'][i] >= x['onset']: continue
    qa = x['chk']['end']['q'][i]
    os_ = x.get('onset_sent')
    qo = next((q for t, q in x['pts2'] if os_ is not None and t == x['sents'][os_]['t1']), None)
    print(f"\n[{x['task_type']}] 경보 문장: {Q_TEXTS[qa][:160]}")
    if qo is not None: print(f"          환각 시작 문장: {Q_TEXTS[qo][:160]}")
    shown += 1
""")

md(r"""## 6. 학습 규칙 — 판정 지점별 "이미 환각이 시작됐나" 예측

정답: 환각 응답에서 시작 토큰 이후의 판정 지점은 1, 그 전과 멀쩡한 응답은 0. 학습 데이터는 cal-L뿐이다.
cal-L 안에서는 근거 문서 단위 5겹 교차 예측값으로 보정·선택하고, dev-L에는 cal-L 전체로 학습한 모델을 적용한다.""")
code(r"""
FEATS = ['zB', 'zE', 'is_end', 'k', 'sent_idx', 't', 'prevB', 'prevE', 'dE', 'maxB_sent', 'maxE_sent',
         'prev_endB_max', 'prev_endE_max', 'prev_endB_last', 'task']
B_ONLY = ['zB', 'is_end', 'k', 'sent_idx', 't', 'prevB', 'maxB_sent', 'prev_endB_max', 'prev_endB_last', 'task']
E_ONLY = ['zE', 'is_end', 'k', 'sent_idx', 't', 'prevE', 'dE', 'maxE_sent', 'prev_endE_max', 'task']

def feats(x, kind, mB='B', mE='E2'):
    c = x['chk'][kind]
    zB, zE = Z[mB][c['q']], Z[mE][c['q']]
    n = len(c['t']); F = np.full((n, len(FEATS)), np.nan, dtype=np.float32)
    pe_B, pe_E = [], []
    cur, mxB, mxE, pB, pE = -1, -np.inf, -np.inf, np.nan, np.nan
    for i in range(n):
        s = c['s'][i]
        if s != cur:
            cur, mxB, mxE, pB, pE = s, -np.inf, -np.inf, np.nan, np.nan
        mxB, mxE = max(mxB, zB[i]), max(mxE, zE[i])
        F[i] = [zB[i], zE[i], c['end'][i], c['k'][i], s, c['t'][i], pB, pE, zE[i] - pE, mxB, mxE,
                max(pe_B) if pe_B else np.nan, max(pe_E) if pe_E else np.nan, pe_B[-1] if pe_B else np.nan,
                TASK_ID[x['task_type']]]
        pB, pE = zB[i], zE[i]
        if c['end'][i]:
            pe_B.append(zB[i]); pe_E.append(zE[i])
    y = np.zeros(n, dtype=np.int8) if x['onset'] is None else (c['t'] >= x['onset']).astype(np.int8)
    return F, y

def fit_learned(name, kind, cols, mB='B', mE='E2'):
    idx = [FEATS.index(c) for c in cols]
    Fc = [feats(x, kind, mB, mE) for x in CAL]
    X = np.concatenate([f[0][:, idx] for f in Fc]); Y = np.concatenate([f[1] for f in Fc])
    G = np.concatenate([[x['ref_key']] * len(f[1]) for x, f in zip(CAL, Fc)])
    lens = [len(f[1]) for f in Fc]; offs = np.r_[0, np.cumsum(lens)]
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                l2_regularization=1.0, random_state=0)
    oof = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(X, Y, G):
        oof[te] = mk().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    LR[name] = {x['id']: logit(oof[offs[j]:offs[j + 1]]) for j, x in enumerate(CAL)}
    full = mk().fit(X, Y)
    for x in DEV:
        F, _ = feats(x, kind, mB, mE)
        LR[name][x['id']] = logit(full.predict_proba(F[:, idx])[:, 1]) if len(F) else np.zeros(0)
    return full

t0 = time.perf_counter()
LEARNED = []
for kind in ['4', '2']:
    for tag, cols in [('B+E2', FEATS), ('E2', E_ONLY), ('B', B_ONLY)]:
        nm = f'lr_{tag}_{kind}'
        fit_learned(nm, kind, cols)
        LEARNED.append(R('learned', kind, lr=nm))
print(f'학습 규칙 {len(LEARNED)}개, {time.perf_counter()-t0:.0f}초')
""")

md(r"""## 7. 후보 규칙 전체 — cal-L에서 보정·평가하고 대표 규칙을 고른다""")
code(r"""
CANDS = []
for m in ['B', 'E2']:
    for k in ['end', '4', '2']:
        CANDS.append(R('thr', k, model=m))
    for k in ['4', '2']:
        for lam in [0.3, 0.5]:
            CANDS.append(R('ewma', k, model=m, lam=lam))
        for mm in [2, 3]:
            CANDS.append(R('consec', k, model=m, m=mm))
        clean_z = np.concatenate([Z[m][x['chk'][k]['q']] for x in CAL if x['onset'] is None])
        for qq in [0.90, 0.97, 0.99]:
            kref = round(float(np.quantile(clean_z, qq)), 4)
            for rs in [True, False]:
                CANDS.append(R('cusum', k, model=m, kref=kref, reset_sent=rs, kq=qq))
CANDS += LEARNED
print(f'후보 {len(CANDS)}개')

t0 = time.perf_counter(); rows = []
for r in CANDS:
    h = calibrate_arl(CAL, r)
    e = evaluate(CAL, r, h)
    rows.append({'rule': r['name'], 'type': r['type'], 'kind': r['kind'], 'h100': h, **{f'cal_{k}': v for k, v in e.items()}})
cal_df = pd.DataFrame(rows)
print(f'{time.perf_counter()-t0:.0f}초')
print(cal_df.sort_values('cal_recall', ascending=False).head(15)[
    ['rule', 'cal_recall', 'cal_delay_med', 'cal_pre_onset', 'cal_missed', 'cal_realized_ARL0', 'cal_calls']].round(3).to_string(index=False))

# 대표 규칙: cal-L 지연 ≤ 9 중 cal-L 재현율 최대, 동점이면 판정 횟수 적은 것 (결과 보기 전 고정한 규칙)
elig = cal_df[cal_df.cal_delay_med <= 9].sort_values(['cal_recall', 'cal_calls'], ascending=[False, True])
REP = next(r for r in CANDS if r['name'] == elig.iloc[0].rule)
print('\n대표 규칙 (cal-L에서 선택):', REP['name'])
""")

md(r"""## 8. dev-L 평가 — 모든 후보(보고)와 대표 규칙(판정)""")
code(r"""
rows = []
for r in CANDS:
    h100 = cal_df.set_index('rule').loc[r['name'], 'h100']
    e = evaluate(DEV, r, h100)
    h10 = calibrate_flag(CAL, r, 0.10)
    e10 = evaluate(DEV, r, h10)
    rows.append({'rule': r['name'], 'type': r['type'], 'kind': r['kind'], **e,
                 'svc_recall': e10['recall'], 'svc_delay_med': e10['delay_med'], 'svc_clean_flag': e10['clean_flag']})
dev_df = pd.DataFrame(rows)
cols = ['rule', 'recall', 'delay_med', 'pre_onset', 'missed', 'realized_ARL0', 'svc_recall', 'svc_delay_med', 'svc_clean_flag', 'calls']
print('=== dev-L 상위 15 (보고용, 선택에는 쓰지 않음) ===')
print(dev_df.sort_values('recall', ascending=False).head(15)[cols].round(3).to_string(index=False))
print('\n=== 유형별 최고 (dev-L, 보고용) ===')
print(dev_df.loc[dev_df.groupby('type').recall.idxmax()][cols].round(3).to_string(index=False))
""")

md(r"""## 9. 사전 기준 판정 — 대표 규칙 vs 기준선 (쌍체 부트스트랩, 환각 응답 재표집 1,000회)""")
code(r"""
def boot(a, b, ha, hb, items=DEV, n=1000, seed=0):
    da, la = per_resp(items, a, ha); db, lb = per_resp(items, b, hb)
    rng = np.random.default_rng(seed); N = len(da); rd, dd = [], []
    for _ in range(n):
        i = rng.integers(0, N, N)
        rd.append(da[i].mean() - db[i].mean()); dd.append(np.nanmedian(la[i]) - np.nanmedian(lb[i]))
    q = lambda v: np.nanpercentile(v, [2.5, 97.5])
    return {'A': a['name'], 'B': b['name'], 'recall_A': da.mean(), 'recall_B': db.mean(),
            'recall_diff': da.mean() - db.mean(), 'rd_lo': q(rd)[0], 'rd_hi': q(rd)[1],
            'delay_A': np.nanmedian(la), 'delay_B': np.nanmedian(lb), 'delay_diff': np.nanmedian(la) - np.nanmedian(lb),
            'dd_lo': q(dd)[0], 'dd_hi': q(dd)[1]}

h_of = lambda r: calibrate_arl(CAL, r)
B_END, B_4, E2_4 = R('thr', 'end', model='B'), R('thr', '4', model='B'), R('thr', '4', model='E2')
bt = [boot(REP, B_END, h_of(REP), h_of(B_END)), boot(REP, B_4, h_of(REP), h_of(B_4)), boot(REP, E2_4, h_of(REP), h_of(E2_4))]
boot_df = pd.DataFrame(bt)
print(boot_df.round(3).to_string(index=False))

rep_dev = evaluate(DEV, REP, h_of(REP))
b0 = bt[0]
C_rec, C_ci, C_del = rep_dev['recall'] >= 0.531, b0['rd_lo'] > 0, rep_dev['delay_med'] <= 9
R2 = C_rec and C_ci and C_del
print(f"\n대표 규칙 {REP['name']}")
print(f"  재현율 {rep_dev['recall']:.3f} ≥ 0.531: {'충족' if C_rec else '미달'}")
print(f"  B 문장 끝 대비 {b0['recall_diff']:+.3f} [{b0['rd_lo']:+.3f}, {b0['rd_hi']:+.3f}], 하한 > 0: {'충족' if C_ci else '미달'}")
print(f"  지연 중앙값 {rep_dev['delay_med']} ≤ 9: {'충족' if C_del else '미달'}")
print('R2 주 기준:', '충족' if R2 else '미달')
h10 = calibrate_flag(CAL, REP, 0.10); e10 = evaluate(DEV, REP, h10)
print(f"보조(보고): 서비스 10% 재현율 {e10['recall']:.3f}, 지연 {e10['delay_med']}, 실제 표시율 {e10['clean_flag']:.1%} "
      f"(참고: B 문장 끝 0.395/21, B s=4 0.331/15)")
""")

md(r"""## 10. 대표 규칙 계열을 시드 판정기에 적용 (보고) — 규칙의 이득이 특정 판정기에만 있는가""")
code(r"""
rob = []
for sm in SEEDS:
    if REP['type'] == 'learned':
        tag, kind = REP['lr'].split('_')[1], REP['kind']
        if 'E' not in tag:
            continue
        cols = FEATS if tag == 'B+E2' else E_ONLY
        nm = f'lr_{tag}_{kind}_{sm}'
        fit_learned(nm, kind, cols, mE=sm)
        r = R('learned', kind, lr=nm)
    else:
        kw = {k: v for k, v in REP.items() if k not in ('type', 'kind', 'name')}
        if kw.get('model') != 'E2':
            continue
        kw['model'] = sm
        if REP['type'] == 'cusum':          # 기준값 k는 같은 분위수로 다시 계산
            clean_z = np.concatenate([Z[sm][x['chk'][REP['kind']]['q']] for x in CAL if x['onset'] is None])
            kw['kref'] = round(float(np.quantile(clean_z, kw['kq'])), 4)
        r = R(REP['type'], REP['kind'], **kw)
    thr = R('thr', REP['kind'], model=sm)
    e_r, e_t = evaluate(DEV, r, h_of(r)), evaluate(DEV, thr, h_of(thr))
    rob.append({'scorer': sm, 'rule_recall': e_r['recall'], 'rule_delay': e_r['delay_med'],
                'thr_recall': e_t['recall'], 'thr_delay': e_t['delay_med'], 'gain': e_r['recall'] - e_t['recall']})
rob_df = pd.DataFrame(rob)
print(rob_df.round(3).to_string(index=False) if len(rob_df) else '대표 규칙이 E2 점수를 쓰지 않아 해당 없음')
""")

md(r"""## 11. R3 — 오경보 보장 보정 (등각, α = 10%, δ = 10%)

cal-L의 멀쩡한 응답으로 "표시 비율의 90% 신뢰 상한 ≤ 10%"가 되는 가장 낮은 임계값을 고른다.
같은 근거 문서의 응답끼리는 독립이 아니므로 보장은 근사다. 그래서 dev-L 실제 표시율로 확인한다.""")
code(r"""
r3 = []
for r in [REP, B_END, E2_4, B_4]:
    hq, hc = calibrate_flag(CAL, r, 0.10), calibrate_conformal(CAL, r, 0.10, 0.10)
    eq, ec = evaluate(DEV, r, hq), evaluate(DEV, r, hc)
    r3.append({'rule': r['name'], 'quantile_flag': eq['clean_flag'], 'quantile_recall': eq['recall'],
               'conformal_flag': ec['clean_flag'], 'conformal_recall': ec['recall'],
               'recall_cost': eq['recall'] - ec['recall'], 'ok': ec['clean_flag'] <= 0.10})
r3_df = pd.DataFrame(r3)
print(r3_df.round(3).to_string(index=False))
R3 = bool(r3_df.iloc[:3].ok.all())
print('R3 (대표 규칙·B 문장 끝·E2 s=4 모두 dev-L 표시율 ≤ 10%):', '충족' if R3 else '미달')
""")

md(r"""## 12. 게이트웨이 판정기 결정 — 서비스 조건(표시 10%)에서 E2 s=4 vs B s=4""")
code(r"""
gw = boot(E2_4, B_4, calibrate_flag(CAL, E2_4, 0.10), calibrate_flag(CAL, B_4, 0.10))
keep = gw['rd_lo'] > 0 or (gw['rd_lo'] <= 0 <= gw['rd_hi'] and gw['dd_hi'] < 0)
print(f"재현율 {gw['recall_A']:.3f} vs {gw['recall_B']:.3f}: 차이 {gw['recall_diff']:+.3f} [{gw['rd_lo']:+.3f}, {gw['rd_hi']:+.3f}]")
print(f"지연 {gw['delay_A']} vs {gw['delay_B']}: 차이 {gw['delay_diff']:+.1f} [{gw['dd_lo']:+.1f}, {gw['dd_hi']:+.1f}]")
print('결정:', 'student_prefix 유지' if keep else '기본 판정기를 student_vark(+재판정)로 변경')
""")

md(r"""## 13. 가변 재판정 간격 (보고) — 점수가 높으면 자주, 낮으면 드물게

문장마다 2토큰에서 첫 판정, 이후 직전 점수가 τ 이상이면 2토큰 뒤, 아니면 s_lo토큰 뒤에 다시 판정한다(문장 끝은 항상 판정).
임계값 규칙으로 보정·평가하고, 고정 간격(2·4·8토큰)과 판정 횟수 대비 재현율·지연을 비교한다.""")
code(r"""
def add_adaptive(m, s_lo, tau, tag):
    kind = f'ad_{m}_{s_lo}_{tag}'          # 이름이 겹치면 캐시가 섞이므로 분위수로 이름 짓기
    for x in CAL + DEV:
        c = x['chk']['2']; z = Z[m][c['q']]; keep = []; due = 2; cur = -1
        for i in range(len(c['t'])):
            if c['s'][i] != cur:
                cur, due = c['s'][i], 2
            if c['end'][i] or c['k'][i] >= due:
                keep.append(i); due = c['k'][i] + (2 if z[i] >= tau else s_lo)
        keep = np.array(keep, dtype=np.int64)
        x['chk'][kind] = {f: v[keep] for f, v in c.items() if f != 'new'}
        S_ = x['chk'][kind]['s']; x['chk'][kind]['new'] = np.r_[True, S_[1:] != S_[:-1]] if len(S_) else np.zeros(0, bool)
    return kind

ad_rows = []
for m in ['B', 'E2']:
    clean_z = np.concatenate([Z[m][x['chk']['2']['q']] for x in CAL if x['onset'] is None])
    for k in ['end', '8', '4', '2']:
        r = R('thr', k, model=m); e = evaluate(DEV, r, h_of(r))
        ad_rows.append({'model': m, 'policy': f'고정 {k}', **{kk: e[kk] for kk in ['calls', 'recall', 'delay_med', 'pre_onset']}})
    for s_lo in [4, 8]:
        for qq in [0.5, 0.8, 0.95]:
            kind = add_adaptive(m, s_lo, float(np.quantile(clean_z, qq)), int(qq * 100))
            r = R('thr', kind, model=m); e = evaluate(DEV, r, h_of(r))
            ad_rows.append({'model': m, 'policy': f'가변 s_lo={s_lo}, τ=멀쩡한 점수 {int(qq*100)}%', **{kk: e[kk] for kk in ['calls', 'recall', 'delay_med', 'pre_onset']}})
ad_df = pd.DataFrame(ad_rows)
print(ad_df.round(3).to_string(index=False))
""")

md("## 14. 판정 요약과 저장")
code(r"""
print('=== R2 사전 기준 ===')
print('R2-0 재현     :', '충족' if R20 else '미달')
print('대표 규칙     :', REP['name'])
print('R2 주 기준    :', '충족' if R2 else '미달',
      f"(재현율 {rep_dev['recall']:.3f}, B 문장 끝 대비 {b0['recall_diff']:+.3f} [{b0['rd_lo']:+.3f}, {b0['rd_hi']:+.3f}], 지연 {rep_dev['delay_med']})")
print('R3 등각 보정  :', '충족' if R3 else '미달')
print('게이트웨이    :', 'student_prefix 유지' if keep else 'student_vark(+재판정)로 변경')

out = {'r2_cal.csv': cal_df, 'r2_devL.csv': dev_df, 'r2_bootstrap.csv': boot_df, 'r2_preonset.csv': pre_df,
       'r2_conformal.csv': r3_df, 'r2_adaptive.csv': ad_df, 'r2_seed_robust.csv': rob_df,
       'r2_gateway.csv': pd.DataFrame([gw])}
for f, d in out.items():
    d.to_csv(os.path.join(R2_DIR, f), index=False); d.to_csv(f, index=False)
json.dump({'R2_0': bool(R20), 'rep': REP['name'], 'R2': bool(R2), 'R3': bool(R3), 'keep_prefix': bool(keep),
           'rep_dev': rep_dev, 'rep_svc10': e10}, open('r2_summary.json', 'w'), default=float, ensure_ascii=False, indent=1)
shutil.copy('r2_summary.json', os.path.join(R2_DIR, 'r2_summary.json'))
os.system('zip -q r2_results.zip ' + ' '.join(list(out) + ['r2_summary.json']))
print('저장 완료: 드라이브 selfproject/r2/, 업로드용 r2_results.zip')
try:
    from google.colab import files; files.download('r2_results.zip')
except Exception:
    pass
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r2_sequential_rules.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
