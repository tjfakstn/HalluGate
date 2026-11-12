"""K2b 노트북 생성: K2a의 최신 LLM 학습 응답(Sonnet 정답)을 더해 F1 base를 다시 학습하고(F1과 같은 설정·시드 42),
새 조합(새 F1 + I2)을 기존 조합과 비교한다. 임계값은 (A) RAGTruth cal-L 서비스 10%, (B) cal-M 최신 LLM 멀쩡한 응답 10% 두 가지.
주 평가 = dev-M(K2a, Opus 정답). 사전 기준은 SSOT §6 K2. 학습 셀은 F1, 조합·cal-L 비교는 K1b(I3), 최신 LLM 판정은 M1·K1b 코드를 그대로 쓴다.
배포 가중치(final/)는 건드리지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')

def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    m = importlib.util.module_from_spec(spec); argv = sys.argv; spec.loader.exec_module(m); sys.argv = argv
    return m
KB = _load('build_k1b')
F1, X1, I3, M1, L3 = KB.F1, KB.X1, KB.I3, KB.M1, KB.L3
sub = KB.sub

ENV = sub(F1.ENV, "R1_DIR = FINAL_DIR = os.path.join(OUT_DIR, 'final')",
          "R1_DIR = FINAL_DIR = K2_DIR = os.path.join(OUT_DIR, 'k2')          # 배포 가중치(final/)와 따로\n"
          "OLD_DIR, I2_DIR = os.path.join(OUT_DIR, 'final'), os.path.join(OUT_DIR, 'i2')\n"
          "# 최신 LLM 세트: (태그, 폴더, 파일 접미사)\n"
          "MSETS = [('devM', K2_DIR, 'devM'), ('calM', K2_DIR, 'calM'), ('fresh', os.path.join(OUT_DIR, 'm1_fresh'), 'devL'), ('m1', os.path.join(OUT_DIR, 'm1'), 'devL')]\n"
          "for p in [os.path.join(OLD_DIR, 'scores_base.npy'), os.path.join(I2_DIR, 'scores_i2.npy'), os.path.join(OLD_DIR, 'judge_base', 'judge_config.json')] + \\\n"
          "         [os.path.join(K2_DIR, f'{n}_train.json') for n in ['cases', 'gen', 'label']] + \\\n"
          "         [os.path.join(d, f'{n}_{s}.json') for _, d, s in MSETS for n in ['cases', 'gen', 'label', 'scores']]:\n"
          "    assert os.path.exists(p), f'{p} 없음 (K2a·m1_fresh 먼저)'")

# M1 정답 지시문·문장 분할·파싱을 학습 노트북 이름과 겹치지 않게 따로 실행할 소스
K2_LABEL_SRC = M1.M1_LABEL

# ---- K2B_* 셀은 smoke_k2.py가 확인한다 ----
K2B_ADD = r"""
# K2a 최신 LLM 학습 응답(Sonnet 문장 정답)을 학습 판정 지점에 더한다. 지점·범주 규칙은 위 셀(F1)과 같다:
# H 문장은 정답 문구 위치(locate, R4b ADD와 같음)부터 환각, 나머지 문장은 음성. 근거는 판정 때와 같게(QA = 검색된 조각)
import pickle
def _lj(p): return json.load(open(p))
K2T = {n: _lj(os.path.join(K2_DIR, f'{n}_train.json')) for n in ['cases', 'gen', 'label']}
_hold = {x['ref_key'] for k in ['calL', 'devL'] for x in pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))['items'][k]}
assert not _hold & {c['sid'] for c in K2T['cases'].values()}, 'K2 학습 응답에 cal-L·dev-L 근거가 섞임'
_K2L = {'SEG': seg, 'json': json, 're': __import__('re')}
exec(K2_LABEL_SRC, _K2L)                           # sentences, parse (M1과 같은 코드, 이 노트북 이름과 분리)
K2_MAX_RATIO = 1.0                                  # 구현 보충(SSOT §6 K2): 최신 LLM 지점 수 ≤ 기존 학습 지점 수 (응답 단위 무작위, 시드 2032)

def k2_item(cid, c, text):
    key = c['sid'] if c['flow'] == 'direct' else f'k2q:{cid}'
    x = prep({'response': text, 'id': cid, 'task_type': c['task'], 'model': c['gen'], 'source_id': key, 'labels': []})
    return key, x

def k2_points(x, labs, text):
    ss = _K2L['sentences'](text)
    starts = [a for a, _, _ in ss]
    cs = content_starts(x)
    son = []
    for sn in x['sents']:
        i = next((k for k, a in enumerate(starts) if sn['a'] <= a < sn['b']), None)
        lab = labs.get(i + 1) if i is not None else None
        son.append(locate(x, sn, lab.get('phrase'), cs) if lab and lab['label'] == 'H' else None)
    sid = {sn['t0']: i for i, sn in enumerate(x['sents'])}
    out = []
    for t, txt in points(x, 2):
        if not txt: continue
        si = sid[x['sent_of'][t]]; o = son[si]
        cat = 'neg' if o is None else 'pre' if t < o else 'win' if t <= o + WIN_TOK else 'post'
        out.append((si, txt, cat))
    return out, sum(o is not None for o in son)

K2_RESP, K2_NH, K2_NPARSE, K2_NVALID = [], 0, 0, 0
for cid, c in K2T['cases'].items():
    g = K2T['gen'].get(cid)
    if not g or g.get('stop') == 'refusal' or not g['text'].strip() or not _K2L['sentences'](g['text']): continue
    K2_NVALID += 1
    raw = K2T['label'].get(cid)
    if raw is None: continue
    labs, ok = _K2L['parse'](raw['text'], len(_K2L['sentences'](g['text'])))
    if not ok: continue
    K2_NPARSE += 1
    key, x = k2_item(cid, c, g['text'])
    pts, nh = k2_points(x, labs, g['text'])
    if not pts: continue
    K2_NH += nh
    K2_RESP.append({'cid': cid, 'key': key, 'ref': c['ref_judge'], 'pts': pts, 'nh': nh, 'gen': c['gen'], 'task': c['task']})
_n_orig, _n_k2 = len(P_TXT), sum(len(r['pts']) for r in K2_RESP)
_order = list(range(len(K2_RESP))); random.Random(2032).shuffle(_order)
K2_USE, _tot = [], 0
for k in _order:
    if _tot + len(K2_RESP[k]['pts']) > K2_MAX_RATIO * _n_orig: continue
    K2_USE.append(K2_RESP[k]); _tot += len(K2_RESP[k]['pts'])
P_CAT = list(P_CAT); _n0 = len(P_TXT)
for r in K2_USE:
    REFS.setdefault(r['key'], r['ref'])
    groups = {}
    for si, txt, cat in r['pts']:
        g = len(P_TXT); P_KEY.append(r['key']); P_TXT.append(txt); P_CAT.append(cat)
        if cat in ('win', 'post'): groups.setdefault(si, []).append(g)
    gl = []
    for si, lst in groups.items():
        win = [g for g in lst if P_CAT[g] == 'win']
        if not win: P_CAT[lst[0]] = 'win'; win = [lst[0]]
        gl.append(win)
    TR.append({'groups': gl})
P_CAT = np.array(P_CAT)
K2_STATS = {'valid': K2_NVALID, 'parsed': K2_NPARSE, 'parse_ratio': K2_NPARSE / max(K2_NVALID, 1), 'H_sent': int(K2_NH),
            'resp_all': len(K2_RESP), 'pts_all': _n_k2, 'resp_used': len(K2_USE), 'pts_used': len(P_TXT) - _n0,
            'H_sent_used': int(sum(r['nh'] for r in K2_USE)), 'hal_resp_used': int(sum(r['nh'] > 0 for r in K2_USE)), 'pts_orig': _n_orig}
print('최신 LLM 학습 응답:', K2_STATS)
print('추가 지점 범주', pd.Series(P_CAT[_n0:]).value_counts().to_dict(), '| 전체', pd.Series(P_CAT).value_counts().to_dict())
"""

K2B_COMBO = sub(sub(sub(KB.K1B_COMBO, "SC['F_k1'] = SC['F_base']", "SC['F_k2'] = SC['F_base']"),
                    "SC['Kb'] = to_E(CAL + DEV, 'F_k1', E_MAP, len(E_KEYS))", "SC['Kb'] = to_E(CAL + DEV, 'F_k2', E_MAP, len(E_KEYS))"),
                "K1_3 = bool(", "K2_3 = bool(")
K2B_COMBO = sub(K2B_COMBO, "print(f\"=== K1-3 (안전, RAGTruth)", "print(f\"=== K2-3 (안전, RAGTruth, 임계값 A)")
K2B_COMBO = sub(K2B_COMBO, "→ {'충족' if K1_3 else '미달'} ===\")", "→ {'충족' if K2_3 else '미달'} ===\")")

K2B_EVAL = r"""
def rows_of(R, model, kind):
    # model: 'old' | 'new', kind: 'A' | 'B'
    sc = R['old'] if model == 'old' else R['new']
    return response_rows(R['cases'], R['gens'], R['labels'], sc, THR[model][kind], THR_F[model])

def clean_max(R, model):
    d = response_rows(R['cases'], R['gens'], R['labels'], R['old'] if model == 'old' else R['new'], 0.0, 0.0)
    d = d[d.status == 'ok']
    return d[d.clean.astype(bool)].max_c.values

def k2_decide(RES):
    P = paired(rows_of(RES['devM'], 'old', 'B'), rows_of(RES['devM'], 'new', 'B'))
    rep_new_B = report(rows_of(RES['devM'], 'new', 'B'))
    k2_1 = bool(P['delta'] >= 0.10 and P['ci_lo'] > 0)
    k2_2 = bool(rep_new_B['clean_combo']['rate'] <= 0.15)
    return P, rep_new_B, k2_1, k2_2
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# K2b — 최신 LLM 실제 응답으로 F1 base 재학습 · 새 조합 판정

## 왜 하나

배포 판정기(I3 조합: F1 base + I2)는 최신 Claude 응답의 환각을 10~14%밖에 잡지 못했고(M1·K1), 합성 모순 학습(K1)으로도 나아지지 않았다.
K2a에서 **최신 LLM이 실제로 쓴 응답**(학습용, Sonnet 문장 정답)을 모았다. 이것을 학습 데이터에 더해 경량 판정기가 최신 LLM 환각을 배우는지 본다.
임계값도 RAGTruth 기준(A)과 **최신 LLM 멀쩡한 응답 기준(B, cal-M 10%)** 두 가지로 본다.

## 무엇을 하나

1. F1 노트북과 **같은 데이터·설정·시드 42**로 DeBERTa-v3-base 학습, 학습 지점에 K2a 학습 응답의 모든 판정 지점을 더함(정답 문구 위치부터 환각, R4b와 같은 매핑).
   **구현 보충(실행 전)**: 학습 시간을 F1의 약 2배로 묶기 위해 최신 LLM 지점 수가 기존 학습 지점 수를 넘으면 응답 단위로 무작위(시드 2032)로 줄인다. 후보 1개, 선택 없음
2. 새 F1 점수 + 저장된 I2 점수 → I3와 같은 방식(cal-L 표준화) → **새 조합**. 임계값 A = cal-L 서비스 10%, 임계값 B = cal-M 멀쩡한 응답 표시율 10%(등각)
3. RAGTruth cal-L에서 기존 조합과 비교(K2-3), dev-L은 보고만
4. 최신 LLM 세트(dev-M·cal-M·m1_fresh·m1)를 게이트웨이 `judge.py`로 새 F1만 다시 판정(저장 지점·I2 점수 그대로) → K2-0·K2-1·K2-2

## 사전 정의 기준 (SSOT §6 K2, 실행 전 고정)

- **K2-0(유효성)**: train 정답 파싱 ≥ 90%, train H 문장 ≥ 500, cal-M 멀쩡한 응답 ≥ 400, dev-M 환각 응답 ≥ 150 — 미달이면 보류
- **K2-1(주)**: dev-M 환각 응답 탐지율(아무 지점) **새 조합(B) − 기존 조합(B) ≥ +0.10**, 응답 쌍체 부트스트랩 95% 하한 > 0
- **K2-2(안전, 최신 LLM)**: dev-M 멀쩡한 응답 표시율 새 조합(B) ≤ 15%
- **K2-3(안전, RAGTruth)**: 임계값 A에서 cal-L 8토큰 서비스 10% 원래 정답 탐지율 ≥ 기존 조합 − 0.02(0.543 → 0.523 이상), C1도 ≥ −0.02
- **채택 = K2-0~3 모두** → 배포 판정기 = 새 조합
- **임계값 B 채택(모델 채택과 별개)**: 배포 판정기(채택이면 새, 아니면 기존 조합)의 dev-M 멀쩡한 응답 표시율이 B에서 ≤ 15%면 서비스 임계값을 B로

> **런타임: A100**(L4 가능, 20GB 미만이면 첫 셀에서 멈춤). 학습 약 2.5~3.5시간 + 최신 LLM 세트 다시 판정 약 1시간. 끊기면 다시 모두 실행 — 끝난 단계는 드라이브 `k2/`에서 건너뛴다(학습 도중 끊기면 학습은 처음부터).
> 먼저: **K2a**(`k2/`의 train·calM·devM 파일)와 `m1_fresh/`. **실행 전에 드라이브 휴지통 비우기**(가중치 약 370MB).
> 결과: 드라이브 `k2/`(가중치 `k2/judge_base` — 배포 `final/`은 그대로)와 `k2_results.zip`.
""")

md("## 1. 환경 준비")
code(ENV + r"""
assert DEVICE == 'cuda' and torch.cuda.get_device_properties(0).total_memory >= 20 * 2**30, \
    f"GPU 메모리 부족: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'} — 런타임 유형을 A100으로 바꾸고 다시 실행"
""")
md("## 2. 데이터와 보정 학습 정답 (F1과 같음)")
code(F1.DATA)
code(F1.TRAIN_SPLIT)
code(F1.PREP)
code(F1.LOADLAB)
code(F1.ADD)
md("## 3. 학습 판정 지점 (F1과 같음) + 최신 LLM 학습 응답")
code(F1.LABELS)
code("K2_LABEL_SRC = " + repr(K2_LABEL_SRC) + "\n" + K2B_ADD)
md("## 4. 근거 검색·평가 세트·평가 함수 (F1과 같음)")
code(F1.RETR)
code(F1.CORR)
code(F1.SERIES)
code(F1.EVAL_DEFS)
md("## 5. 학습 (F1 base와 같은 루프, 시드 42) — 저장 위치 `k2/`")
code(F1.TRAIN_BASE)
code(sub(KB.FINALIZE, "LOCAL = '/content/k1_local'", "LOCAL = '/content/k2_local'") + r"""
for d in [os.path.join(LOCAL, 'judge_base'), os.path.join(K2_DIR, 'judge_base')]:
    try: shutil.copy(os.path.join(OLD_DIR, 'judge_base', 'judge_config.json'), os.path.join(d, 'judge_config.json'))
    except OSError as e: print('설정 파일 복사 실패', d, e)
""")
md("## 6. 새 조합과 K2-3 (RAGTruth cal-L, 임계값 A)")
code(r"""
_tr = {str(r['id']): r for r in responses if r['split'] == 'train'}
assert all(x['id'] in _tr for x in CAL + DEV)
RESP_TEXT = {x['id']: _tr[x['id']]['response'] for x in CAL + DEV}
""" + X1.CONF)
code(I3.I2_EVAL + KB.I3_CORE + K2B_COMBO)
md("""## 7. 최신 LLM 세트 다시 판정 · 임계값 B

게이트웨이 `judge.py`로 새 F1 판정기를 불러, K2a·M1과 같은 판정 지점에서 F1 점수만 다시 낸다(I2 점수·지점 위치는 저장된 것 그대로).
먼저 기존 판정기로 저장 점수가 재현되는지 확인한다. 임계값 B는 cal-M 멀쩡한 응답의 최대 점수로 등각 보정(10%)한다.""")
code("SRC = " + repr(M1.SRC) + r"""
import sys
os.makedirs('/content/hg/hallugate', exist_ok=True)
open('/content/hg/hallugate/__init__.py', 'w').write('')
for n, s in SRC.items(): open(f'/content/hg/hallugate/{n}.py', 'w').write(s)
sys.path.insert(0, '/content/hg')
from hallugate.judge import StudentJudge, read_config, input_mode
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
COMBO = COMBO_OLD
M1A = """ + repr(KB.M1A) + "\n" + L3.L3_INPUTS + L3.L3_CORE + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + KB.K1B_M1 + K2B_EVAL + r"""
THR_F = {'old': read_config(os.path.join(OLD_DIR, 'judge_base'))['thresholds']['8']['svc10_orig'], 'new': THR_KF}
jO = make_judge(os.path.join(OLD_DIR, 'judge_base'), DEVICE)
jK = make_judge(os.path.join(LOCAL, 'judge_base'), DEVICE)
assert jK.mode == 'retrieval'
RES = {}
for tag, mdir, suf in MSETS:
    cases, gens, raw, old = [load_json(os.path.join(mdir, f'{n}_{suf}.json')) for n in ['cases', 'gen', 'label', 'scores']]
    labels = parse_labels(gens, raw)
    pd_ = parity(jO, cases, gens, old)
    print(f'[{tag}] 응답 {len(old)} | 정답 파싱 {len(labels)}/{len(raw)} | 기존 판정기 재현 최대 차이 {pd_:.4f}')
    assert pd_ < 0.1, '저장 점수 재현 실패 — 판정 지점·앞 문맥이 저장 때와 다름'
    new = with_combo(rescore(jK, cases, gens, old, os.path.join(K2_DIR, f'scores_k2_{tag}.json')), COMBO_NEW)
    RES[tag] = {'cases': cases, 'gens': gens, 'labels': labels, 'old': with_combo(old, COMBO_OLD), 'new': new}
del jO; gc.collect()
THR = {'old': {'A': THR_OLD, 'B': conformal(clean_max(RES['calM'], 'old'))},
       'new': {'A': THR_NEW, 'B': conformal(clean_max(RES['calM'], 'new'))}}
print('임계값 (A = cal-L, B = cal-M 멀쩡한 응답 10%):', {m: {k: round(v, 4) for k, v in t.items()} for m, t in THR.items()})
""")
md("## 8. 사전 기준 판정")
code(r"""
P, REP_NEW_B, K2_1, K2_2 = k2_decide(RES)
K2_0 = bool(K2_STATS['parse_ratio'] >= 0.90 and K2_STATS['H_sent'] >= 500 and len(clean_max(RES['calM'], 'old')) >= 400 and P['n_hal'] >= 150)
ADOPT = bool(K2_0 and K2_1 and K2_2 and K2_3)
DEPLOY = 'new' if ADOPT else 'old'
REPS = {(tag, m, k): report(rows_of(RES[tag], m, k)) for tag in RES for m in ['old', 'new'] for k in ['A', 'B']}
THR_B_OK = bool(REPS[('devM', DEPLOY, 'B')]['clean_combo']['rate'] <= 0.15)
for tag in ['devM', 'fresh', 'm1', 'calM']:
    print(f'[{tag}]')
    for m in ['old', 'new']:
        for k in ['A', 'B']:
            r = REPS[(tag, m, k)]
            print(f"  {m:3} {k} (임계값 {THR[m][k]:.3f}) | 환각 응답 탐지 {fmt(r['hal_any'])} | 첫 H 이후 {fmt(r['hal_strict'])} | 멀쩡한 응답 표시율 {fmt(r['clean_combo'])}")
    for t in TASKS:
        ro, rn = rows_of(RES[tag], 'old', 'B'), rows_of(RES[tag], 'new', 'B')
        h = (ro.status == 'ok') & (ro.task == t) & ~ro.clean.astype(bool)
        if h.sum():
            print(f"    {t:9} 환각 {int(h.sum()):3d} | 탐지(B) 기존 {ro.flag_c[h].astype(bool).mean():.3f} → 새 {rn.set_index('cid').loc[ro.cid[h]].flag_c.astype(bool).mean():.3f}")
SENT = {}
for m in ['old', 'new']:
    for k in ['A', 'B']:
        SENT[(m, k)] = sentence_flags(RES['m1']['gens'], RES['m1'][m], M1A, THR[m][k])
ct = {mk: d[d.type == 'contradiction'] for mk, d in SENT.items()}
print('(보고) M1a 직접 모순 문장 경보:', {f'{m}{k}': f"{int(d.flag.sum())}/{len(d)}" for (m, k), d in ct.items()})
print(f"\n=== K2-0: train 파싱 {K2_STATS['parse_ratio']:.3f}, H 문장 {K2_STATS['H_sent']}, cal-M 멀쩡 {len(clean_max(RES['calM'], 'old'))}, dev-M 환각 {P['n_hal']} → {'충족' if K2_0 else '미달 → 보류'} ===")
print(f"=== K2-1 (주): dev-M 환각 응답 탐지율(B) 기존 {P['old']:.3f} → 새 {P['new']:.3f} (Δ {P['delta']:+.3f} [{P['ci_lo']:+.3f}, {P['ci_hi']:+.3f}]; 기준 ≥ +0.10, 하한 > 0) → {'충족' if K2_1 else '미달'} ===")
print(f"=== K2-2 (안전, 최신 LLM): 새 조합(B) dev-M 멀쩡한 응답 표시율 {REP_NEW_B['clean_combo']['rate']:.3f} (≤ 0.15) → {'충족' if K2_2 else '미달'} ===")
print(f"=== K2-3 (안전, RAGTruth): {'충족' if K2_3 else '미달'} ===")
print(f"=== K2: {'채택 → 배포 판정기 = 새 조합' if ADOPT else '미채택 → 배포 판정기 = 기존 조합'} | 임계값 B({DEPLOY}): dev-M 멀쩡한 응답 표시율 "
      f"{REPS[('devM', DEPLOY, 'B')]['clean_combo']['rate']:.3f} → {'B 채택' if THR_B_OK else 'B 미채택(A 유지)'} ===")
""")
md("## 9. 저장과 zip 다운로드")
code(r"""
OUTL = '/content/k2_out'; os.makedirs(OUTL, exist_ok=True)
cal_tab.to_csv(os.path.join(OUTL, 'k2_calL.csv'), index=False)
dev_tab.to_csv(os.path.join(OUTL, 'k2_devL.csv'), index=False)
for tag in RES:
    base = rows_of(RES[tag], 'old', 'A')
    for m in ['old', 'new']:
        for k in ['A', 'B']:
            if (m, k) == ('old', 'A'): continue
            r = rows_of(RES[tag], m, k)[['cid', 'max_c', 'max_f', 'flag_c', 'flag_f', 'first_alarm_a', 'strict_det']]
            base = base.merge(r.add_suffix(f'_{m}{k}').rename(columns={f'cid_{m}{k}': 'cid'}), on='cid', how='left')
    base.to_csv(os.path.join(OUTL, f'k2_{tag}.csv'), index=False)
pd.concat([d.assign(model=m, thr=k) for (m, k), d in SENT.items()]).to_csv(os.path.join(OUTL, 'k2_m1a_sentences.csv'), index=False)
summ = {'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu', 'train_k2': K2_STATS,
        'combo_old': COMBO_OLD, 'combo_new': COMBO_NEW, 'thr': THR, 'thr_k2_alone': THR_KF,
        'K2_0': K2_0, 'K2_1': K2_1, 'K2_2': K2_2, 'K2_3': K2_3, 'adopt': ADOPT, 'deploy': DEPLOY, 'thr_B_adopt': THR_B_OK,
        'devM_paired_B': P, 'calL_all_orig': _row('orig'), 'calL_all_C1': _row('C1'),
        'reports': {f'{t}|{m}|{k}': v for (t, m, k), v in REPS.items()},
        'm1a_contradiction': {f'{m}{k}': {'flag': int(d.flag.sum()), 'n': len(d)} for (m, k), d in ct.items()}}
json.dump(summ, open(os.path.join(OUTL, 'k2_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(OUTL):
    try: shutil.copy(os.path.join(OUTL, f), os.path.join(K2_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/k2_results', 'zip', OUTL)
from google.colab import files
files.download(zp)
print(json.dumps({k: summ[k] for k in ['K2_0', 'K2_1', 'K2_2', 'K2_3', 'adopt', 'thr_B_adopt']}, ensure_ascii=False))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'k2b_modern_llm_retrain.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
