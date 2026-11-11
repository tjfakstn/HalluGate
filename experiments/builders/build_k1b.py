"""K1b 노트북 생성: 직접 모순 합성 데이터를 더해 F1 base를 다시 학습하고(F1과 같은 설정·시드 42), 새 조합(새 F1 + I2)을
RAGTruth cal-L(K1-3)과 최신 LLM 새 생성 세트 m1_fresh(K1-0·K1-1·K1-2)에서 기존 조합과 비교한다. 사전 기준은 SSOT §6 K1.
학습 셀은 F1 노트북, 조합·비교 함수는 I3 노트북, 최신 LLM 응답 판정·지표는 M1 노트북 코드를 그대로 쓴다. 배포 가중치(final/)는 건드리지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')

def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    m = importlib.util.module_from_spec(spec); argv = sys.argv; spec.loader.exec_module(m); sys.argv = argv
    return m
F1, X1, I3, M1 = _load('build_f1'), _load('build_x1'), _load('build_i3'), _load('build_m1')
L3 = M1.L3

def sub(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:70])
    return s.replace(a, b)
def _src(c): return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])

ENV = sub(F1.ENV, "R1_DIR = FINAL_DIR = os.path.join(OUT_DIR, 'final')",
          "R1_DIR = FINAL_DIR = K1_DIR = os.path.join(OUT_DIR, 'k1')          # 배포 가중치(final/)와 따로\n"
          "OLD_DIR, I2_DIR = os.path.join(OUT_DIR, 'final'), os.path.join(OUT_DIR, 'i2')\n"
          "M1_DIRS = {'fresh': os.path.join(OUT_DIR, 'm1_fresh'), 'm1': os.path.join(OUT_DIR, 'm1')}\n"
          "for p in [os.path.join(K1_DIR, 'synth_train.json'), os.path.join(OLD_DIR, 'scores_base.npy'), os.path.join(I2_DIR, 'scores_i2.npy'),\n"
          "          os.path.join(OLD_DIR, 'judge_base', 'judge_config.json')] + \\\n"
          "         [os.path.join(d, f'{n}_devL.json') for d in M1_DIRS.values() for n in ['cases', 'gen', 'label', 'scores']]:\n"
          "    assert os.path.exists(p), f'{p} 없음 (K1a·m1_fresh_eval 먼저)'")
FINALIZE = next(_src(c) for c in F1.cells if _src(c).startswith("\nLOCAL = '/content/final_local'") or _src(c).startswith("LOCAL = '/content/final_local'"))
FINALIZE = sub(FINALIZE, "LOCAL = '/content/final_local'", "LOCAL = '/content/k1_local'")
FINALIZE = sub(FINALIZE, "for name in ['mini', 'base']:          # 짧은 MiniLM 먼저 (저장 경로 점검)", "for name in ['base']:")
I3_CORE = sub(I3.I3_CORE, "def report(cal, dev, name, kind='8'):", "def report_e(cal, dev, name, kind='8'):")   # M1 report와 이름 겹침 방지
M1A = [(r['cid'], r['i'], r['type'], r['validity'])
       for r in json.load(open(os.path.join(ROOT, 'docs', 'analysis', 'm1a', 'm1a_reviews.json')))]

# ---- K1B_* 셀은 smoke_k1.py가 확인한다 ----
K1B_SYN = r"""
# K1a 합성 문장을 학습 판정 지점에 더한다. F1 base는 문장 앞부분만 보므로 바뀐 문장 하나의 지점만 추가(규칙은 위 셀과 같음)
import pickle
SYN = json.load(open(os.path.join(K1_DIR, 'synth_train.json')))
_hold = {x['ref_key'] for k in ['calL', 'devL'] for x in pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))['items'][k]}
assert not _hold & {c['ref_key'] for c in SYN}, '합성 데이터에 cal-L·dev-L 근거가 섞임'
assert {c['ref_key'] for c in SYN} <= {x['ref_key'] for x in TRAIN_ITEMS}, '학습 응답에 없는 근거'
for c in SYN: REFS.setdefault(c['ref_key'], build_reference(sources[c['ref_key']]))

def syn_points(c):
    text = c['sentence']
    ts, te = tokenize_offsets(text)
    x = {'text': text, 'ts': ts, 'te': te, 'sents': [{'a': 0, 'b': len(text), 't0': 0, 't1': len(ts) - 1}]}
    cs = [a + (len(text[a:b]) - len(text[a:b].lstrip())) for a, b in zip(ts, te)]
    cs = [s if s < b else a for s, a, b in zip(cs, ts, te)]
    o = None
    if c['kind'] == 'contra':
        k = text.find(c['phrase'])
        if k < 0: k = text.lower().find(c['phrase'].lower())
        a, b = (k, k + len(c['phrase'])) if k >= 0 else (0, len(text))
        o = next((t for t in range(len(ts)) if min(te[t], b) > max(cs[t], a)), 0)
    out = []
    for t, txt in points(x, 2):
        if not txt: continue
        out.append((txt, 'neg' if o is None else 'pre' if t < o else 'win' if t <= o + WIN_TOK else 'post'))
    return out

P_CAT = list(P_CAT); _n0 = len(P_TXT)
for c in SYN:
    lst = []
    for txt, cat in syn_points(c):
        g = len(P_TXT); P_KEY.append(c['ref_key']); P_TXT.append(txt); P_CAT.append(cat)
        if cat in ('win', 'post'): lst.append(g)
    if lst and not any(P_CAT[g] == 'win' for g in lst): P_CAT[lst[0]] = 'win'
    TR.append({'groups': [[g for g in lst if P_CAT[g] == 'win']] if lst else []})
P_CAT = np.array(P_CAT)
print(f"합성 문장 {len(SYN)} (모순 {sum(c['kind'] == 'contra' for c in SYN)}, 바꿔쓰기 {sum(c['kind'] == 'para' for c in SYN)}) → 판정 지점 +{len(P_TXT) - _n0:,} |",
      pd.Series(P_CAT[_n0:]).value_counts().to_dict(), '| 전체', pd.Series(P_CAT).value_counts().to_dict())
"""

K1B_COMBO = r"""
# 기존 조합(F1 base + I2)과 새 조합(K1 F1 + I2). 표준화·임계값은 I3와 같은 방식으로 cal-L에서만
SC['F_k1'] = SC['F_base']
SC['F_old'] = np.load(os.path.join(OLD_DIR, 'scores_base.npy'))
E_KEYS, E_TEXTS, E_NEED, E_MAP = build_eval_points(CAL + DEV, RESP_TEXT)
SC['I2'] = np.load(os.path.join(I2_DIR, 'scores_i2.npy'))
assert len(SC['I2']) == len(E_KEYS), 'I2 점수 길이가 평가 지점 수와 다름'
SC['Fb'] = to_E(CAL + DEV, 'F_old', E_MAP, len(E_KEYS))
SC['Kb'] = to_E(CAL + DEV, 'F_k1', E_MAP, len(E_KEYS))
LS = {lab: (remap(c, E_MAP), remap(d, E_MAP)) for lab, (c, d) in LAB_SETS.items()}
CALo, CALc = LS['orig'][0], LS['C1'][0]
_r = compare_e(CALo, 'I2', 'Fb')
assert abs(_r['thr_base'] - 3.968751) < 1e-3 and abs(_r['thr_new'] - 4.562501) < 1e-3, 'F1 base·I2 재현 실패 (I3와 다름)'
STATS = {m: clean_stats(CALo, f'{m}|8') for m in ['Fb', 'Kb', 'I2']}
Z = {m: (SC[m] - mu) / sd for m, (mu, sd) in STATS.items()}
SC['OLD'], SC['NEW'] = (Z['Fb'] + Z['I2']) / 2, (Z['Kb'] + Z['I2']) / 2
THR_OLD, THR_NEW = conformal_flag(CALo, 'OLD|8'), conformal_flag(CALo, 'NEW|8')
THR_KF = conformal_flag(CALo, 'Kb|8')
assert abs(THR_OLD - 2.991638706756592) < 1e-3, f'기존 조합 임계값 재현 실패 {THR_OLD}'
COMBO_OLD = {'F_mean': STATS['Fb'][0], 'F_std': STATS['Fb'][1], 'I_mean': STATS['I2'][0], 'I_std': STATS['I2'][1], 'threshold': THR_OLD}
COMBO_NEW = {'F_mean': STATS['Kb'][0], 'F_std': STATS['Kb'][1], 'I_mean': STATS['I2'][0], 'I_std': STATS['I2'][1], 'threshold': THR_NEW}
print('표준화 (평균, 표준편차):', {m: tuple(round(v, 4) for v in s) for m, s in STATS.items()})
print(f'임계값: 기존 조합 {THR_OLD:.4f} (재현) | 새 조합 {THR_NEW:.4f} | 새 F1 단독 {THR_KF:.4f}')

# K1-3 (안전, RAGTruth cal-L 8토큰 서비스 10%)
ROWS = []
for lab, cal in [('orig', CALo), ('C1', CALc)]:
    for task in [None] + TASKS:
        ROWS.append({'labels': lab, 'model': 'NEW', **compare_e(cal, 'NEW', 'OLD', task=task)})
        ROWS.append({'labels': lab, 'model': 'Kb', **compare_e(cal, 'Kb', 'Fb', task=task)})
cal_tab = pd.DataFrame(ROWS)
pd.set_option('display.width', 250)
print(cal_tab[['labels', 'model', 'task', 'recall_new', 'recall_base', 'delta', 'ci_lo', 'ci_hi', 'delay_new', 'delay_base', 'n_hall']].round(3).to_string(index=False))
_row = lambda lab: next(r for r in ROWS if r['labels'] == lab and r['task'] == 'all' and r['model'] == 'NEW')
K1_3 = bool(_row('orig')['delta'] >= -0.02 and _row('C1')['delta'] >= -0.02)
print(f"=== K1-3 (안전, RAGTruth): 원래 정답 {_row('orig')['recall_base']:.3f} → {_row('orig')['recall_new']:.3f} (Δ {_row('orig')['delta']:+.3f}), "
      f"C1 Δ {_row('C1')['delta']:+.3f} (각각 ≥ −0.02) → {'충족' if K1_3 else '미달'} ===")
DEV_ROWS = []
for lab in ['orig', 'C1']:
    cal, dev = LS[lab]
    for m in ['Fb', 'Kb', 'OLD', 'NEW']:
        DEV_ROWS += [{'labels': lab, **r} for r in report_e(cal, dev, m)]
dev_tab = pd.DataFrame(DEV_ROWS)
print('dev-L (보고만)'); print(dev_tab.round(3).to_string(index=False))
"""

K1B_M1 = r"""
# 최신 LLM 응답(M1 흐름) 다시 판정: 저장된 판정 지점·I2 점수는 그대로, F1 점수만 새 판정기(게이트웨이 judge.py)로
def zc(sf, si, C): return ((sf - C['F_mean']) / C['F_std'] + (si - C['I_mean']) / C['I_std']) / 2

def parse_labels(gens, raw):
    out = {}
    for c, r in raw.items():
        if c not in gens: continue
        labs, ok = parse(r['text'], len(sentences(gens[c]['text'])))
        if ok: out[c] = labs
    return out

def parity(j, cases, gens, old, n=30):
    # 기존 판정기로 저장 점수가 재현되는지 (지점 위치·앞 문맥이 M1과 같은지 확인)
    diffs = []
    for c in list(old)[:n]:
        g, ref = gens[c]['text'], j.prepare(cases[c]['ref_judge'])
        for a, cut, sf, si, _ in old[c][:3]:
            diffs.append(abs(j.judge(ref, g[a:cut].strip(), g[:a]).score - sf))
    return float(np.max(diffs)) if diffs else 0.0

def rescore(j, cases, gens, old, path, every=50):
    new = load_json(path) or {}
    todo = [c for c in old if c not in new]
    cache, t0 = {}, time.time()
    for k, c in enumerate(todo):
        r = cases[c]['ref_judge']
        if r not in cache: cache = {r: j.prepare(r)}
        g = gens[c]['text']
        new[c] = [[a, cut, j.judge(cache[r], g[a:cut].strip(), g[:a]).score, si, None] for a, cut, _, si, _ in old[c]]
        if (k + 1) % every == 0:
            save_json(new, path); print(f'  다시 판정 {len(new)}/{len(old)} ({time.time() - t0:.0f}s)', flush=True)
    save_json(new, path)
    return new

def with_combo(scores, C):
    return {c: [[a, cut, sf, si, zc(sf, si, C)] for a, cut, sf, si, _ in v] for c, v in scores.items()}

def paired(old_rows, new_rows, n_boot=2000, seed=2027):
    o = old_rows[old_rows.status == 'ok'].set_index('cid')
    n = new_rows.set_index('cid').loc[o.index]
    hal = ~o.clean.astype(bool)
    do, dn = o.flag_c[hal].astype(float).values, n.flag_c[hal].astype(float).values
    rng = np.random.default_rng(seed); m = len(do)
    bt = [dn[i].mean() - do[i].mean() for i in (rng.integers(0, m, m) for _ in range(n_boot))] if m else [np.nan]
    return {'n_hal': int(m), 'old': float(do.mean()) if m else np.nan, 'new': float(dn.mean()) if m else np.nan,
            'delta': float(dn.mean() - do.mean()) if m else np.nan,
            'ci_lo': float(np.percentile(bt, 2.5)), 'ci_hi': float(np.percentile(bt, 97.5))}

def sentence_flags(gens, scores, items, thr):
    # M1a 모순 문장: 그 문장의 판정 지점에서 경보가 울렸나
    out = []
    for cid, i, typ, val in items:
        if cid not in scores: continue
        a0 = sentences(gens[cid]['text'])[i - 1][0]
        out.append({'cid': cid, 'i': i, 'type': typ, 'validity': val,
                    'flag': bool(any(p[0] == a0 and p[4] >= thr for p in scores[cid]))})
    return pd.DataFrame(out)
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# K1b — 직접 모순 합성 데이터로 F1 base 재학습 · 새 조합 판정

## 왜 하나

M1a: 최신 LLM 환각 중 판정기가 놓친 진짜 환각의 **68%가 직접 모순**(근거의 단어를 쓰면서 누가·무엇을만 바꿈), 모순 46문장 중 경보 1개.
K1a에서 만든 **최소 수정 모순 + 충실한 바꿔쓰기**를 학습 데이터에 더해 F1 base를 한 번 더 학습하고, 배포 조합(F1 base + I2)의 F1 자리를 바꿨을 때
**최신 LLM 환각을 더 잡는지**, **RAGTruth 성능과 오경보는 지키는지** 본다.

## 무엇을 하나

1. F1 노트북과 **같은 데이터·설정·시드 42**로 DeBERTa-v3-base 학습, 학습 판정 지점에 합성 문장(바뀐 문장 하나)의 지점만 더함. 후보 1개라 선택 과정 없음
2. 새 F1 점수 + 저장된 I2 점수 → I3와 같은 방식으로 cal-L에서 표준화·임계값(서비스 10%) → **새 조합**
3. RAGTruth cal-L에서 기존 조합과 비교(K1-3), dev-L은 보고만
4. 최신 LLM 새 생성 세트(`m1_fresh/`, 아무도 보지 않은 응답)를 게이트웨이 `judge.py`로 새 F1만 다시 판정 → K1-0·K1-1·K1-2. 기존 M1 응답과 M1a 모순 문장은 보고만

## 사전 정의 기준 (SSOT §6 K1, 실행 전 고정)

- **K1-0(유효성)**: 검증 통과 모순 ≥ 300(K1a), 가린 점검 40개 중 진짜 모순 ≥ 85%(K1a 뒤 Claude Code), **새 생성 환각 응답 ≥ 60**(여기) — 미달이면 보류
- **K1-1(주)**: 새 생성 환각 응답 탐지율(아무 지점) **새 조합 − 기존 조합 ≥ +0.10**, 응답 쌍체 부트스트랩 95% 하한 > 0
- **K1-2(안전, 최신 LLM)**: 새 생성 멀쩡한 응답 표시율(새 조합) ≤ 15%
- **K1-3(안전, RAGTruth)**: cal-L 8토큰 서비스 10% 원래 정답 탐지율 ≥ 기존 조합 − 0.02(0.543 → 0.523 이상), C1도 ≥ −0.02
- 채택 = K1-0~3 모두 (판정기 구조가 같아 지연은 L3 그대로). 각 조합은 자기 cal-L 임계값을 쓴다

> **런타임: A100**(L4 가능). 학습 약 80~95분 + 채점·다시 판정 약 30분. 끊기면 다시 모두 실행 — 끝난 단계는 드라이브 `k1/`에서 건너뛴다.
> 먼저: **K1a**(`k1/synth_train.json`)와 **`m1_fresh_eval.ipynb`**(`m1_fresh/`)를 끝낼 것. **실행 전에 드라이브 휴지통 비우기**(가중치 약 370MB).
> 결과: 드라이브 `k1/`(가중치 `k1/judge_base` — 배포 `final/`은 그대로)와 `k1_results.zip`.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터와 보정 학습 정답 (F1과 같음)")
code(F1.DATA)
code(F1.TRAIN_SPLIT)
code(F1.PREP)
code(F1.LOADLAB)
code(F1.ADD)
md("## 3. 학습 판정 지점 (F1과 같음) + 합성 문장")
code(F1.LABELS)
code(K1B_SYN)
md("## 4. 근거 검색·평가 세트·평가 함수 (F1과 같음)")
code(F1.RETR)
code(F1.CORR)
code(F1.SERIES)
code(F1.EVAL_DEFS)
md("## 5. 학습 (F1 base와 같은 루프, 시드 42) — 저장 위치 `k1/`")
code(F1.TRAIN_BASE)
code(FINALIZE + r"""
# 게이트웨이 판정기 폴더로 쓰도록 배포 판정기의 설정 파일(입력 방식·검색 설정)을 복사 (임계값은 이 노트북에서 새로 정함)
for d in [os.path.join(LOCAL, 'judge_base'), os.path.join(K1_DIR, 'judge_base')]:
    try: shutil.copy(os.path.join(OLD_DIR, 'judge_base', 'judge_config.json'), os.path.join(d, 'judge_config.json'))
    except OSError as e: print('설정 파일 복사 실패', d, e)
""")
md("## 6. 새 조합과 K1-3 (RAGTruth cal-L, I3와 같은 방식)")
code(r"""
_tr = {str(r['id']): r for r in responses if r['split'] == 'train'}
assert all(x['id'] in _tr for x in CAL + DEV)
RESP_TEXT = {x['id']: _tr[x['id']]['response'] for x in CAL + DEV}
""" + X1.CONF)
code(I3.I2_EVAL + I3_CORE + K1B_COMBO)
md("""## 7. 최신 LLM 새 생성 세트 — K1-0·K1-1·K1-2

게이트웨이 `judge.py`로 새 F1 판정기를 불러, M1과 같은 판정 지점에서 F1 점수만 다시 낸다(I2 점수·지점 위치는 저장된 것 그대로).
먼저 기존 판정기로 저장 점수가 재현되는지 확인한다.""")
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
M1A = """ + repr(M1A) + "\n" + L3.L3_INPUTS + L3.L3_CORE + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + K1B_M1 + r"""
THR_F_OLD = read_config(os.path.join(OLD_DIR, 'judge_base'))['thresholds']['8']['svc10_orig']
jO = make_judge(os.path.join(OLD_DIR, 'judge_base'), DEVICE)
jK = make_judge(os.path.join(LOCAL, 'judge_base'), DEVICE)
assert jK.mode == 'retrieval'
RES = {}
for tag, mdir in M1_DIRS.items():
    cases, gens, raw, old = [load_json(os.path.join(mdir, f'{n}_devL.json')) for n in ['cases', 'gen', 'label', 'scores']]
    labels = parse_labels(gens, raw)
    pd_ = parity(jO, cases, gens, old)
    print(f'[{tag}] 응답 {len(old)} | 정답 파싱 {len(labels)}/{len(raw)} | 기존 판정기 재현 최대 차이 {pd_:.4f}')
    assert pd_ < 0.1, '저장 점수 재현 실패 — 판정 지점·앞 문맥이 M1과 다름'
    new = with_combo(rescore(jK, cases, gens, old, os.path.join(K1_DIR, f'scores_k1_{tag}.json')), COMBO_NEW)
    old_rows = response_rows(cases, gens, labels, old, THR_OLD, THR_F_OLD)
    new_rows = response_rows(cases, gens, labels, new, THR_NEW, THR_KF)
    RES[tag] = {'old_rows': old_rows, 'new_rows': new_rows, 'old_rep': report(old_rows), 'new_rep': report(new_rows),
                'paired': paired(old_rows, new_rows), 'gens': gens, 'old': old, 'new': new}
del jO; gc.collect()
""")
md("## 8. 사전 기준 판정")
code(r"""
F = RES['fresh']; P = F['paired']
K1_0_fresh = P['n_hal'] >= 60
K1_1 = bool(K1_0_fresh and P['delta'] >= 0.10 and P['ci_lo'] > 0)
K1_2 = bool(F['new_rep']['clean_combo']['rate'] <= 0.15)
for tag, R in RES.items():
    o, n, p = R['old_rep'], R['new_rep'], R['paired']
    print(f"[{tag}] 환각 응답 탐지(아무 지점) 기존 {fmt(o['hal_any'])} → 새 {fmt(n['hal_any'])} | Δ {p['delta']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]")
    print(f"       첫 H 이후 기존 {fmt(o['hal_strict'])} → 새 {fmt(n['hal_strict'])}")
    print(f"       멀쩡한 응답 표시율 기존 {fmt(o['clean_combo'])} → 새 {fmt(n['clean_combo'])} | 새 F1 단독 {fmt(n['clean_f1'])}")
    for t in TASKS:
        ro, rn = R['old_rows'], R['new_rows']
        h = (ro.status == 'ok') & (ro.task == t) & ~ro.clean.astype(bool)
        print(f"       {t:9} 환각 {int(h.sum()):3d} | 탐지 기존 {ro.flag_c[h].astype(bool).mean():.3f} → 새 {rn.set_index('cid').loc[ro.cid[h]].flag_c.astype(bool).mean():.3f}")
SENT = {}
for name, scores, thr in [('old', RES['m1']['old'], THR_OLD), ('new', RES['m1']['new'], THR_NEW)]:
    SENT[name] = sentence_flags(RES['m1']['gens'], scores, M1A, thr)
ct = SENT['old'][SENT['old'].type == 'contradiction']; cn = SENT['new'][SENT['new'].type == 'contradiction']
print(f"(보고) M1a 직접 모순 문장 경보: 기존 {int(ct.flag.sum())}/{len(ct)} → 새 {int(cn.flag.sum())}/{len(cn)} | "
      f"모든 H 문장: 기존 {int(SENT['old'].flag.sum())}/{len(SENT['old'])} → 새 {int(SENT['new'].flag.sum())}/{len(SENT['new'])}")
print(f"\n=== K1-0(새 생성 부분): 환각 응답 {P['n_hal']} (≥ 60) → {'충족' if K1_0_fresh else '미달 → 보류'} (나머지 K1-0은 K1a 결과) ===")
print(f"=== K1-1 (주): 새 생성 환각 응답 탐지율 {P['old']:.3f} → {P['new']:.3f} (Δ {P['delta']:+.3f}, 하한 {P['ci_lo']:+.3f}; 기준 ≥ +0.10, 하한 > 0) → {'충족' if K1_1 else '미달'} ===")
print(f"=== K1-2 (안전, 최신 LLM): 새 조합 멀쩡한 응답 표시율 {F['new_rep']['clean_combo']['rate']:.3f} (≤ 0.15) → {'충족' if K1_2 else '미달'} ===")
print(f"=== K1-3 (안전, RAGTruth): {'충족' if K1_3 else '미달'} ===")
print(f"=== K1: {'채택 조건 충족 (K1-0 점검 결과와 함께 최종 판정)' if (K1_0_fresh and K1_1 and K1_2 and K1_3) else '미채택'} ===")
""")
md("## 9. 저장과 zip 다운로드")
code(r"""
OUTL = '/content/k1_out'; os.makedirs(OUTL, exist_ok=True)
cal_tab.to_csv(os.path.join(OUTL, 'k1_calL.csv'), index=False)
dev_tab.to_csv(os.path.join(OUTL, 'k1_devL.csv'), index=False)
for tag, R in RES.items():
    m = R['old_rows'].merge(R['new_rows'][['cid', 'max_c', 'max_f', 'flag_c', 'flag_f', 'first_alarm_a', 'strict_det']], on='cid', suffixes=('_old', '_new'))
    m.to_csv(os.path.join(OUTL, f'k1_{tag}.csv'), index=False)
pd.concat([SENT['old'].assign(model='old'), SENT['new'].assign(model='new')]).to_csv(os.path.join(OUTL, 'k1_m1a_sentences.csv'), index=False)
summ = {'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu', 'n_syn': len(SYN),
        'combo_old': COMBO_OLD, 'combo_new': COMBO_NEW, 'thr_k1_alone': THR_KF,
        'K1_0_fresh': bool(K1_0_fresh), 'K1_1': K1_1, 'K1_2': K1_2, 'K1_3': K1_3,
        'calL_all_orig': _row('orig'), 'calL_all_C1': _row('C1'),
        'm1': {tag: {'paired': R['paired'], 'old': R['old_rep'], 'new': R['new_rep']} for tag, R in RES.items()},
        'm1a_contradiction': {'old': int(ct.flag.sum()), 'new': int(cn.flag.sum()), 'n': len(ct)}}
json.dump(summ, open(os.path.join(OUTL, 'k1_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(OUTL):
    try: shutil.copy(os.path.join(OUTL, f), os.path.join(K1_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/k1_results', 'zip', OUTL)
from google.colab import files
files.download(zp)
print(json.dumps({k: summ[k] for k in ['K1_0_fresh', 'K1_1', 'K1_2', 'K1_3']}, ensure_ascii=False))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'k1b_contradiction_retrain.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
