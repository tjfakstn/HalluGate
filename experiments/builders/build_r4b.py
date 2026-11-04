"""R4b-B 노트북 생성: R4b-A(공개 LLM 재판정)가 만든 학습 정답 보정(labels_train.json)을 읽어,
R1b V0b와 같은 학습 방식으로 다시 학습해 원래 정답·보정 정답(R4 C1) 두 기준으로 평가한다.
데이터·전처리·근거 검색·평가·학습 코드는 R1b 노트북에서 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
r1b = json.load(open(os.path.join(NB_DIR, 'r1b_sentence_local_labels.ipynb')))
C = [''.join(c['source']) for c in r1b['cells'] if c['cell_type'] == 'code']
def find(prefix):
    return next(c for c in C if c.startswith(prefix))
def sub(s, a, b):
    assert a in s, a[:70]
    return s.replace(a, b, 1)

ENV = find("!pip -q install pysbd")
ENV = sub(ENV, "!pip -q install pysbd sentence-transformers tiktoken", "!pip -q install pysbd sentence-transformers tiktoken anthropic")
ENV = sub(ENV, "R1_DIR = os.path.join(OUT_DIR, 'r1b')\nR1_OLD = os.path.join(OUT_DIR, 'r1')",
          "R1_DIR = os.path.join(OUT_DIR, 'r4b')\nR1B_DIR = os.path.join(OUT_DIR, 'r1b')\nR4_DIR = os.path.join(OUT_DIR, 'r4')")
ENV = sub(ENV, "assert os.path.isdir(STUDENT_DIR), f'{STUDENT_DIR} 없음'",
          "assert os.path.isdir(STUDENT_DIR), f'{STUDENT_DIR} 없음'\nassert os.path.exists(os.path.join(R4_DIR, 'r4_corrected_labels.json')), 'R4 보정 정답(r4/r4_corrected_labels.json)이 있어야 한다'")
DATA = find("BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find("# 학습 응답: S2/S2-b의 gold-large")
PREP = find("ENC = tiktoken.get_encoding")
LABELS = find("def points(item, s):")
LABELS = sub(LABELS, "train_items = to_items(train)", "train_items = TRAIN_ITEMS")
LABELS = sub(LABELS, "    son = [s_onset(sn) for sn in x['sents']]",
             "    son = [s_onset(sn) for sn in x['sents']]\n"
             "    son = [o if o is not None else ADD_ONSET.get((x['id'], i)) for i, o in enumerate(son)]   # R4b: 재판정이 추가한 환각 문장")
RETR = find("from sentence_transformers import SentenceTransformer")
SERIES = find("SC = dict(Z0)")
EVAL_DEFS = find("def calibrate(items, scorer, arl0):")
AUCS = find("def resp_auc(items, m):")
TRAIN = find("from transformers import get_linear_schedule_with_warmup")
# Claude 재판정 셀은 build_r4b_claude.py가 만든 노트북에서 가져와 한 노트북으로 합친다
_cj = json.load(open(os.path.join(NB_DIR, 'r4b_a_claude_judge.ipynb')))
CJ = [''.join(c['source']) for c in _cj['cells'] if c['cell_type'] == 'code']
def cj(prefix):
    return next(c for c in CJ if c.startswith(prefix)).replace('R4B_DIR', 'R1_DIR')
GATE_CELL = "import pickle\nfrom sklearn.metrics import cohen_kappa_score\n" + cj("R0P = pickle.load")
API_SETUP, API_PROBE, API_BATCH = cj("import anthropic"), cj("A = anthropic.Anthropic"), cj("CHUNK = 400")
assert "def train_variant" in TRAIN and "def run(variant, seed)" in TRAIN

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R4b — 학습 데이터 정답을 고쳐서 다시 학습하면 재현율이 오르는가

## 왜 하나

R4에서 dev-L·cal-L 정답을 LLM으로 재검증한 결과, 원래 '멀쩡한' 응답의 40%가 표시되지 않은 환각을 포함했고 엄격 재현율 상한(약 0.50)의 상당 부분이 정답 누락이었다(B 문장 끝 0.501 → 0.621, 위약을 뺀 순이득 +0.063).
그런데 보정 정답에서는 **문장 끝(0.621)이 4토큰 앞부분 판정(0.48~0.54)보다 크게 높았고**, 판정기들의 완성 문장 AUROC는 오히려 떨어졌다(B 0.857 → 0.810).
판정기들이 **같은 정답 누락이 있는 학습 데이터**(gold-large 3,000응답)로 학습돼, 누락된 종류의 환각(근거 없는 꾸밈말, 리뷰 내용 왜곡, 미묘한 사실 오류)을 "멀쩡함"으로 배웠기 때문으로 본다.

## 무엇을 하나

1. **학습 데이터 재판정**: `claude-sonnet-5-5`(배치, 약 $10)가 gold-large 3,000응답의 모든 문장을 R4와 같은 지시문으로 재판정한다. 공개 LLM 2개(Qwen 14B·72B)는 Opus와의 일치도 관문을 넘지 못해 쓰지 않는다.
   관문 G는 R4에서 받은 Sonnet 판정의 cal-L 부분으로 계산한다(비용 없음). Sonnet은 평가 정답(Opus)과 같은 계열이라 R4b-2(원래 정답 비열등)를 함께 본다.
2. **보정 학습 정답**: 원래 정답 ∪ 재판정 H (R4 C1과 같은 규칙, 원래 정답은 지우지 않음). 추가 문장의 시작 토큰은 재판정 구절 위치.
3. **재학습**: R1b **V0b와 완전히 같은 방식**(student_vark에서 시작, 문장 단위 정답, 모든 판정 지점, 지점별 이진 분류, 1에폭, 시드 42·7·2026)으로 학습 정답만 바꾼다.
   체크포인트 선택은 cal-sub 300응답의 **보정 정답(R4 C1)** 으로 한다(학습 정답과 같은 기준).
4. **평가**: dev-L에서 **원래 정답과 보정 정답(C1) 둘 다**로 잰다. 임계값은 각 기준의 cal-L로 보정한다. 비교 대상은 R1b V0b(같은 방식, 원래 학습 정답)의 저장된 점수.
   비교하는 두 모델을 같은 평가 정답으로 재므로 R4의 위약 문제(정답 추가에 따른 기계적 변화)는 생기지 않는다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **R4b-0 재판정 점검**: R4b-A 관문 통과(Opus와 κ ≥ 0.50, 원래 환각 문장 H·A ≥ 0.70), 학습 응답 판정 확보 ≥ 98%, 학습 데이터의 원래 환각 문장 H·A ≥ 0.70. 미달이면 이후 수치는 해석 보류.
- **판정 방식**: 4토큰으로 고정(R1c 선택과 같음). 문장 끝은 보고.
- **R4b-1 보정 정답(주)**: dev-L C1에서 4토큰 **새 모델 시드 평균 − R1b V0b 시드 평균 ≥ +0.03**, 그리고 같은 시드끼리 쌍체 부트스트랩 하한 > 0이 3개 중 2개 이상.
- **R4b-2 원래 정답(비열등)**: dev-L 원래 정답에서 4토큰 새 모델 시드 평균 − R1b V0b 시드 평균 ≥ −0.02. (보정 정답에만 맞춘 것이 아님을 확인)
- **보고(판정 없음)**: 문장 끝 판정의 같은 비교, C1에서 B 문장 끝(0.621)과의 차이, 지연, 발생 전 경보, 서비스 10% 재현율, 응답·완성 문장 AUROC(두 기준), 과제별, 학습 데이터에서 바뀐 비율.
- 해석 규칙: R4b-1·R4b-2 모두 충족이면 "정답 정제가 앞부분 판정기의 성능을 올린다 → 게이트웨이 판정기 교체 후보". R4b-1 미달이면 "학습 정답 누락은 병목이 아님 → 문장 끝까지 봐야 하는 환각은 앞부분 판정으로 잡기 어려움(과제 정의 쪽)".

> **런타임: A100 권장**(L4 가능). 학습 3회 × 약 15~25분.
> **API 키**: 코랩 보안 비밀 `ANTHROPIC_API_KEY`. **크레딧 $15 이상** 필요(예상 약 $10, 3절에서 다시 계산).
> 드라이브에 `student_vark`, `r0/`, `r1b/`(V0b·V1b scores.npy), `r4/r4_corrected_labels.json`이 있어야 한다.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터 (R1b와 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)

md(r"""## 3. 관문 G — R4에서 받은 Sonnet·Opus 판정의 cal-L 부분으로 계산 (비용 없음)""")
code(GATE_CELL)
md(r"""## 3-1. 학습 데이터 재판정 요청 — `claude-sonnet-5-5` 배치 (R4와 같은 지시문)""")
code(API_SETUP)
md(r"""### 3-1-1. 3건만 바로 보내 형식 확인 (비용 수 센트)""")
code(API_PROBE)
md(r"""### 3-2. 배치 제출(400건씩) → 결과 받기

배치 번호·결과는 드라이브 `r4b/`에 저장되고, 다시 실행하면 이미 제출한 묶음·받은 결과는 건너뛴다. 배치를 기다리는 동안 GPU는 놀고 있으므로, 길어지면 런타임을 끊었다가 나중에 1절부터 다시 실행해도 된다(재제출하지 않음).""")
code(API_BATCH)
code(r"""
GATE_META = json.load(open(os.path.join(R1_DIR, 'judge_gate.json')))
print(f"재판정자 {GATE_META['llm']} | 관문: κ {GATE_META['kappa_H']:.3f}, H·A {GATE_META['sens_HA']:.3f} → {'통과' if GATE_META['gate'] else '미달'} | 학습 판정 {len(LABELS):,}/{len(ITEMS):,}")
""")

md(r"""### 3-3. R4b-0 점검과 보정 학습 정답""")
code(r"""
coverage = len(LABELS) / len(ITEMS)
pos_lab = [LABELS[c][i + 1]['label'] for c, x in ITEMS.items() if c in LABELS for i, sn in enumerate(x['sents']) if sn['y'] == 1]
neg_lab = [LABELS[c][i + 1]['label'] for c, x in ITEMS.items() if c in LABELS for i, sn in enumerate(x['sents']) if sn['y'] == 0]
sens_HA = float(np.mean([l in 'HA' for l in pos_lab]))
R4B_0 = bool(GATE_META['gate']) and coverage >= 0.98 and sens_HA >= 0.70
print(f"관문 {'통과' if GATE_META['gate'] else '미달'} | 확보율 {coverage:.3f} (≥ 0.98)" + f' | 원래 환각 문장 {len(pos_lab):,}개에 H·A {sens_HA:.3f} (≥ 0.70) | 원래 멀쩡한 문장 {len(neg_lab):,}개에 H {np.mean([l == "H" for l in neg_lab]):.3f}')
print(f"=== R4b-0 재판정 점검: {'충족' if R4B_0 else '미달 — 이후 수치는 해석 보류'} ===")

def content_starts(x):
    t = x['text']
    cs = [a + (len(t[a:b]) - len(t[a:b].lstrip())) for a, b in zip(x['ts'], x['te'])]
    return [c if c < b else a for c, a, b in zip(cs, x['ts'], x['te'])]

def locate(x, sn, phrase, cs):
    seg = x['text'][sn['a']:sn['b']]
    ph = (phrase or '').strip()
    k = seg.find(ph) if ph else -1
    if k < 0 and ph: k = seg.lower().find(ph.lower())
    if k < 0: return sn['t0']
    a, b = sn['a'] + k, sn['a'] + k + len(ph)
    for t in range(sn['t0'], sn['t1'] + 1):
        if min(x['te'][t], b) > max(cs[t], a): return t
    return sn['t0']

ADD_ONSET, flip, earlier = {}, 0, 0
for cid, x in ITEMS.items():
    labs = LABELS.get(cid)
    if not labs: continue
    cs, add = None, []
    for i, sn in enumerate(x['sents']):
        if sn['y'] == 0 and labs[i + 1]['label'] == 'H':
            cs = cs or content_starts(x)
            t = locate(x, sn, labs[i + 1]['phrase'], cs); ADD_ONSET[(x['id'], i)] = t; add.append(t)
    if add:
        if x['onset'] is None: flip += 1
        elif min(add) < x['onset']: earlier += 1
n_clean = sum(x['onset'] is None for x in TRAIN_ITEMS); n_hall = len(TRAIN_ITEMS) - n_clean
print(f'추가된 환각 문장 {len(ADD_ONSET):,} | 멀쩡 → 환각 {flip}/{n_clean} ({flip / max(n_clean, 1):.0%}) | 더 이른 환각 {earlier}/{n_hall} ({earlier / max(n_hall, 1):.0%})')
""")

md("## 4. 학습 판정 지점 — 보정 학습 정답으로 (R1b V0b와 같은 정의)")
code(LABELS)
md("## 5. 근거 검색 (R1b와 같음)과 평가용 보정 정답(R4 C1)")
code(RETR)
code(r"""
# R4 보정 정답 C1 (Opus 재판정)을 cal-L·dev-L에 적용한 사본
CORR = json.load(open(os.path.join(R4_DIR, 'r4_corrected_labels.json')))['C1']
def apply_corr(items, split):
    out = []
    for x in items:
        c = CORR[split].get(x['id'])
        y = copy.copy(x); y['sents'] = [dict(sn) for sn in x['sents']]
        if c is not None:
            y['onset'] = c['onset']
            for i in c['added']: y['sents'][i - 1]['y'] = 1
        out.append(y)
    return out
CAL_C, DEV_C = apply_corr(CAL, 'calL'), apply_corr(DEV, 'devL')
LAB_SETS = {'orig': (CAL, DEV), 'C1': (CAL_C, DEV_C)}
for k, (c, d) in LAB_SETS.items():
    print(f"{k}: cal-L 환각 {sum(x['onset'] is not None for x in c)}, dev-L 환각 {sum(x['onset'] is not None for x in d)}")
# 체크포인트 선택은 보정 정답 기준 cal-sub (같은 300응답)
_byid = {x['id']: x for x in CAL_C}
CALSUB = [_byid[x['id']] for x in CALSUB]
""")
md("## 6. 평가 함수 (R1b와 같음)")
code(SERIES)
code(EVAL_DEFS)
code(AUCS)
md("## 7. 학습 — R1b V0b와 같은 방식, 학습 정답만 보정 (끝난 것은 건너뜀)")
code(TRAIN)
code(r"""
SEEDS = [42, 7, 2026]
for seed in SEEDS:
    run('V0b', seed)                      # 드라이브 r4b/V0b_s*  (SC 키 'V0b_s*' = 새 모델)
for v in ['V0b', 'V1b']:
    for s in SEEDS:
        p = os.path.join(R1B_DIR, f'{v}_s{s}', 'scores.npy')
        if os.path.exists(p): SC[f'R1b{v}_s{s}'] = np.load(p)
assert all(f'R1bV0b_s{s}' in SC for s in SEEDS), 'R1b V0b 점수가 드라이브 r1b/에 있어야 한다'
print('준비된 점수:', [k for k in SC if k[-2:] in ('42', '_7') or k.endswith('2026')])
""")

md("## 8. dev-L 평가 (원래 정답·보정 정답)와 사전 기준 판정")
code(r"""
KINDS = ['end', '4', '2', 'ew2']
MODELS = ['B', 'E2'] + [f'R1b{v}_s{s}' for v in ['V0b', 'V1b'] for s in SEEDS] + [f'V0b_s{s}' for s in SEEDS]
rows = []
for lab, (cal_, dev_) in LAB_SETS.items():
    for m in MODELS:
        ra, sa = resp_auc(dev_, m), sent_auc(dev_, m)
        for k in KINDS:
            sc = f'{m}|{k}'
            r = evaluate(dev_, sc, calibrate(cal_, sc, 100)); r10 = evaluate(dev_, sc, calibrate_flag(cal_, sc, 0.10))
            rows.append({'labels': lab, 'model': m, 'kind': k, 'recall': r['recall'], 'delay_med': r['delay_tok_median'],
                         'pre_onset': r['pre_onset_alarm_rate'], 'svc_recall': r10['recall'], 'svc_delay': r10['delay_tok_median'],
                         'resp_auc': ra, 'sent_auc': sa})
res = pd.DataFrame(rows)

def fam(fm, k, lab, col='recall'):
    ms = [f'{fm}_s{s}' for s in SEEDS] if fm not in ('B', 'E2') else [fm]
    return float(np.mean([res[(res.labels == lab) & (res.model == m) & (res.kind == k)][col].iloc[0] for m in ms]))

agg = pd.DataFrame([{'labels': lab, 'family': f, 'kind': k, **{c: fam(f, k, lab, c) for c in
                     ['recall', 'delay_med', 'pre_onset', 'svc_recall', 'resp_auc', 'sent_auc']}}
                    for lab in LAB_SETS for f in ['B', 'E2', 'R1bV0b', 'R1bV1b', 'V0b'] for k in KINDS])
pd.set_option('display.width', 250)
print(agg[agg.kind.isin(['4', 'end'])].round(3).to_string(index=False))

def per_resp(items, sc, thr):
    d = []
    for x in items:
        if x['onset'] is None: continue
        first = next((t for t, s in series(x, sc) if s >= thr), None)
        d.append(float(first is not None and first >= x['onset']))
    return np.array(d)

def boot_pair(a, b, lab, n=1000, seed=0):
    cal_, dev_ = LAB_SETS[lab]
    da, db = per_resp(dev_, a, calibrate(cal_, a, 100)), per_resp(dev_, b, calibrate(cal_, b, 100))
    rng = np.random.default_rng(seed); N = len(da)
    rd = [da[i].mean() - db[i].mean() for i in (rng.integers(0, N, N) for _ in range(n))]
    lo, hi = np.percentile(rd, [2.5, 97.5])
    return {'A': a, 'B': b, 'labels': lab, 'diff': da.mean() - db.mean(), 'lo': lo, 'hi': hi}

K = '4'
g_c = fam('V0b', K, 'C1') - fam('R1bV0b', K, 'C1')
bts = [boot_pair(f'V0b_s{s}|{K}', f'R1bV0b_s{s}|{K}', 'C1') for s in SEEDS]
n_sig = sum(b['lo'] > 0 for b in bts)
R4B_1 = g_c >= 0.03 and n_sig >= 2
g_o = fam('V0b', K, 'orig') - fam('R1bV0b', K, 'orig')
R4B_2 = g_o >= -0.02
print(f"\n=== R4b-1 보정 정답(4토큰): 새 {fam('V0b', K, 'C1'):.3f} − R1b V0b {fam('R1bV0b', K, 'C1'):.3f} = {g_c:+.3f} (≥ +0.03), 유의 시드 {n_sig}/3 (≥ 2) → {'충족' if R4B_1 else '미달'} ===")
for b in bts:
    print(f"   {b['A']} vs {b['B']}: {b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
print(f"=== R4b-2 원래 정답(4토큰): 새 {fam('V0b', K, 'orig'):.3f} − R1b V0b {fam('R1bV0b', K, 'orig'):.3f} = {g_o:+.3f} (≥ −0.02) → {'충족' if R4B_2 else '미달'} ===")
if not R4B_0:
    print('※ R4b-0 미달: 위 판정은 해석 보류')
print(f"\n(보고) 문장 끝: C1 새 {fam('V0b', 'end', 'C1'):.3f} vs R1b V0b {fam('R1bV0b', 'end', 'C1'):.3f} | 원래 새 {fam('V0b', 'end', 'orig'):.3f} vs {fam('R1bV0b', 'end', 'orig'):.3f}")
print(f"(보고) C1에서 B 문장 끝 {fam('B', 'end', 'C1'):.3f} 대비 새 4토큰 {fam('V0b', K, 'C1'):.3f} (차이 {fam('V0b', K, 'C1') - fam('B', 'end', 'C1'):+.3f}), 지연 {fam('V0b', K, 'C1', 'delay_med'):.1f} vs {fam('B', 'end', 'C1', 'delay_med'):.1f}")
print(f"(보고) 완성 문장 AUROC C1: 새 {fam('V0b', K, 'C1', 'sent_auc'):.3f} vs R1b V0b {fam('R1bV0b', K, 'C1', 'sent_auc'):.3f} | 원래: {fam('V0b', K, 'orig', 'sent_auc'):.3f} vs {fam('R1bV0b', K, 'orig', 'sent_auc'):.3f}")
print(f"(보고) 서비스 10% 재현율 C1: 새 {fam('V0b', K, 'C1', 'svc_recall'):.3f} vs R1b V0b {fam('R1bV0b', K, 'C1', 'svc_recall'):.3f}")
""")

md("## 9. 과제별 (보고)")
code(r"""
tr = []
for lab, (cal_, dev_) in LAB_SETS.items():
    for tt in TASKS:
        its = [x for x in dev_ if x['task_type'] == tt]
        row = {'labels': lab, 'task': tt, 'n_hall': sum(x['onset'] is not None for x in its)}
        for sc in ['B|end'] + [f'R1bV0b_s{s}|4' for s in SEEDS] + [f'V0b_s{s}|4' for s in SEEDS]:
            row[sc] = evaluate(its, sc, calibrate(cal_, sc, 100))['recall']
        row['R1bV0b|4 평균'] = np.mean([row[f'R1bV0b_s{s}|4'] for s in SEEDS]); row['새 V0b|4 평균'] = np.mean([row[f'V0b_s{s}|4'] for s in SEEDS])
        tr.append(row)
task_df = pd.DataFrame(tr)
print(task_df[['labels', 'task', 'n_hall', 'B|end', 'R1bV0b|4 평균', '새 V0b|4 평균']].round(3).to_string(index=False))
""")

md("## 10. 저장")
code(r"""
import shutil
summ = {'R4b_0': bool(R4B_0), 'coverage': coverage, 'sens_HA_train': sens_HA, 'added_sent_train': len(ADD_ONSET), 'flip_clean_train': flip,
        'earlier_train': earlier, 'R4b_1': bool(R4B_1), 'gain_C1': g_c, 'n_sig_C1': int(n_sig), 'boot_C1': bts,
        'R4b_2': bool(R4B_2), 'gain_orig': g_o, 'judge': JUDGE}
json.dump(summ, open('r4b_summary.json', 'w'), default=float, ensure_ascii=False, indent=1)
hists = []
for s in SEEDS:
    p = os.path.join(R1_DIR, f'V0b_s{s}', 'hist.csv')
    if os.path.exists(p):
        h = pd.read_csv(p); h['run'] = f'V0b_s{s}'; hists.append(h)
out = {'r4b_devL.csv': res, 'r4b_family.csv': agg, 'r4b_task.csv': task_df, 'r4b_hist.csv': pd.concat(hists) if hists else pd.DataFrame()}
for f, d in out.items():
    d.to_csv(f, index=False); d.to_csv(os.path.join(R1_DIR, f), index=False)
shutil.copy('r4b_summary.json', os.path.join(R1_DIR, 'r4b_summary.json'))
print('저장 완료: 드라이브 selfproject/r4b/')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r4b_relabel_retrain.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
