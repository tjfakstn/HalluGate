"""I1 노트북 생성: 판정기 입력 개선 (근거 예산 채우기 + 앞 문장).
F1 base와 학습 정답·백본·학습 루프·시드는 같고, 판정기 입력만 바꾼다.
  - 근거: top-8 고정 → 유사도 순으로 512토큰 예산을 채움(잘림은 위치가 아니라 유사도 낮은 조각부터)
  - 판정 텍스트: 문장 앞부분 → 앞 문장(끝 48단어) [SEP] 문장 앞부분
데이터·평가·학습 코드는 R4c 노트북(=F1)에서 가져온다. 조사 근거: ORION(arXiv 2504.15771)의 예산 채우기, 앞 문장 미사용 약점(SSOT §5)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
def cells_of(name):
    nb = json.load(open(os.path.join(NB_DIR, name)))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
R4C = cells_of('r4c_clean_labels_base.ipynb')
def find(C, prefix):
    return next(c for c in C if c.startswith(prefix))
def sub(s, a, b):
    assert a in s, a[:70]
    return s.replace(a, b, 1)

ENV = find(R4C, "!pip -q install pysbd")
ENV = sub(ENV, "/content/drive/MyDrive/selfproject", "/content/drive/MyDrive/Colab Notebooks/HalluGate")
ENV = sub(ENV, "R1_DIR = os.path.join(OUT_DIR, 'r4c')",
          "R1_DIR = I1_DIR = os.path.join(OUT_DIR, 'i1')\nFINAL_DIR = os.path.join(OUT_DIR, 'final')")
ENV = sub(ENV, "print('device', DEVICE)",
          "assert os.path.exists(os.path.join(FINAL_DIR, 'scores_base.npy')), 'final/scores_base.npy 없음 (F1 재실행 점수, 비교 기준)'\n"
          "import shutil\n"
          "_t = os.path.join(I1_DIR, '_write_test.bin')\n"
          "try:\n"
          "    with open(_t, 'wb') as f: f.write(os.urandom(50 * 2**20))\n"
          "    ok = os.path.getsize(_t) == 50 * 2**20; os.remove(_t)\n"
          "except OSError as e:\n"
          "    ok = False; print('드라이브 쓰기 실패:', e)\n"
          "print('드라이브 50MB 쓰기', '정상' if ok else '실패 — 휴지통 비우기·용량 확인. 가중치는 마지막 셀 zip으로 받는다')\n"
          "print('device', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')")
DATA = find(R4C, "BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find(R4C, "# 학습 응답: S2/S2-b의 gold-large")
PREP = find(R4C, "ENC = tiktoken.get_encoding")
LOADLAB = find(R4C, "TRAIN_ITEMS = to_items(train)")
ADD = find(R4C, "def content_starts(x):")
LABELS = find(R4C, "def points(item, s):")
RETR = find(R4C, "from sentence_transformers import SentenceTransformer")
# top-8 검색 결과는 I1에서 쓰지 않는다(예산 채우기로 대체). 시간 절약을 위해 계산하지 않는다.
RETR = sub(RETR, "P_IDX = topk_idx(P_KEY, P_TXT)", "P_IDX = None   # I1: 아래 예산 채우기로 대체")
RETR = sub(RETR, "EQ_IDX = dict(zip(EVAL_Q, topk_idx([Q_KEYS[q] for q in EVAL_Q], [Q_TEXTS[q] for q in EVAL_Q])))",
           "EQ_IDX = None  # I1: 아래 예산 채우기로 대체")
CORR = find(R4C, "# R4 보정 정답 C1")
SERIES = find(R4C, "SC = dict(Z0)")
EVAL_DEFS = find(R4C, "def calibrate(items, scorer, arl0):")
TRAIN_BASE = find(R4C, "KEEP_WEIGHTS = {42}")
assert 'def train_backbone' in TRAIN_BASE and 'def calsub_metric' in TRAIN_BASE and 'def enc_ids' in TRAIN_BASE

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

# ---- 아래 I1 전용 셀(I1_INPUT, I1_TRAIN_OVERRIDE, DECIDE)은 smoke_i1.py가 가짜 데이터로 그대로 실행해 확인한다 ----
I1_INPUT = r"""
# I1 입력 구성: 앞 문장 + 근거 예산 채우기
I1_TOK = AutoTokenizer.from_pretrained(BACKBONES_PATH)
PREV_WORDS = 48          # 앞 문장은 끝 48단어까지 (긴 문장이 근거 예산을 다 먹지 않게)
SEP_COST = len(I1_TOK(' ... ', add_special_tokens=False)['input_ids'])   # 조각 사이 ' ... ' 토큰 수 (DeBERTa-v3: 3)

def text_b(prev, txt):
    if not prev:
        return txt
    return ' '.join(prev.split()[-PREV_WORDS:]) + f' {I1_TOK.sep_token} ' + txt

chunk_len_cache = {}
def chunk_lens(key):
    if key not in chunk_len_cache:
        ch, _ = get_chunks(key)
        chunk_len_cache[key] = [len(v) for v in I1_TOK(ch, add_special_tokens=False)['input_ids']]
    return chunk_len_cache[key]

def fill_select(key, e, b_len):
    # 유사도 높은 조각부터 예산(MAX_LEN − 판정 텍스트 − 특수 토큰 3개)이 찰 때까지 담고, 원문 순서로 정렬
    ch, ce = get_chunks(key)
    L = chunk_lens(key)
    budget = MAX_LEN - b_len - 3
    sel, used = [], 0
    for j in torch.argsort(ce @ e, descending=True).tolist():
        c = L[j] + (SEP_COST if sel else 0)
        if used + c <= budget:
            sel.append(j); used += c
    if not sel:                                   # 판정 텍스트가 너무 길면 가장 비슷한 조각 하나 (잘림 허용)
        sel = [int(torch.argmax(ce @ e))]
    return sorted(sel)

@torch.no_grad()
def build_inputs(keys, prevs, texts, bs=512):
    # 검색 질의는 F1과 같이 '문장 앞부분'만 쓴다 (앞 문장은 판정 입력에만)
    B = [text_b(p, t) for p, t in zip(prevs, texts)]
    blen = [len(v) for v in I1_TOK(B, add_special_tokens=False)['input_ids']]
    sels = []
    for i in range(0, len(texts), bs):
        embs = retriever.encode(list(texts[i:i + bs]), convert_to_tensor=True, normalize_embeddings=True,
                                batch_size=bs, show_progress_bar=False)
        for key, e, bl in zip(keys[i:i + bs], embs, blen[i:i + bs]):
            sels.append(fill_select(key, e, bl))
    return B, sels

def prev_of_train(items):
    # LABELS 셀과 같은 순서로 학습 지점마다 앞 문장을 만든다
    out = []
    for x in items:
        sidx = {sn['t0']: i for i, sn in enumerate(x['sents'])}
        for t, txt in points(x, 2):
            if not txt: continue
            si = sidx[x['sent_of'][t]]
            out.append(x['text'][x['sents'][si - 1]['a']:x['sents'][si - 1]['b']].strip() if si > 0 else '')
    return out

def build_eval_queries(items):
    # R0 질의 번호는 (근거, 텍스트)로 묶여 앞 문장이 다른 지점을 구분하지 못한다 → (근거, 앞 문장, 텍스트)로 새 번호
    index, keys, prevs, texts, MAP = {}, [], [], [], {}
    for x in items:
        d = dict(x['pts2'])
        sidx = {sn['t0']: i for i, sn in enumerate(x['sents'])}
        for t, q in x['pts2']:
            si = sidx[x['sent_of'][t]]
            pe = x['sents'][si - 1]['t1'] if si > 0 else None
            prev = Q_TEXTS[d[pe]] if pe is not None and pe in d else ''
            k = (Q_KEYS[q], prev, Q_TEXTS[q])
            if k not in index:
                index[k] = len(keys); keys.append(k[0]); prevs.append(prev); texts.append(k[2])
            MAP[(x['id'], t)] = index[k]
    return keys, prevs, texts, MAP

def remap(items, MAP):
    out = []
    for x in items:
        y = copy.copy(x); y['pts2'] = [(t, MAP[(x['id'], t)]) for t, _ in x['pts2']]
        out.append(y)
    return out
"""

I1_TRAIN_OVERRIDE = r"""
# F1 학습 루프(train_backbone)는 그대로 쓰고, 입력·체크포인트 평가만 I1 입력으로 바꾼다
def enc_ids(tok, idxs):
    return tok([ctx_of(P_KEY[g], P_SEL[g]) for g in idxs], [P_B[g] for g in idxs], truncation='only_first',
               max_length=MAX_LEN, padding=True, return_tensors='pt').to(DEVICE)

@torch.no_grad()
def score_q2(tok, model, qs, bs=128):
    model.eval(); out = np.zeros(len(qs), dtype=np.float32)
    order = sorted(range(len(qs)), key=lambda i: len(Q2_B[qs[i]]))
    for i in range(0, len(order), bs):
        b = [qs[j] for j in order[i:i + bs]]
        enc = tok([ctx_of(Q2_KEYS[q], Q2_SEL[q]) for q in b], [Q2_B[q] for q in b], truncation='only_first',
                  max_length=MAX_LEN, padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
            out[order[i:i + bs]] = model(**enc).logits[:, 0].float().cpu().numpy()
    model.train()
    return out

CALSUB2 = remap(CALSUB, Q2_MAP)                       # 보정 정답(C1) cal-sub 300응답, F1과 같은 응답
CALSUB2_Q = sorted({q for x in CALSUB2 for _, q in x['pts2']})
def calsub_metric(tok, model):
    v = score_q2(tok, model, CALSUB2_Q)
    SC['_cur'] = np.zeros(len(Q2_KEYS), dtype=np.float32); SC['_cur'][CALSUB2_Q] = v
    r4 = evaluate(CALSUB2, '_cur|4', calibrate(CALSUB2, '_cur|4', 100))
    r2 = evaluate(CALSUB2, '_cur|2', calibrate(CALSUB2, '_cur|2', 100))
    d4 = r4['delay_tok_median']
    return float((r4['recall'] + r2['recall']) / 2), float(d4) if d4 == d4 else 1e9
"""

DECIDE = r"""
ALPHA = 0.10
def conformal_flag(items, scorer, alpha=ALPHA):
    mx = np.sort([max([s for _, s in series(x, scorer)] or [-1e9]) for x in items if x['onset'] is None])
    n = len(mx); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(mx[k - 1]) + 1e-6 if k <= n else float('inf')

def detect_vec(items, scorer, thr):
    # 환각 응답마다 엄격 탐지 여부(첫 경보가 발생 이후)와 지연
    det, dl = [], []
    for x in items:
        if x['onset'] is None: continue
        first = next((t for t, s in series(x, scorer) if s >= thr), None)
        ok = first is not None and first >= x['onset']
        det.append(ok); dl.append(first - x['onset'] if ok else np.nan)
    return np.array(det, dtype=float), np.array(dl)

def compare(sets_b, sets_i, kind, n_boot=2000, seed=2027):
    # sets_*: {'orig': (cal, dev), 'C1': (cal, dev)} — 기준선은 R0 질의 번호, I1은 새 질의 번호로 매긴 같은 응답
    rows = {}
    for lab in ['orig', 'C1']:
        cb, ci = sets_b[lab][0], sets_i[lab][0]
        tb, ti = conformal_flag(cb, f'F_base|{kind}'), conformal_flag(ci, f'I1|{kind}')
        db, lb = detect_vec(cb, f'F_base|{kind}', tb); di, li = detect_vec(ci, f'I1|{kind}', ti)
        rng = np.random.default_rng(seed); n = len(db)
        boot = [(di[ix].mean() - db[ix].mean()) for ix in (rng.integers(0, n, n) for _ in range(n_boot))]
        rows[lab] = {'thr_base': tb, 'thr_i1': ti, 'recall_base': db.mean(), 'recall_i1': di.mean(),
                     'delta': di.mean() - db.mean(), 'ci_lo': float(np.percentile(boot, 2.5)), 'ci_hi': float(np.percentile(boot, 97.5)),
                     'delay_base': float(np.nanmedian(lb)), 'delay_i1': float(np.nanmedian(li)), 'n_hall': n}
    return rows
"""

md(r"""# I1 — 판정기 입력 개선: 근거 예산 채우기 + 앞 문장

## 왜 하나

F1 base(배포 판정기)는 근거를 **40단어 조각 중 비슷한 8개**만 보고, 판정 텍스트에 **앞 문장이 없다**(SSOT §5 F1 한계).
8개 × 40단어 ≈ 420토큰에 판정 텍스트를 더하면 512토큰에 가까워, 넘치면 **유사도와 무관하게 원문 뒤쪽 조각이 잘린다**.
조사(11/04)에서 RAGTruth로 학습하지 않고도 높은 성능을 낸 ORION(arXiv 2504.15771)은 "입력 창을 근거로 꽉 채우는 것"을 핵심으로 꼽았다.
재학습 1회·지연 변화 없이(입력 길이 512 그대로) 서비스 탐지율을 올릴 수 있는지 확인한다. 10/28 결정의 "서비스 성능 향상" 후보 (2)에 해당한다.

## 무엇을 하나 (F1 base와 입력만 다름)

| | F1 base | I1 |
|---|---|---|
| 근거 | 비슷한 조각 top-8, 원문 순서, 넘치면 뒤쪽 잘림 | 비슷한 조각부터 **512토큰 예산이 찰 때까지**, 원문 순서 (잘리지 않음) |
| 판정 텍스트 | 문장 앞부분 | **앞 문장(끝 48단어) [SEP] 문장 앞부분** (응답 첫 문장은 앞부분만) |
| 검색 질의 | 문장 앞부분 | 같음 |
| 백본·학습 정답·학습 루프·시드·체크포인트 선택 | DeBERTa-v3-base, R4b 보정 정답, R1c 루프, 42, cal-sub | 같음 |

## 사전 정의 기준 (결과 보기 전, SSOT §6 I1에 고정)

비교는 **cal-L에서만**(F1 base 점수 = `final/scores_base.npy`, 재실행 배포 가중치). 임계값은 각 모델의 cal-L 등각 보정 서비스 10%(멀쩡한 응답 표시 ≤ 10%).
- **I1-0 입력 점검**: I1 입력 중 512토큰을 넘어 잘리는 비율 ≤ 1% (예산 채우기 구현 확인. 미달이면 버그로 보고 결과 해석 보류)
- **I1-1 (주)**: 8토큰 재판정, 원래 정답에서 탐지율 차이(I1 − F1 base) **≥ +0.03, 그리고** 응답 단위 쌍체 부트스트랩 95% 하한 > 0
- **I1-2 (안전)**: 보정 정답(C1)에서 탐지율 차이 ≥ −0.01, 그리고 원래 정답 탐지 지연 중앙값 증가 ≤ 2토큰
- **I1-3 (지연)**: GPU 판정 1회(검색·예산 채우기 포함) p95 ≤ 68ms (8토큰 × 8.5ms)
- **채택** = I1-1·I1-2·I1-3 모두 충족. 채택하면 게이트웨이 판정기 입력을 바꾼다(SPEC → 테스트 → 코드, S4). dev-L은 **채택 판정 후 보고만** 한다.
- 단서: 시드 1개. F1 재실행에서 같은 시드 재학습 간 변동이 약 0.015~0.018이어서 주 기준을 그 2배인 +0.03으로 둔다. cal-sub(cal-L 300응답)는 두 모델 모두 체크포인트 선택에 썼다.

> **런타임: A100.** 학습 약 90분 + 입력 구성·채점 약 20분. 드라이브에 F1과 같은 입력(`r0/`, `r4/`, `r4b/`)과 `final/scores_base.npy`가 있어야 한다.
> 결과는 드라이브 `Colab Notebooks/HalluGate/i1/`과 `i1_judge.zip`(자동 다운로드).
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터 (F1과 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
md("## 3. 보정 학습 정답 (F1과 같음)")
code(LOADLAB)
code(ADD)
md("## 4. 학습 판정 지점 (F1과 같음)")
code(LABELS)
md("## 5. 근거 조각·평가 세트 (F1과 같음, top-8 검색은 생략)")
code(RETR)
code(CORR)
md("## 6. 평가 함수 (F1과 같음)")
code(SERIES)
code(EVAL_DEFS)
md("## 7. 학습 루프 (F1과 같음)")
code(TRAIN_BASE)

md("""## 8. I1 입력 구성

학습 지점과 평가 질의마다 (앞 문장 [SEP] 문장 앞부분)과 예산 채우기 근거 조각을 만든다.
평가 질의는 앞 문장까지 포함한 새 번호를 쓰므로, cal-L·dev-L 응답의 판정 지점을 새 번호로 다시 매긴 사본을 만든다(지점·정답은 그대로).""")
code("BACKBONES_PATH = BACKBONES['base'][0]\n" + I1_INPUT + r"""
t0 = time.perf_counter()
P_PREV = prev_of_train(train_items)
assert len(P_PREV) == len(P_TXT), (len(P_PREV), len(P_TXT))
P_B, P_SEL = build_inputs(P_KEY, P_PREV, P_TXT)
print(f'학습 지점 {len(P_TXT):,}개 입력 구성 {time.perf_counter() - t0:.0f}초 | 앞 문장 있는 지점 {np.mean([bool(p) for p in P_PREV]):.1%}')

Q2_KEYS, Q2_PREV, Q2_TEXTS, Q2_MAP = build_eval_queries(CAL + DEV)
t0 = time.perf_counter()
Q2_B, Q2_SEL = build_inputs(Q2_KEYS, Q2_PREV, Q2_TEXTS)
print(f'평가 질의 {len(Q2_KEYS):,}개 (R0 번호 {len(EVAL_Q):,}개) 입력 구성 {time.perf_counter() - t0:.0f}초')
LAB_SETS_I1 = {lab: (remap(c, Q2_MAP), remap(d, Q2_MAP)) for lab, (c, d) in LAB_SETS.items()}

# I1-0 입력 점검: 잘림 비율 (I1) vs top-8 방식(F1)의 잘림 비율, 평가 질의 5,000개 표본
rs = random.Random(5).sample(range(len(Q2_KEYS)), min(5000, len(Q2_KEYS)))
n_i1 = [len(I1_TOK(ctx_of(Q2_KEYS[q], Q2_SEL[q]), Q2_B[q])['input_ids']) for q in rs]
emb = retriever.encode([Q2_TEXTS[q] for q in rs], convert_to_tensor=True, normalize_embeddings=True, batch_size=512, show_progress_bar=False)
top8 = [sorted(torch.topk(get_chunks(Q2_KEYS[q])[1] @ e, min(K, get_chunks(Q2_KEYS[q])[1].shape[0])).indices.tolist()) for q, e in zip(rs, emb)]
n_f1 = [len(I1_TOK(ctx_of(Q2_KEYS[q], ix), Q2_TEXTS[q])['input_ids']) for q, ix in zip(rs, top8)]
n_ch_i1 = [len(Q2_SEL[q]) for q in rs]
TRUNC_I1, TRUNC_F1 = float(np.mean(np.array(n_i1) > MAX_LEN)), float(np.mean(np.array(n_f1) > MAX_LEN))
I1_0 = TRUNC_I1 <= 0.01
print(f'잘림 비율: F1 방식 {TRUNC_F1:.1%} → I1 {TRUNC_I1:.1%} | I1 근거 조각 수 중앙값 {np.median(n_ch_i1):.0f} (F1 8개) | 입력 길이 중앙값 F1 {np.median(n_f1):.0f} / I1 {np.median(n_i1):.0f}')
print(f"=== I1-0 입력 점검: {'충족' if I1_0 else '미달 → 예산 채우기 버그 확인, 결과 해석 보류'} ===")
""")
code(I1_TRAIN_OVERRIDE)

md("""## 9. 학습·저장 (시드 42, F1 base와 같은 루프)

로컬에 fp16으로 먼저 저장하고 드라이브에 복사·크기 검증. 점수는 저장한 파일을 다시 불러 매긴다.""")
code(r"""
LOCAL = '/content/i1_local'
D = os.path.join(LOCAL, 'judge_i1')
os.makedirs(LOCAL, exist_ok=True)
WFILES = ['model.safetensors', 'pytorch_model.bin']
def has_weights(d):
    return os.path.isdir(d) and any(os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0 for f in WFILES)
def sizes(d):
    return {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))}
def to_drive():
    dst = os.path.join(I1_DIR, 'judge_i1')
    try:
        shutil.copytree(D, dst, dirs_exist_ok=True)
        ok = sizes(D) == {f: v for f, v in sizes(dst).items() if f in sizes(D)}
    except OSError as e:
        print('드라이브 복사 실패:', e); ok = False
    print(f'드라이브 저장 {"확인" if ok else "실패 — 마지막 셀 zip을 꼭 내려받을 것"}')
    return ok

DD = os.path.join(I1_DIR, 'judge_i1')
if not has_weights(D) and has_weights(DD):
    shutil.copytree(DD, D, dirs_exist_ok=True); print('드라이브 저장본 사용')
if not has_weights(D):
    t0 = time.perf_counter()
    tok, model, hist = train_backbone('base', 42)
    os.makedirs(D, exist_ok=True); tok.save_pretrained(D); model.half().save_pretrained(D)
    pd.DataFrame(hist, columns=['step', 'calsub_recall', 'calsub_delay4']).to_csv(os.path.join(D, 'hist.csv'), index=False)
    print(f'학습 {(time.perf_counter() - t0) / 60:.0f}분')
    del model; gc.collect(); torch.cuda.empty_cache()
    DRIVE_OK = to_drive()
else:
    DRIVE_OK = has_weights(DD)

sp, spd = os.path.join(LOCAL, 'scores_i1.npy'), os.path.join(I1_DIR, 'scores_i1.npy')
if not os.path.exists(sp) and os.path.exists(spd): shutil.copy(spd, sp)
if not os.path.exists(sp):
    tok = AutoTokenizer.from_pretrained(D)
    model = AutoModelForSequenceClassification.from_pretrained(D).float().to(DEVICE)
    t0 = time.perf_counter()
    np.save(sp, score_q2(tok, model, list(range(len(Q2_KEYS)))))
    try: shutil.copy(sp, spd)
    except OSError as e: print('점수 드라이브 저장 실패', e)
    print(f'평가 질의 채점 {time.perf_counter() - t0:.0f}초')
    del model; gc.collect(); torch.cuda.empty_cache()
SC['I1'] = np.load(sp)
SC['F_base'] = np.load(os.path.join(FINAL_DIR, 'scores_base.npy'))
""")

md("""## 10. 채택 판정 (cal-L만)

먼저 F1 base 점수로 F1 재실행의 8토큰 서비스 임계값(3.969)이 재현되는지 확인한다(데이터·점수 불러오기 점검).""")
code(DECIDE + r"""
_chk = conformal_flag(CAL, 'F_base|8')
assert abs(_chk - 3.968751) < 1e-3, f'F1 재실행 임계값 재현 실패: {_chk}'
print(f'F1 base 8토큰 cal-L 서비스 임계값 재현: {_chk:.4f}')

CMP = {k: compare(LAB_SETS, LAB_SETS_I1, k) for k in ['end', '4', '8', '16']}
rows = [{'kind': k, 'labels': lab, **v} for k, d in CMP.items() for lab, v in d.items()]
cmp_tab = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(cmp_tab.round(3).to_string(index=False))

# I1-3 지연: 판정 1회 = 검색(임베딩) + 예산 채우기 + 교차 인코더, 배치 1 (게이트웨이 judge()와 같은 조건)
tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForSequenceClassification.from_pretrained(D).float().to(DEVICE).eval()
@torch.no_grad()
def judge_once(key, prev, txt):
    e = retriever.encode([txt], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)[0]
    b = text_b(prev, txt); bl = len(I1_TOK(b, add_special_tokens=False)['input_ids'])
    enc = tok([ctx_of(key, fill_select(key, e, bl))], [b], truncation='only_first', max_length=MAX_LEN, return_tensors='pt').to(DEVICE)
    with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
        s = float(model(**enc).logits.squeeze().float().cpu())
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return s
LQ = random.Random(2027).sample(range(len(Q2_KEYS)), 200)
for q in LQ[:5]: judge_once(Q2_KEYS[q], Q2_PREV[q], Q2_TEXTS[q])
ms = []
for q in LQ:
    t0 = time.perf_counter(); judge_once(Q2_KEYS[q], Q2_PREV[q], Q2_TEXTS[q]); ms.append((time.perf_counter() - t0) * 1000)
P95 = float(np.percentile(ms, 95))
print(f'판정 1회 {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"}: p50 {np.percentile(ms, 50):.1f} / p95 {P95:.1f} ms')
del model; gc.collect(); torch.cuda.empty_cache()

o, c1 = CMP['8']['orig'], CMP['8']['C1']
I1_1 = o['delta'] >= 0.03 and o['ci_lo'] > 0
I1_2 = c1['delta'] >= -0.01 and (o['delay_i1'] - o['delay_base']) <= 2
I1_3 = P95 <= 68
ADOPT = bool(I1_0 and I1_1 and I1_2 and I1_3)
print(f"=== I1-1 (주) 8토큰 원래 정답: {o['recall_base']:.3f} → {o['recall_i1']:.3f} (Δ {o['delta']:+.3f} [{o['ci_lo']:+.3f}, {o['ci_hi']:+.3f}]) → {'충족' if I1_1 else '미달'} ===")
print(f"=== I1-2 (안전) C1 Δ {c1['delta']:+.3f}, 지연 {o['delay_base']:.0f} → {o['delay_i1']:.0f}토큰 → {'충족' if I1_2 else '미달'} ===")
print(f"=== I1-3 (지연) p95 {P95:.1f}ms → {'충족' if I1_3 else '미달'} ===")
print(f"=== I1 채택: {'채택 → 게이트웨이 입력 변경(S4)' if ADOPT else '미채택 → F1 base 유지'} ===")
""")

md("""## 11. dev-L 확인 (보고만, 채택 판정에 쓰지 않음)

cal-L에서 정한 각 모델의 서비스 임계값을 dev-L에 그대로 적용한다. 과제별로도 본다.""")
code(r"""
def evaluate_at(items, scorer, thr):
    r = evaluate(items, scorer, thr)
    return {'recall': r['recall'], 'delay_med': r['delay_tok_median'], 'clean_flagged': r['clean_resp_flagged']}
DEV_ROWS = []
for k in ['end', '4', '8', '16']:
    for lab in ['orig', 'C1']:
        c = CMP[k][lab]
        for model_name, sets, sc, thr in [('F_base', LAB_SETS, f'F_base|{k}', c['thr_base']), ('I1', LAB_SETS_I1, f'I1|{k}', c['thr_i1'])]:
            dev_ = sets[lab][1]
            DEV_ROWS.append({'model': model_name, 'kind': k, 'labels': lab, 'task': 'all', **evaluate_at(dev_, sc, thr)})
            if k == '8' and lab == 'orig':
                for task in sorted({x['task_type'] for x in dev_}):
                    DEV_ROWS.append({'model': model_name, 'kind': k, 'labels': lab, 'task': task,
                                     **evaluate_at([x for x in dev_ if x['task_type'] == task], sc, thr)})
dev_tab = pd.DataFrame(DEV_ROWS)
print(dev_tab.round(3).to_string(index=False))
""")

md("## 12. 저장과 zip 다운로드")
code(r"""
cfg = {'name': 'judge_i1', 'backbone': BACKBONES_PATH, 'seed': 42,
       'input': {'evidence': 'similarity-ranked 40-word chunks filled up to max_len budget, original order, joined by " ... "',
                 'text_b': f'previous sentence (last {PREV_WORDS} words) [SEP] sentence prefix; first sentence: prefix only',
                 'retrieval_query': 'sentence prefix'},
       'embedder': 'sentence-transformers/all-MiniLM-L6-v2', 'max_len': MAX_LEN, 'chunk_words': CHUNK_WORDS,
       'stride_unit': 'tiktoken cl100k_base tokens, counted from sentence start (sentence end always judged)',
       'default': {'kind': '8', 'threshold': 'svc10_orig'}, 'threshold': CMP['8']['orig']['thr_i1'],
       'thresholds': {k: {f'svc10_{lab}': CMP[k][lab]['thr_i1'] for lab in ['orig', 'C1']} for k in CMP},
       'score': 'logit (higher = hallucinated)', 'weights_dtype': 'float16 (load as float32)',
       'i1': {'I1_0': bool(I1_0), 'I1_1': bool(I1_1), 'I1_2': bool(I1_2), 'I1_3': bool(I1_3), 'adopt': ADOPT}}
json.dump(cfg, open(os.path.join(D, 'judge_config.json'), 'w'), indent=1, ensure_ascii=False)
summ = {'I1_0': bool(I1_0), 'I1_1': bool(I1_1), 'I1_2': bool(I1_2), 'I1_3': bool(I1_3), 'adopt': ADOPT,
        'trunc': {'f1_style': TRUNC_F1, 'i1': TRUNC_I1}, 'p95_ms': P95, 'drive_weights_ok': DRIVE_OK,
        'calL_compare': CMP}
json.dump(summ, open(os.path.join(LOCAL, 'i1_summary.json'), 'w'), default=float, ensure_ascii=False, indent=1)
cmp_tab.to_csv(os.path.join(LOCAL, 'i1_calL.csv'), index=False)
dev_tab.to_csv(os.path.join(LOCAL, 'i1_devL.csv'), index=False)
if DRIVE_OK: to_drive()
for f in ['i1_summary.json', 'i1_calL.csv', 'i1_devL.csv']:
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(I1_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/i1_judge', 'zip', LOCAL)
print(f'zip {os.path.getsize(zp) / 2**20:.0f}MB:', zp)
from google.colab import files
files.download(zp)
print(json.dumps({k: summ[k] for k in ['I1_0', 'I1_1', 'I1_2', 'I1_3', 'adopt', 'trunc', 'p95_ms']}, ensure_ascii=False, indent=1, default=float))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'i1_input_fill_prev.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
