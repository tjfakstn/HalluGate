"""R1 노트북 생성: 판정기의 학습 목표를 평가 목표(발생 전에는 조용히, 발생 직후에는 울리기)에 맞춘다.
데이터·전처리·근거 검색·평가 코드는 E2 노트북(그리고 그 안의 E1 코드)에서, 평가 세트와 기준선 점수는 R0에서 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
e2 = json.load(open(os.path.join(NB_DIR, 'e2_prefix_trained_scorer.ipynb')))
C = [''.join(c['source']) for c in e2['cells']]
def find(prefix):
    return next(c for c in C if c.startswith(prefix))

ENV = find("!pip -q install pysbd").replace(
    "SAVE_DIR = os.path.join(OUT_DIR, 'student_prefix')",
    "R0_DIR = os.path.join(OUT_DIR, 'r0')\n"
    "R1_DIR = os.path.join(OUT_DIR, 'r1')\n"
    "os.makedirs(R1_DIR, exist_ok=True)")
assert 'R1_DIR' in ENV
DATA = find("BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find("# 학습 응답: S2/S2-b의 gold-large")
PREP = find("ENC = tiktoken.get_encoding")
RETR = find("from sentence_transformers import SentenceTransformer").split("\nt0 = time.perf_counter()")[0]
RETR = RETR.split("\n@torch.no_grad()")[0]
PTS = find("def points(item, s):")
POINTS_DEF = PTS.split("\nQUERY_ID")[0]
EVAL_DEFS = find("def calibrate(items, scorer, arl0):")
assert 'def points' in POINTS_DEF and 'def evaluate' in EVAL_DEFS

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R1 — 판정기 학습 목표를 평가 목표에 맞추기 (발생 시점 인식 학습)

## 왜 하나

지금까지의 결과(dev-L, 환각 응답 759개)를 모으면 병목이 한 곳으로 모인다.
- **R0**: 문장 앞부분을 지점마다 독립 이진 분류로 학습한 E2는, 기존 판정기 B에 재판정만 붙인 것보다 엄격 재현율이 낫지 않았다.
- **R2**: 경보 규칙(누적합·평활·학습 규칙 52개)을 바꿔도 재현율 약 0.50을 넘지 못했다. 규칙은 "발생 전 경보"를 "끝까지 못 잡음"으로 옮길 뿐이었다.
- **사례 검토**: 이른 경보의 절반 이상(53%)은 진짜 오경보였다.

즉 **응답 하나 안에서 환각 시작 전 지점과 시작 후 지점을 가르는 능력**이 부족하다. 그런데 지금 학습 목표는 그 능력을 직접 요구하지 않는다.
엄격 재현율은 "시작 전 모든 지점은 임계값 아래, 시작 직후 **어느 한** 지점은 임계값 위"일 때만 성공으로 친다.
- 지점별 이진 분류는 시작 후 **모든** 지점을 올리라고 요구한다. 시작 후에도 애매한 지점(아직 근거와 구별되지 않는 앞부분)을 억지로 올리면, 비슷하게 생긴 시작 전 지점도 함께 올라간다.
- 지점별 이진 분류는 시작 전 지점들을 **평균적으로** 내린다. 하지만 실패를 만드는 것은 그중 **가장 높은 하나**다.

## 무엇을 하나 — 학습 목표 4가지 (모두 student_vark에서 시작, 같은 데이터·같은 판정 지점)

| 변형 | 학습 목표 | 의도 |
|---|---|---|
| **V0 (대조군)** | 판정 지점마다 이진 분류: 시작 후 = 1, 시작 전·멀쩡한 응답 = 0 | E2와 같은 목표를 E2보다 촘촘한 지점(2토큰 간격 전부)으로. 개선이 "데이터를 더 본 것"인지 "목표를 바꾼 것"인지 가르는 기준 |
| **V1 (창 안 하나)** | 시작 전·멀쩡한 지점은 0. 시작 후 10토큰 창 안에서는 **적어도 하나가 울릴 확률**을 최대화(다중 사례 학습, 이산시간 위험률의 탐지 항). 창 이후 지점은 약한 양성(가중 0.2) | 애매한 시작 직후 지점을 억지로 올리지 않음 |
| **V2 (V1 + 응답 안 순위)** | V1에 더해, 같은 응답에서 **가장 높은 시작 전 지점 < 가장 높은 창 안 지점** (여유 1.0)을 요구하는 순위 손실. 시작 전 지점 중 현재 점수가 높은 4개를 매 단계 골라 씀(어려운 음성 채굴) | 엄격 재현율의 성공 조건을 그대로 학습 |
| **V3 (V2 + 정답 누락 완화)** | V2에서, 시작 문장보다 2문장 이상 앞 지점은 음성 가중 0.5, 순위 손실에서 제외 | 사례 검토에서 이른 경보의 30~47%가 정답 누락이었음 |

- 학습 데이터: gold-large 3,000응답(E2·student_vark와 같음)의 **모든 판정 지점**(문장 안 2토큰 간격 + 문장 끝, 약 28만 개). 1에폭.
- 체크포인트 선택: cal-L에서 고정한 300응답(cal-sub)의 엄격 재현율(2·4토큰 간격 평균)로, 에폭 중 4번.
- **1단계**: 시드 42로 V0~V3 학습 → cal-L로 변형과 판정 간격 선택(아래 규칙) → **2단계**: 선택된 변형과 V0를 시드 7, 2026으로 추가 학습.
- 평가: R0와 같은 질의·근거 검색으로 cal-L·dev-L 모든 판정 지점 점수를 계산하고, cal-L로 보정, dev-L로 판정. 기준선 B·E2 점수는 R0 파일을 그대로 쓴다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **선택 규칙**: 1단계(시드 42)의 V1·V2·V3 × 판정 방식 {문장 끝, 4토큰, 2토큰, 2토큰+EWMA(λ=0.5)} 중, cal-L 지연 중앙값 ≤ 9토큰인 조합에서 cal-L 엄격 재현율(ARL₀=100) 최대.
- **R1 주 기준** (선택된 변형·판정 방식, 시드 3개): dev-L ARL₀=100에서 **시드 평균 재현율 ≥ 0.531**, **시드 평균 지연 중앙값 ≤ 9토큰**, **시드 3개 중 2개 이상에서 B 문장 끝 대비 쌍체 부트스트랩 95% 하한 > 0**.
- **R1 목표 효과**: 같은 판정 방식에서 선택된 변형의 시드 평균 재현율 − V0 시드 평균 ≥ +0.03 (개선이 학습 목표 때문이라는 근거).
- **보고(판정 없음)**: 응답 단위 AUROC(정답 시작 위치에 덜 민감), 서비스 10% 재현율·지연, 완성 문장 AUROC, 발생 전 경보·끝까지 못 잡음 비율, 과제별, 변형 4개 전체 비교.

> 런타임: **GPU 필수, A100 또는 L4 권장**. 학습 1회(1에폭) A100 약 10~20분, T4 약 40~60분. 1단계 4회 + 2단계 4회 = 8회.
> 모델은 드라이브 `selfproject/r1/`에 저장되고, 이미 있는 모델은 건너뛴다(끊겨도 이어서 실행).
> 드라이브에 `student_vark`, R0 결과(`r0/r0_items.pkl`, `r0/r0_scores.npz`)가 있어야 한다.
""")

md("## 1. 환경 준비")
code(ENV)

md("## 2. 데이터 — gold-large 학습 응답 (E1·E2 코드 그대로)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)

md(r"""## 3. 학습 판정 지점 — 2토큰 간격 + 문장 끝, 지점마다 범주

- `neg`: 멀쩡한 응답 / `pre`: 시작 전(시작 문장과 바로 앞 문장) / `farpre`: 시작 문장보다 2문장 이상 앞
- `win`: 시작 후 10토큰 안(지연 목표 9토큰과 맞춤) / `post`: 그 이후""")
code(POINTS_DEF + r"""

WIN_TOK = 10
train_items = to_items(train)
TR, P_KEY, P_TXT, P_CAT, P_RESP = [], [], [], [], []
for x in train_items:
    sid = {sn['t0']: i for i, sn in enumerate(x['sents'])}
    on = x['onset']
    os_ = None if on is None else next((i for i, sn in enumerate(x['sents']) if sn['t0'] <= on <= sn['t1']), None)
    pts = []
    for t, txt in points(x, 2):
        if not txt: continue
        si = sid[x['sent_of'][t]]
        if on is None: cat = 'neg'
        elif t < on: cat = 'farpre' if (os_ is not None and si <= os_ - 2) else 'pre'
        elif t <= on + WIN_TOK: cat = 'win'
        else: cat = 'post'
        pts.append([t, txt, cat])
    if on is not None and not any(p[2] == 'win' for p in pts):
        for p in pts:                                   # 창 안에 지점이 없으면 시작 이후 첫 지점을 창으로
            if p[2] == 'post': p[2] = 'win'; break
    ids = []
    for t, txt, cat in pts:
        ids.append(len(P_TXT)); P_KEY.append(x['ref_key']); P_TXT.append(txt); P_CAT.append(cat); P_RESP.append(len(TR))
    TR.append({'ids': ids, 'hall': on is not None})
P_CAT = np.array(P_CAT)
print(f'학습 응답 {len(TR)} | 판정 지점 {len(P_TXT):,} |', pd.Series(P_CAT).value_counts().to_dict())
""")

md(r"""## 4. 근거 검색 (E2와 같은 설정) — 지점마다 top-8 조각 번호만 저장""")
code(RETR + r"""

# R0 평가 세트(cal-L, dev-L)와 기존 판정기 점수
import pickle
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
Z0 = dict(np.load(os.path.join(R0_DIR, 'r0_scores.npz')))
CAL, DEV = R0P['items']['calL'], R0P['items']['devL']
Q_KEYS, Q_TEXTS = R0P['q_keys'], R0P['q_texts']
for k in set(Q_KEYS):
    REFS.setdefault(k, build_reference(sources[k]))

@torch.no_grad()
def topk_idx(keys, texts, bs=512):
    out = []
    for i in range(0, len(texts), bs):
        embs = retriever.encode(texts[i:i + bs], convert_to_tensor=True, normalize_embeddings=True,
                                batch_size=bs, show_progress_bar=False)
        for key, e in zip(keys[i:i + bs], embs):
            ch, ce = get_chunks(key)
            out.append(sorted(torch.topk(ce @ e, min(K, ce.shape[0])).indices.tolist()))
    return out
ctx_of = lambda key, idx: ' ... '.join(get_chunks(key)[0][j] for j in idx)

t0 = time.perf_counter()
P_IDX = topk_idx(P_KEY, P_TXT)
print(f'학습 지점 근거 검색 {time.perf_counter()-t0:.0f}초')

# 평가 질의: cal-L·dev-L 지점만 (R0와 같은 질의 번호)
EVAL_Q = sorted({q for x in CAL + DEV for _, q in x['pts2']})
t0 = time.perf_counter()
EQ_IDX = dict(zip(EVAL_Q, topk_idx([Q_KEYS[q] for q in EVAL_Q], [Q_TEXTS[q] for q in EVAL_Q])))
print(f'평가 질의 {len(EVAL_Q):,}개 근거 검색 {time.perf_counter()-t0:.0f}초')

# 체크포인트 선택용 cal-sub (cal-L에서 300응답 고정)
CALSUB = random.Random(1).sample(CAL, min(300, len(CAL)))
""")

md(r"""## 5. 평가 함수 (E1 코드 그대로) + 판정 방식 `2토큰+EWMA`

점수기 이름은 `모델|판정 방식`. 판정 방식: `end`(문장 끝), `4`, `2`, `ew2`(2토큰 간격 점수를 응답 전체에 걸쳐 λ=0.5로 지수 평활).""")
code(r"""
SC = dict(Z0)

def series(x, scorer):
    m, k = scorer.split('|')
    S_ = SC[m]
    if k == 'end':
        return [(t, S_[q]) for t, q in x['pts2'] if t in x['ends']]
    if k == 'ew2':
        out, s = [], None
        for t, q in x['pts2']:
            v = float(S_[q]); s = v if s is None else 0.5 * v + 0.5 * s
            out.append((t, s))
        return out
    s = int(k)
    return [(t, S_[q]) for t, q in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % s == 0]
""")
code(EVAL_DEFS)
code(r"""
def resp_auc(items, m):
    y = [int(x['onset'] is not None) for x in items]
    s = [max([SC[m][q] for _, q in x['pts2']] or [-1e9]) for x in items]
    return roc_auc_score(y, s)

def sent_auc(items, m):
    ys, ss = [], []
    for x in items:
        d = dict(x['pts2'])
        for sn in x['sents']:
            if sn['t1'] in d: ys.append(sn['y']); ss.append(SC[m][d[sn['t1']]])
    return roc_auc_score(ys, ss)

def per_resp(items, sc, thr):
    d, l = [], []
    for x in items:
        if x['onset'] is None: continue
        first = next((t for t, s in series(x, sc) if s >= thr), None)
        ok = first is not None and first >= x['onset']
        d.append(float(ok)); l.append(first - x['onset'] if ok else np.nan)
    return np.array(d), np.array(l)

def boot(a, b, n=1000, seed=0):
    da, la = per_resp(DEV, a, calibrate(CAL, a, 100)); db, lb = per_resp(DEV, b, calibrate(CAL, b, 100))
    rng = np.random.default_rng(seed); N = len(da); rd = []
    for _ in range(n):
        i = rng.integers(0, N, N); rd.append(da[i].mean() - db[i].mean())
    lo, hi = np.percentile(rd, [2.5, 97.5])
    return {'A': a, 'B': b, 'recall_A': da.mean(), 'recall_B': db.mean(), 'diff': da.mean() - db.mean(), 'lo': lo, 'hi': hi}

# 기준선 확인 (R0 dev-L과 같아야 함)
for sc, ref in [('B|end', 0.5007), ('B|4', 0.4809), ('E2|4', 0.4506)]:
    got = evaluate(DEV, sc, calibrate(CAL, sc, 100))['recall']
    print(f'{sc:6} R0 {ref:.4f} | 지금 {got:.4f} | {"일치" if abs(got-ref) <= 0.002 else "불일치 — 확인 필요"}')
""")

md(r"""## 6. 학습 — 변형 4가지 공통 루프

한 배치 = 판정 지점 약 64개. 단일 지점 항(이진 분류)과 응답 단위 항(창 안 하나, 순위)을 섞는다.
V2·V3의 순위 항은 매 단계 그 응답의 시작 전 지점을 학습 중인 모델로 먼저 채점해(기울기 없이) 가장 높은 4개를 쓴다.""")
code(r"""
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

BATCH_SEQ, LR, POST_W, GROUP_W, MARGIN, MINE_M, N_EVALS = 64, 2e-5, 0.2, 5.0, 1.0, 4, 4
BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16

def encode(tok, idxs):
    ctx = [ctx_of(P_KEY[g], P_IDX[g]) for g in idxs]
    return tok(ctx, [P_TXT[g] for g in idxs], truncation='only_first', max_length=MAX_LEN,
               padding=True, return_tensors='pt').to(DEVICE)

def make_units(variant, rng):
    units = []
    for ri, r in enumerate(TR):
        for g in r['ids']:
            c = P_CAT[g]
            if variant == 'V0':
                units.append(('s', g, 0.0 if c in ('neg', 'pre', 'farpre') else 1.0, 1.0))
            elif c in ('neg', 'pre'):
                units.append(('s', g, 0.0, 1.0))
            elif c == 'farpre':
                units.append(('s', g, 0.0, 0.5 if variant == 'V3' else 1.0))
            elif c == 'post':
                units.append(('s', g, 1.0, POST_W))
        if r['hall'] and variant != 'V0':
            units.append(('g', ri))
    rng.shuffle(units)
    return units

@torch.no_grad()
def score_ids(tok, model, idxs, bs=256):
    model.eval(); out = []
    for i in range(0, len(idxs), bs):
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
            out.append(model(**encode(tok, idxs[i:i + bs])).logits.squeeze(-1).float().cpu())
    model.train()
    return torch.cat(out).numpy() if out else np.zeros(0)

@torch.no_grad()
def score_queries(tok, model, qs, bs=256):
    model.eval(); out = np.zeros(len(qs), dtype=np.float32)
    order = sorted(range(len(qs)), key=lambda i: len(Q_TEXTS[qs[i]]))
    for i in range(0, len(order), bs):
        b = [qs[j] for j in order[i:i + bs]]
        ctx = [ctx_of(Q_KEYS[q], EQ_IDX[q]) for q in b]
        enc = tok(ctx, [Q_TEXTS[q] for q in b], truncation='only_first', max_length=MAX_LEN,
                  padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
            out[order[i:i + bs]] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    model.train()
    return out

CALSUB_Q = sorted({q for x in CALSUB for _, q in x['pts2']})
def calsub_metric(tok, model):
    v = score_queries(tok, model, CALSUB_Q)
    SC['_cur'] = np.zeros(len(Q_KEYS), dtype=np.float32); SC['_cur'][CALSUB_Q] = v
    rec = [evaluate(CALSUB, f'_cur|{k}', calibrate(CALSUB, f'_cur|{k}', 100))['recall'] for k in ['4', '2']]
    return float(np.mean(rec))

def group_loss(variant, ri, tok, model):
    r = TR[ri]
    win = [g for g in r['ids'] if P_CAT[g] == 'win']
    neg_pool = [g for g in r['ids'] if P_CAT[g] == 'pre' or (P_CAT[g] == 'farpre' and variant == 'V2')]
    hard = []
    if variant in ('V2', 'V3') and neg_pool:
        s = score_ids(tok, model, neg_pool)
        hard = [neg_pool[j] for j in np.argsort(-s)[:MINE_M]]
    return win, hard

def train_variant(variant, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(STUDENT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR).to(DEVICE)
    units = make_units(variant, random.Random(seed))
    n_seq = sum(1 if u[0] == 's' else 6 for u in units)
    n_steps = max(1, n_seq // BATCH_SEQ)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda' and not BF16))
    best, best_state, hist = calsub_metric(tok, model), copy.deepcopy(model.state_dict()), []
    hist.append((0, best)); print(f'  [{variant} s{seed}] 시작 cal-sub {best:.3f} | 예상 {n_steps}단계')
    eval_at = {int(n_steps * (i + 1) / N_EVALS) for i in range(N_EVALS)}
    model.train(); step, ui, t0 = 0, 0, time.perf_counter()
    while ui < len(units):
        singles, groups, n = [], [], 0
        while ui < len(units) and n < BATCH_SEQ:
            u = units[ui]; ui += 1
            if u[0] == 's':
                singles.append(u); n += 1
            else:
                win, hard = group_loss(variant, u[1], tok, model)
                if win:
                    groups.append((win, hard)); n += len(win) + len(hard)
        ids = [u[1] for u in singles] + [g for w, h in groups for g in w + h]
        if not ids: continue
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
            z = model(**encode(tok, ids)).logits.squeeze(-1).float()
        ns = len(singles)
        y = torch.tensor([u[2] for u in singles], device=DEVICE); w = torch.tensor([u[3] for u in singles], device=DEVICE)
        loss_num = (F.binary_cross_entropy_with_logits(z[:ns], y, reduction='none') * w).sum() if ns else z.sum() * 0
        den = w.sum() if ns else torch.tensor(0.0, device=DEVICE)
        pos = ns
        for win, hard in groups:
            zw = z[pos:pos + len(win)]; zh = z[pos + len(win):pos + len(win) + len(hard)]; pos += len(win) + len(hard)
            s_no = F.logsigmoid(-zw).sum().clamp(max=-1e-6)              # 창 안에서 한 번도 안 울릴 로그확률
            l = -torch.log(-torch.expm1(s_no))                            # 적어도 하나 울릴 확률의 음의 로그
            if len(hard):
                l = l + F.softplus(torch.logsumexp(zh, 0) - torch.logsumexp(zw, 0) + MARGIN)
            loss_num = loss_num + GROUP_W * l; den = den + GROUP_W
        loss = loss_num / den.clamp(min=1e-6)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        step += 1
        if step in eval_at or ui >= len(units):
            va = calsub_metric(tok, model); hist.append((step, va))
            if va > best: best, best_state = va, copy.deepcopy(model.state_dict())
            print(f'  [{variant} s{seed}] 단계 {step} | 손실 {loss.item():.3f} | cal-sub {va:.3f} | {(time.perf_counter()-t0)/60:.1f}분')
    model.load_state_dict(best_state)
    return tok, model, best, hist

import shutil
def run(variant, seed):
    out = os.path.join(R1_DIR, f'{variant}_s{seed}')
    sp = os.path.join(out, 'scores.npy')
    if not os.path.exists(sp):
        has_w = any(os.path.exists(os.path.join(out, f)) for f in ['model.safetensors', 'pytorch_model.bin'])
        if os.path.exists(os.path.join(out, 'config.json')) and not has_w:
            print(f'  [{variant} s{seed}] 저장이 끊긴 폴더(가중치 없음) → 지우고 다시 학습'); shutil.rmtree(out)
        if has_w:
            tok = AutoTokenizer.from_pretrained(out); model = AutoModelForSequenceClassification.from_pretrained(out).to(DEVICE)
        else:
            tok, model, best, hist = train_variant(variant, seed)
            os.makedirs(out, exist_ok=True); tok.save_pretrained(out); model.save_pretrained(out)
            pd.DataFrame(hist, columns=['step', 'calsub_recall']).to_csv(os.path.join(out, 'hist.csv'), index=False)
        t0 = time.perf_counter()
        v = score_queries(tok, model, EVAL_Q)
        full = np.full(len(Q_KEYS), np.nan, dtype=np.float32); full[EVAL_Q] = v
        np.save(sp, full)
        print(f'  [{variant} s{seed}] 평가 질의 채점 {time.perf_counter()-t0:.0f}초')
        del model; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    SC[f'{variant}_s{seed}'] = np.load(sp)
    print(f'[{variant} s{seed}] 준비 완료')
""")

md("## 7. 1단계 — 시드 42로 V0~V3 학습")
code(r"""
for v in ['V0', 'V1', 'V2', 'V3']:
    run(v, 42)
""")

md(r"""## 8. 선택 — cal-L로 변형과 판정 방식 고르기 (사전 고정 규칙)

V1·V2·V3 × {문장 끝, 4, 2, 2+EWMA} 중 cal-L 지연 중앙값 ≤ 9토큰에서 cal-L 엄격 재현율 최대. dev-L은 보지 않는다.""")
code(r"""
KINDS = ['end', '4', '2', 'ew2']
sel = []
for v in ['V0', 'V1', 'V2', 'V3']:
    for k in KINDS:
        sc = f'{v}_s42|{k}'
        r = evaluate(CAL, sc, calibrate(CAL, sc, 100))
        sel.append({'variant': v, 'kind': k, 'cal_recall': r['recall'], 'cal_delay': r['delay_tok_median'],
                    'cal_pre_onset': r['pre_onset_alarm_rate']})
sel_df = pd.DataFrame(sel)
print(sel_df.round(3).to_string(index=False))
elig = sel_df[(sel_df.variant != 'V0') & (sel_df.cal_delay <= 9)].sort_values('cal_recall', ascending=False)
SEL_FALLBACK = len(elig) == 0
if SEL_FALLBACK:      # 10/7 추가(실행 중 발견, dev-L 보기 전): 지연 조건을 만족하는 조합이 없으면 지연 조건 없이 cal-L 재현율 최대
    print('⚠️ cal-L 지연 ≤ 9토큰인 V1~V3 조합 없음 → 대체 규칙: 지연 조건 없이 cal-L 재현율 최대 (SSOT 결정 기록)')
    elig = sel_df[sel_df.variant != 'V0'].sort_values('cal_recall', ascending=False)
CHOSEN_V, CHOSEN_K = elig.iloc[0].variant, elig.iloc[0].kind
print(f'\n선택: {CHOSEN_V}, 판정 방식 {CHOSEN_K}')
""")

md("## 9. 2단계 — 선택된 변형과 대조군 V0를 시드 7, 2026으로")
code(r"""
for seed in [7, 2026]:
    for v in [CHOSEN_V, 'V0']:
        run(v, seed)
""")

md(r"""## 10. dev-L 평가와 사전 기준 판정""")
code(r"""
SEEDS = [42, 7, 2026]
rows = []
models = ['B', 'E2'] + [m for m in Z0 if m.startswith('S')] + [f'{v}_s{s}' for v in ['V0', 'V1', 'V2', 'V3'] for s in SEEDS if f'{v}_s{s}' in SC]
for m in models:
    ra, sa = resp_auc(DEV, m), sent_auc(DEV, m)
    for k in KINDS:
        sc = f'{m}|{k}'
        r = evaluate(DEV, sc, calibrate(CAL, sc, 100))
        r10 = evaluate(DEV, sc, calibrate_flag(CAL, sc, 0.10))
        rows.append({'model': m, 'kind': k, 'recall': r['recall'], 'delay_med': r['delay_tok_median'],
                     'delay_chars': r['delay_chars_median'], 'pre_onset': r['pre_onset_alarm_rate'],
                     'realized_ARL0': r['realized_ARL0_dev'], 'svc_recall': r10['recall'], 'svc_delay': r10['delay_tok_median'],
                     'svc_flag': r10['clean_resp_flagged'], 'resp_auc': ra, 'sent_auc': sa})
res = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(res[res.kind.isin(['end', '4'])].round(3).to_string(index=False))

def seed_mean(v, k, col='recall'):
    return float(np.mean([res[(res.model == f'{v}_s{s}') & (res.kind == k)][col].iloc[0] for s in SEEDS if f'{v}_s{s}' in SC]))

rec, dl = seed_mean(CHOSEN_V, CHOSEN_K), seed_mean(CHOSEN_V, CHOSEN_K, 'delay_med')
bts = [boot(f'{CHOSEN_V}_s{s}|{CHOSEN_K}', 'B|end') for s in SEEDS]
n_sig = sum(b['lo'] > 0 for b in bts)
c1, c2, c3 = rec >= 0.531, dl <= 9, n_sig >= 2
R1 = c1 and c2 and c3
gain = rec - seed_mean('V0', CHOSEN_K)
R1_OBJ = gain >= 0.03
print(f"\n=== R1 사전 기준 ({CHOSEN_V}, {CHOSEN_K}, 시드 {SEEDS}) ===")
print(f"시드 평균 재현율 {rec:.3f} ≥ 0.531: {'충족' if c1 else '미달'}")
print(f"시드 평균 지연 {dl:.1f} ≤ 9: {'충족' if c2 else '미달'}")
for b in bts:
    print(f"   {b['A']}: B 문장 끝 대비 {b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
print(f"B 문장 끝 대비 유의한 시드 {n_sig}/3 (≥ 2): {'충족' if c3 else '미달'}")
print('R1 주 기준:', '충족' if R1 else '미달')
print(f"목표 효과: V0 대비 시드 평균 {gain:+.3f} (≥ +0.03): {'충족' if R1_OBJ else '미달'}")

agg = []
for v in ['V0', CHOSEN_V]:
    for k in KINDS:
        agg.append({'variant': v, 'kind': k, **{c: seed_mean(v, k, c) for c in
                    ['recall', 'delay_med', 'pre_onset', 'svc_recall', 'svc_delay', 'resp_auc', 'sent_auc']}})
for m, k in [('B', 'end'), ('B', '4'), ('E2', '4')]:
    r = res[(res.model == m) & (res.kind == k)].iloc[0]
    agg.append({'variant': m, 'kind': k, **{c: r[c] for c in ['recall', 'delay_med', 'pre_onset', 'svc_recall', 'svc_delay', 'resp_auc', 'sent_auc']}})
agg_df = pd.DataFrame(agg)
print('\n시드 평균 비교 (dev-L)'); print(agg_df.round(3).to_string(index=False))
""")

md("## 11. 과제별 (선택된 변형 시드 평균 vs B 문장 끝, 보고)")
code(r"""
tr = []
for tt in TASKS:
    its = [x for x in DEV if x['task_type'] == tt]
    row = {'task': tt, 'n_hall': sum(x['onset'] is not None for x in its)}
    for sc in ['B|end', 'B|4'] + [f'{CHOSEN_V}_s{s}|{CHOSEN_K}' for s in SEEDS]:
        row[sc] = evaluate(its, sc, calibrate(CAL, sc, 100))['recall']
    tr.append(row)
task_df = pd.DataFrame(tr)
print(task_df.round(3).to_string(index=False))
""")

md("## 12. 저장")
code(r"""
import shutil
summ = {'chosen_variant': CHOSEN_V, 'chosen_kind': CHOSEN_K, 'selection_fallback': bool(SEL_FALLBACK), 'seed_mean_recall': rec, 'seed_mean_delay': dl,
        'n_sig_vs_Bend': int(n_sig), 'R1': bool(R1), 'gain_vs_V0': gain, 'R1_objective': bool(R1_OBJ),
        'boot': bts}
json.dump(summ, open('r1_summary.json', 'w'), default=float, ensure_ascii=False, indent=1)
hists = []
for v in ['V0', 'V1', 'V2', 'V3']:
    for s in SEEDS:
        p = os.path.join(R1_DIR, f'{v}_s{s}', 'hist.csv')
        if os.path.exists(p):
            h = pd.read_csv(p); h['run'] = f'{v}_s{s}'; hists.append(h)
out = {'r1_devL.csv': res, 'r1_select_cal.csv': sel_df, 'r1_seedmean.csv': agg_df, 'r1_task.csv': task_df,
       'r1_hist.csv': pd.concat(hists) if hists else pd.DataFrame()}
for f, d in out.items():
    d.to_csv(f, index=False); d.to_csv(os.path.join(R1_DIR, f), index=False)
shutil.copy('r1_summary.json', os.path.join(R1_DIR, 'r1_summary.json'))
os.system('zip -q r1_results.zip ' + ' '.join(list(out) + ['r1_summary.json']))
print('저장 완료: 드라이브 selfproject/r1/, 업로드용 r1_results.zip')
try:
    from google.colab import files; files.download('r1_results.zip')
except Exception:
    pass
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r1_onset_aware_training.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
