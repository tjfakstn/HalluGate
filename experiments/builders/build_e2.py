"""E2 노트북 생성. 평가 코드(데이터 분할, 토큰화·발생 위치, calibrate/evaluate)는 E1 노트북에서 그대로 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
e1 = json.load(open(os.path.join(NB_DIR, 'e1_streaming_protocol_baselines.ipynb')))
e1_code = [''.join(c['source']) for c in e1['cells'] if c['cell_type'] == 'code']
E1_DATA = next(c for c in e1_code if c.startswith("BASE = 'RAGTruth/dataset'"))
E1_PREP = next(c for c in e1_code if c.startswith("ENC = tiktoken.get_encoding"))
E1_EVAL = next(c for c in e1_code if c.startswith("def calibrate(items, scorer, arl0):"))
E1_EVAL_DEFS = E1_EVAL.split('\nrows = []')[0]          # 함수 정의만 (calibrate, evaluate, calibrate_flag)
assert 'def evaluate' in E1_EVAL_DEFS and 'def calibrate_flag' in E1_EVAL_DEFS

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# E2 — 문장 앞부분을 판정하도록 학습한 인과적 점수기

## 왜 하나

E1에서 두 기준선의 장단점이 갈렸다 (엄격 기준, ARL₀=100, dev).

| 기준선 | 재현율 | 탐지 지연 중앙값 |
|---|---|---|
| B1 문장 끝 판정 | **0.474** | 17토큰 |
| B2 앞부분 재판정 s=8 (완성 문장용 판정기를 앞부분에 그대로 사용) | 0.448 | **9토큰** |

B2의 재현율이 떨어지는 이유는 판정기가 **완성된 문장으로만 학습**되어, 덜 끝난 문장에서 판단이 불안정하기 때문이라고 본다(가설 H1, E1에서 지지).
E2는 판정기에게 **"지금까지 나온 부분만 보고, 환각이 이미 시작됐는가"**를 직접 가르친다.

## 무엇을 하나

1. **앞부분 학습 데이터**: 학습 응답(gold-large 3,000개, dev·검증과 겹치지 않음)의 각 문장을 여러 위치에서 자른다.
   정답은 RAGTruth 환각 구간으로 만든다. **자른 앞부분 안에 환각 구간이 한 글자라도 들어 있으면 양성**이다.
   - 환각 없는 문장: 무작위 위치 3곳 + 문장 전체
   - 환각 있는 문장: 무작위 2곳 + **발생 직전·직후 구간(θ−1 ~ θ+8)에서 2곳** + 문장 전체
   - 앞부분은 최소 2토큰
2. **발생 직후 가중치**: 발생 후 8토큰 안의 양성 앞부분은 손실 가중치 2배. 조기 탐지의 핵심은 여기를 맞히는 것이다.
3. **모델**: student_vark(완성 문장 판정기, AUROC 0.859)에서 이어서 학습. 같은 구조(MiniLM-L6)이므로 **판정 1회 지연은 그대로**(약 17ms)다.
   근거 검색도 E1과 같다(앞부분 텍스트로 top-8 조각 검색, 512토큰).
4. **평가**: E1과 **같은 코드**로 한다. 기준선 B1·B2도 이 노트북에서 다시 계산해 E1 결과와 일치하는지 먼저 확인한다.

## 사전 정의 기준 (SSOT §6에 고정됨, 결과 보기 전)

- **주 기준**: 엄격 기준, ARL₀=100 dev에서 **재현율 ≥ 0.474 그리고 탐지 지연 중앙값 ≤ 9토큰**
  (두 기준선의 장점을 동시에 달성). 재판정 간격 s=2·4·8 중 하나라도 만족하면 충족으로 본다.
- **도전**: 재현율 ≥ 0.50 그리고 지연 ≤ 8토큰
- **보조**: 서비스 조건(멀쩡한 응답 표시 10%로 보정)에서 재현율 ≥ 0.383, 완성 문장 AUROC ≥ 0.85
- 임계값 보정과 학습 중 체크포인트 선택은 **검증 분할**로만 한다. dev는 최종 측정에만 쓴다.

> 런타임 **GPU(T4)**. 드라이브의 `selfproject/student_vark` 필요. 학습 약 20~30분 + 평가 약 10분.
> 학습된 모델은 드라이브 `selfproject/student_prefix/`에 저장된다.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install pysbd sentence-transformers tiktoken
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'

import json, time, random, os, gc, copy
import numpy as np, pandas as pd, torch, pysbd, tiktoken
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

USE_DRIVE = True
OUT_DIR = '/content/drive/MyDrive/selfproject' if USE_DRIVE else '.'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
STUDENT_DIR = os.path.join(OUT_DIR, 'student_vark')
SAVE_DIR = os.path.join(OUT_DIR, 'student_prefix')
assert os.path.isdir(STUDENT_DIR), f'{STUDENT_DIR} 없음'
print('device', DEVICE)
""")

md("## 2. 데이터 — E1과 같은 dev·검증 분할 (E1 코드 그대로)")
code(E1_DATA)

code(r"""
# 학습 응답: S2/S2-b의 gold-large와 같은 구성 (teacher 학습 510 + 추가 2,490). dev·검증과 겹치지 않음을 확인한다.
teacher_train = teacher_resp[len(val):]
used = dev_ids | {r['id'] for r in teacher_resp}
gl_extra = rng.sample([r for r in rest if r['id'] not in used], 3000 - len(teacher_train))
train = teacher_train + gl_extra
tr_ids = {r['id'] for r in train}
assert not (tr_ids & dev_ids) and not (tr_ids & {r['id'] for r in val}), '학습과 dev/검증 겹침'
print(f'학습 {len(train)} 응답 (dev·검증과 겹침 없음)')
""")

md("## 3. 토큰 스트림과 발생 위치 (E1 코드 그대로)")
code(E1_PREP)
code(r"""
train_items = to_items(train)
print(f'학습: 응답 {len(train_items)}, 문장 {sum(len(x["sents"]) for x in train_items):,}')
""")

md(r"""## 4. 앞부분 학습 데이터

문장 안의 자르는 위치 c(토큰 번호)에 대해, 판정 텍스트는 문장 시작부터 토큰 c의 끝까지다.
정답은 그 앞부분에 환각 토큰이 하나라도 있으면 1이다. 문장 단위 발생 위치는 문장 안에서 처음 환각 구간과 겹치는 토큰으로 정한다.""")
code(r"""
MIN_PREFIX, NEAR_AFTER, W_NEAR = 2, 8, 2.0

def sent_onset(x, sn):
    # 문장 안 첫 환각 토큰 번호 (없으면 None). 응답 전체의 첫 발생이 아니라 '이 문장 안'의 첫 발생
    spans = [(l['start'], l['end']) for l in x['labels']]
    for t in range(sn['t0'], sn['t1'] + 1):
        if any(min(x['te'][t], b) > max(x['cs'][t], a) for a, b in spans):
            return t
    return None

def prefix_examples(x, rng):
    out = []
    for sn in x['sents']:
        t0, t1 = sn['t0'], sn['t1']
        if t1 - t0 + 1 < MIN_PREFIX: continue
        on = sent_onset(x, sn)
        cand = list(range(t0 + MIN_PREFIX - 1, t1))              # 문장 끝 제외한 자르는 위치
        cuts = {t1}
        if on is None:
            cuts |= set(rng.sample(cand, min(3, len(cand))))
        else:
            cuts |= set(rng.sample(cand, min(2, len(cand))))
            near = [c for c in range(max(t0 + MIN_PREFIX - 1, on - 1), min(t1, on + NEAR_AFTER + 1))]
            cuts |= set(rng.sample(near, min(2, len(near))))
        for c in sorted(cuts):
            text = x['text'][sn['a']:min(x['te'][c], sn['b'])].strip()
            if not text: continue
            y = int(on is not None and on <= c)
            w = W_NEAR if (y and c - on <= NEAR_AFTER) else 1.0
            out.append({'ref_key': x['ref_key'], 'text': text, 'y': y, 'w': w, 'full': c == t1})
    return out

# prep()에 labels·cs가 필요하므로 원본 응답에서 붙인다
by_id = {str(r['id']): r for r in train}
for x in train_items:
    r = by_id[x['id']]
    x['labels'] = r.get('labels') or []
    text = x['text']
    x['cs'] = [a + (len(text[a:b]) - len(text[a:b].lstrip())) if text[a:b].strip() else a
               for a, b in zip(x['ts'], x['te'])]

prng = random.Random(SEED)
train_ex = pd.DataFrame([e for x in train_items for e in prefix_examples(x, prng)])
print(f'앞부분 예시 {len(train_ex):,}개 | 양성 {train_ex.y.mean():.1%} | 완성 문장 {train_ex.full.mean():.1%} | '
      f'발생 직후 가중 {(train_ex.w > 1).mean():.1%}')
""")

md("## 5. 판정기와 근거 검색 (E1과 같은 설정)")
code(r"""
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
K, MAX_LEN, CHUNK_WORDS, BS = 8, 512, 40, 32

chunk_cache = {}
def get_chunks(key):
    if key not in chunk_cache:
        w = REFS[key].split()
        ch = [' '.join(w[i:i + CHUNK_WORDS]) for i in range(0, max(len(w), 1), CHUNK_WORDS)] or [REFS[key]]
        chunk_cache[key] = (ch, retriever.encode(ch, convert_to_tensor=True, normalize_embeddings=True,
                                                 batch_size=128, show_progress_bar=False))
    return chunk_cache[key]

def contexts(keys, texts, bs=256):
    # 텍스트마다 근거 top-K 조각을 원문 순서로 이어 붙인 문자열
    out = []
    for i in range(0, len(texts), bs):
        embs = retriever.encode(texts[i:i + bs], convert_to_tensor=True, normalize_embeddings=True,
                                batch_size=bs, show_progress_bar=False)
        for key, e in zip(keys[i:i + bs], embs):
            ch, ce = get_chunks(key)
            top = torch.topk(ce @ e, min(K, ce.shape[0])).indices.tolist()
            out.append(' ... '.join(ch[t] for t in sorted(top)))
    return out

@torch.no_grad()
def score(tok, model, keys, texts, bs=128):
    model.eval()
    out = np.zeros(len(texts), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        ctx = contexts([keys[j] for j in idx], [texts[j] for j in idx])
        enc = tok(ctx, [texts[j] for j in idx], truncation='only_first', max_length=MAX_LEN,
                  padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
            out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out

t0 = time.perf_counter()
train_ex['ctx'] = contexts(train_ex.ref_key.tolist(), train_ex.text.tolist())
print(f'학습 예시 근거 검색 {time.perf_counter()-t0:.0f}초')
""")

md(r"""## 6. 판정 지점 (E1과 같음) — 검증 분할 앞부분 AUROC로 체크포인트 선택""")
code(r"""
def points(item, s):
    pts = []
    for sn in item['sents']:
        for t in range(sn['t0'], sn['t1'] + 1):
            k = t - sn['t0'] + 1
            if (s is not None and k % s == 0) or t == sn['t1']:
                pts.append((t, item['text'][sn['a']:min(item['te'][t], sn['b'])].strip()))
    return pts

QUERY_ID, queries = {}, []
def qid(key, text):
    if (key, text) not in QUERY_ID:
        QUERY_ID[(key, text)] = len(queries); queries.append((key, text))
    return QUERY_ID[(key, text)]

for x in dev_items + val_items:
    x['pts2'] = [(t, qid(x['ref_key'], txt)) for t, txt in points(x, 2) if txt]
Q_KEYS, Q_TEXTS = [q[0] for q in queries], [q[1] for q in queries]

# 검증 분할 앞부분의 정답 (체크포인트 선택용): 판정 지점 t까지 이 문장 안에 환각이 시작됐는가
def point_labels(items):
    ids, ys = [], []
    for x in items:
        spans = [(l['start'], l['end']) for l in x['labels']]
        for t, q in x['pts2']:
            a = x['sent_of'][t]
            ys.append(int(any(min(x['te'][u], b) > max(x['cs'][u], s0) for u in range(a, t + 1) for s0, b in spans)))
            ids.append(q)
    return np.array(ids), np.array(ys)

by_all = {str(r['id']): r for r in dev + val}
for x in dev_items + val_items:
    x['labels'] = by_all[x['id']].get('labels') or []
    text = x['text']
    x['cs'] = [a + (len(text[a:b]) - len(text[a:b].lstrip())) if text[a:b].strip() else a
               for a, b in zip(x['ts'], x['te'])]
VAL_Q, VAL_Y = point_labels(val_items)
print(f'판정 질의 {len(queries):,}개 | 검증 판정 지점 {len(VAL_Q):,}개 (양성 {VAL_Y.mean():.1%})')
""")

md("## 7. 학습 — student_vark에서 이어서, 1 에폭, 에폭 중 3회 검증")
code(r"""
def val_auroc(tok, model):
    uq = np.unique(VAL_Q)
    s = score(tok, model, [Q_KEYS[i] for i in uq], [Q_TEXTS[i] for i in uq])
    m = dict(zip(uq, s))
    return roc_auc_score(VAL_Y, [m[q] for q in VAL_Q])

def train_prefix(init_dir, epochs=1, lr=2e-5, n_evals=3):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(init_dir)
    model = AutoModelForSequenceClassification.from_pretrained(init_dir).to(DEVICE)
    base = val_auroc(tok, model)
    print(f'  시작점 검증 앞부분 AUROC {base:.3f}')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_step = epochs * ((len(train_ex) + BS - 1) // BS)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.06 * n_step)), n_step)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    eval_at = set(np.linspace(0, n_step, n_evals + 1, dtype=int)[1:])
    best, best_state, step, hist = base, copy.deepcopy(model.state_dict()), 0, [(0, base)]
    for ep in range(epochs):
        order = np.random.permutation(len(train_ex))
        for i in range(0, len(order), BS):
            model.train()
            b = train_ex.iloc[order[i:i + BS]]
            enc = tok(b.ctx.tolist(), b.text.tolist(), truncation='only_first', max_length=MAX_LEN,
                      padding=True, return_tensors='pt').to(DEVICE)
            y = torch.tensor(b.y.values, dtype=torch.float32, device=DEVICE)
            w = torch.tensor(b.w.values, dtype=torch.float32, device=DEVICE)
            with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
                logit = model(**enc).logits.squeeze(-1)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y, reduction='none') * w).mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step in eval_at:
                va = val_auroc(tok, model); hist.append((step, va))
                print(f'  step {step}/{n_step}  loss {loss.item():.3f}  검증 앞부분 AUROC {va:.3f}')
                if va > best:
                    best, best_state = va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return tok, model, best, hist

t0 = time.perf_counter()
tok_e2, model_e2, best_va, hist = train_prefix(STUDENT_DIR)
print(f'학습 완료 {(time.perf_counter()-t0)/60:.1f}분 | 선택된 검증 앞부분 AUROC {best_va:.3f}')
if best_va <= hist[0][1]:
    print('⚠ 학습 중 검증 앞부분 AUROC가 시작점보다 나아지지 않아 시작점(student_vark)을 그대로 선택함 → E2 = 기준선')
tok_e2.save_pretrained(SAVE_DIR); model_e2.save_pretrained(SAVE_DIR)
print('저장:', SAVE_DIR)
""")

md("## 8. 판정 — 기준선(student_vark)과 E2를 같은 질의에 적용")
code(r"""
tok_b = AutoTokenizer.from_pretrained(STUDENT_DIR)
model_b = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR).to(DEVICE)
t0 = time.perf_counter()
SC = {'B': score(tok_b, model_b, Q_KEYS, Q_TEXTS), 'E2': score(tok_e2, model_e2, Q_KEYS, Q_TEXTS)}
print(f'판정 {len(queries):,}개 × 2 모델, {time.perf_counter()-t0:.0f}초')

def series(x, scorer):
    # 'B1 문장 끝' / 'B2 앞부분 (s=8)' → 기준선 점수, 'E2 …' → E2 점수. 판정 지점 규칙은 E1과 같다.
    S = SC['E2'] if scorer.startswith('E2') else SC['B']
    if '문장 끝' in scorer:
        return [(t, S[q]) for t, q in x['pts2'] if t in x['ends']]
    s = int(scorer.split('s=')[1].rstrip(')'))
    return [(t, S[q]) for t, q in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % s == 0]
""")
code(E1_EVAL_DEFS)

md("## 9. 점검 — 기준선이 E1 결과와 일치하는가")
code(r"""
e1_path = os.path.join(OUT_DIR, 'e1', 'e1_summary.csv')
ref = pd.read_csv(e1_path) if os.path.exists(e1_path) else None
for sc in ['B1 문장 끝', 'B2 앞부분 (s=8)']:
    r = evaluate(dev_items, sc, calibrate(val_items, sc, 100))
    msg = f"{sc:16} 재현율 {r['recall']:.3f}, 지연 중앙값 {r['delay_tok_median']}"
    if ref is not None:
        e = ref[(ref.target_ARL0 == 100) & (ref.scorer == sc)].iloc[0]
        ok = abs(e.recall - r['recall']) < 0.02
        msg += f" | E1: {e.recall:.3f}, {e.delay_tok_median} → {'일치' if ok else '불일치 — 확인 필요'}"
    print(msg)
""")

md("## 10. 결과 — 엄격 기준, ARL₀ 보정(검증) → dev 평가")
code(r"""
SCORERS = ['B1 문장 끝', 'B2 앞부분 (s=8)', 'B2 앞부분 (s=4)', 'B2 앞부분 (s=2)',
           'E2 문장 끝', 'E2 앞부분 (s=8)', 'E2 앞부분 (s=4)', 'E2 앞부분 (s=2)']
rows = []
for arl0 in [50, 100, 200, 500]:
    for sc in SCORERS:
        thr = calibrate(val_items, sc, arl0)
        rows.append({'target_ARL0': arl0, 'target_clean_flag': None, 'scorer': sc, 'thr': thr, **evaluate(dev_items, sc, thr)})
for rate in [0.05, 0.10, 0.20]:
    for sc in SCORERS:
        thr = calibrate_flag(val_items, sc, rate)
        rows.append({'target_ARL0': None, 'target_clean_flag': rate, 'scorer': sc, 'thr': thr, **evaluate(dev_items, sc, thr)})
e2 = pd.DataFrame(rows)
pd.set_option('display.width', 220)
cols = ['scorer', 'recall', 'delay_tok_mean', 'delay_tok_median', 'delay_chars_median', 'clean_resp_flagged',
        'realized_ARL0_dev', 'pre_onset_alarm_rate', 'calls_per_resp']
print('=== ARL₀=100 (엄격 기준) ===')
print(e2[e2.target_ARL0 == 100][cols].round(3).to_string(index=False))
print('\n=== 서비스 조건: 멀쩡한 응답 표시 10% ===')
print(e2[e2.target_clean_flag == 0.10][cols].round(3).to_string(index=False))
""")

code(r"""
# 완성 문장 판정 능력 (보조 기준 AUROC ≥ 0.85)
ys, sb, se = [], [], []
for x in dev_items:
    d = dict(x['pts2'])
    for sn in x['sents']:
        ys.append(sn['y']); sb.append(SC['B'][d[sn['t1']]]); se.append(SC['E2'][d[sn['t1']]])
AUC_B, AUC_E2 = roc_auc_score(ys, sb), roc_auc_score(ys, se)
# 앞부분 판정 능력 (dev 판정 지점)
DEV_Q, DEV_Y = point_labels(dev_items)
PAUC_B, PAUC_E2 = roc_auc_score(DEV_Y, SC['B'][DEV_Q]), roc_auc_score(DEV_Y, SC['E2'][DEV_Q])
print(f'완성 문장 AUROC: 기준선 {AUC_B:.3f} → E2 {AUC_E2:.3f}')
print(f'앞부분(모든 판정 지점) AUROC: 기준선 {PAUC_B:.3f} → E2 {PAUC_E2:.3f}')
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
a = e2[e2.target_ARL0 == 100]
for r in a.itertuples():
    m = 'o' if r.scorer.startswith('E2') else 's'
    ax[0].scatter(r.delay_tok_median, r.recall, marker=m, s=60, label=r.scorer)
ax[0].axhline(0.474, color='gray', ls=':', lw=1); ax[0].axvline(9, color='gray', ls=':', lw=1)
ax[0].scatter([17.9], [0.40], marker='x', c='k', label='Itkin HistGBM (test, mean delay)')
ax[0].scatter([11.5], [0.24], marker='+', c='k', label='Itkin CUSUM (test, mean delay)')
ax[0].set_xlabel('median detection delay (tokens)'); ax[0].set_ylabel('strict recall'); ax[0].set_title('ARL0 = 100 (upper-left is better)')
ax[0].legend(fontsize=7)
for sc in ['B1 문장 끝', 'B2 앞부분 (s=8)', 'E2 앞부분 (s=8)', 'E2 앞부분 (s=4)']:
    g = e2[(e2.scorer == sc) & e2.target_ARL0.notna()]
    ax[1].plot(g.target_ARL0, g.recall, 'o-', label=sc)
ax[1].set_xscale('log'); ax[1].set_xlabel('target ARL0'); ax[1].set_ylabel('strict recall'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md("## 11. 과제별 (ARL₀=100)")
code(r"""
rows = []
for tt in TASKS:
    its = [x for x in dev_items if x['task_type'] == tt]
    for sc in ['B1 문장 끝', 'B2 앞부분 (s=8)', 'E2 앞부분 (s=8)', 'E2 앞부분 (s=4)']:
        thr = e2[(e2.target_ARL0 == 100) & (e2.scorer == sc)].thr.item()
        r = evaluate(its, sc, thr)
        rows.append({'task': tt, 'scorer': sc, 'recall': r['recall'], 'delay_tok_median': r['delay_tok_median'],
                     'clean_resp_flagged': r['clean_resp_flagged']})
print(pd.DataFrame(rows).round(3).to_string(index=False))
""")

md("## 12. 사전 기준 판정과 저장")
code(r"""
a = e2[e2.target_ARL0 == 100].set_index('scorer')
svc = e2[e2.target_clean_flag == 0.10].set_index('scorer')
print('=== 주 기준: 재현율 ≥ 0.474 그리고 지연 중앙값 ≤ 9토큰 (ARL₀=100) ===')
main_ok, stretch_ok = [], []
for sc in ['E2 앞부분 (s=8)', 'E2 앞부분 (s=4)', 'E2 앞부분 (s=2)']:
    r = a.loc[sc]
    m = r.recall >= 0.474 and r.delay_tok_median <= 9
    st = r.recall >= 0.50 and r.delay_tok_median <= 8
    main_ok.append(m); stretch_ok.append(st)
    print(f'  {sc:16} 재현율 {r.recall:.3f}, 지연 {r.delay_tok_median:.1f} → 주 기준 {"충족" if m else "미달"}'
          f'{", 도전 목표 달성" if st else ""}')
best = max(['E2 앞부분 (s=8)', 'E2 앞부분 (s=4)', 'E2 앞부분 (s=2)'], key=lambda s: svc.loc[s].recall)
print(f'\n보조 ① 서비스 조건 재현율 ≥ 0.383: 최고 {best} {svc.loc[best].recall:.3f} → {"충족" if svc.loc[best].recall >= 0.383 else "미달"}')
print(f'보조 ② 완성 문장 AUROC ≥ 0.85: {AUC_E2:.3f} → {"충족" if AUC_E2 >= 0.85 else "미달"}')
print('\n종합:', '주 기준 충족' if any(main_ok) else '주 기준 미달', '| 도전 목표', '달성' if any(stretch_ok) else '미달')

os.makedirs(os.path.join(OUT_DIR, 'e2'), exist_ok=True)
e2.to_csv(os.path.join(OUT_DIR, 'e2', 'e2_summary.csv'), index=False)
pd.DataFrame(hist, columns=['step', 'val_prefix_auroc']).to_csv(os.path.join(OUT_DIR, 'e2', 'e2_train_hist.csv'), index=False)
pd.DataFrame({'auc_full_B': [AUC_B], 'auc_full_E2': [AUC_E2], 'auc_prefix_B': [PAUC_B], 'auc_prefix_E2': [PAUC_E2],
              'n_train_prefix': [len(train_ex)]}).to_csv(os.path.join(OUT_DIR, 'e2', 'e2_auc.csv'), index=False)
pts = [{'id': x['id'], 'split': sp, 'task_type': x['task_type'], 'onset': x['onset'], 'n_tok': x['n_tok'], 't': t,
        'score_B': float(SC['B'][q]), 'score_E2': float(SC['E2'][q]), 'is_sent_end': t in x['ends']}
       for sp, its in [('dev', dev_items), ('val', val_items)] for x in its for t, q in x['pts2']]
pd.DataFrame(pts).to_csv(os.path.join(OUT_DIR, 'e2', 'e2_points.csv'), index=False)
print('저장:', os.path.join(OUT_DIR, 'e2'))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(NB_DIR, 'e2_prefix_trained_scorer.ipynb'), 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
