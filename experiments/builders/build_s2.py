import os
import json
cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r"""# S2 — MiniCheck 증류: 빠른 학생 판정기 만들기

## 왜 하나

S1 게이트키퍼 실험에서 가벼운 규칙 신호만으로는 환각의 95%를 잡기 위해 문장의 82%를 MiniCheck로
보내야 했고, 사전 기준(50% 이하)에 미달했다. 정해둔 규칙에 따라 S2는 **증류** 방향으로 전환한다.
MiniCheck(선생 모델, 7억 파라미터, 문장당 약 400ms)의 판단을 **작은 학생 모델**이 배우게 하여,
정확도는 최대한 유지하면서 문장당 지연을 크게 줄이는 것이 목표다.

S1에서 단순한 어휘 신호만으로도 MiniCheck가 가장 의심하는 상위 20% 문장을 AUROC 0.812로 맞혔다.
MiniCheck의 판단에는 흉내 낼 수 있는 패턴이 있다는 뜻이며, 이것이 증류를 시도하는 근거다.

## 학생 모델의 구조 — 근거를 다 읽지 않는다

MiniCheck가 느린 이유는 큰 모델이 긴 근거 문서 전체를 읽기 때문이다. 학생은 두 가지로 이를 줄인다.

1. **작은 모델**: NLI로 사전학습된 소형 교차 인코더 (MiniLM-L6 2천만 파라미터급, DeBERTa-v3-xsmall)
2. **근거 검색**: 근거 문서를 40단어 조각으로 나눠 미리 임베딩해두고, 생성된 문장과 가장 비슷한
   조각 k개(기본 4개)만 학생에게 보여준다. 3주차에 "창 3~4개에서 성능이 포화된다"는 결과가 이 설계의 근거다.

```
문장 → [임베딩으로 근거 조각 top-k 검색] → [학생 교차 인코더] → 환각 점수
```

## 비교하는 학습 조건

| 조건 | 학습 목표 | 의미 |
|---|---|---|
| A. gold | RAGTruth 정답 레이블 | 증류 없이 정답만으로 학습 |
| B. teacher | MiniCheck 점수 (soft label) | 순수 증류 |
| C. mix | 정답과 MiniCheck 점수의 평균 | 둘을 결합 |
| D. gold-large | 정답 레이블, 데이터 6배 | 정답은 공짜이므로 양으로 승부 |

A·B·C는 **같은 문장**으로 학습하므로 학습 목표의 효과만 비교된다.
D는 "MiniCheck로 레이블을 만드는 비용(문장당 0.4초)을 들일 가치가 있는가"에 답한다.
RAGTruth에는 정답이 이미 있으므로, 정답만 많이 써도 충분하다면 증류는 필요 없다.

## 과적합 방지

- 학습 응답은 평가 표본(dev, 360개 응답 2,629문장)과 겹치지 않는다 (코드에서 확인).
- 최적 시점 선택(early stopping)은 학습 응답에서 따로 떼어낸 검증 응답으로만 한다. dev는 최종 측정에만 쓴다.
- test 분할은 열지 않는다.
- 과제 유형별, 응답 생성 모델별로 따로 성능을 확인해 특정 부분집합에만 맞춰졌는지 본다.

## 사전 정의 기준 (결과 보기 전에 확정)

- **채택**: dev AUROC 0.78 이상 (MiniCheck 0.814 대비 손실 0.035 이내) **그리고** 문장당 지연 100ms 이하
  (한 문장씩 처리, 근거 검색 포함)
- **도전 목표**: 같은 정확도에서 30ms 이하
- **증류 효과**: B 또는 C가 A보다 AUROC 0.01 이상 높으면 선생 신호가 유효하다고 본다
- **비용 대비 효과**: D가 B·C의 최고치 이상이면 MiniCheck 레이블링 없이 정답만으로 충분하다고 본다

> 런타임 유형 **GPU(T4)**. MiniCheck 레이블링이 약 30분, 학생 학습 8회가 합쳐서 약 30~40분 걸린다.
> 레이블링 결과는 조각마다 저장되므로 세션이 끊겨도 이어서 돌릴 수 있다.
> S1에서 받은 `minicheck_dev_scores.csv`를 작업 폴더(또는 드라이브의 `OUT_DIR`)에 올려두면 dev 레이블링을 건너뛴다.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install pysbd sentencepiece scikit-learn pandas matplotlib nltk sentence-transformers
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'already cloned'
!pip -q install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main" 2>&1 | tail -2

import nltk
for pkg in ['punkt', 'punkt_tab']:
    try: nltk.download(pkg, quiet=True)
    except Exception as e: print(pkg, 'download 실패:', e)

import json, time, random, os, gc, copy
import numpy as np, pandas as pd, torch, pysbd
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEVICE)
""")

code(r"""
# 구글 드라이브에 저장하려면 True (세션이 끊겨도 레이블링 결과가 남음)
USE_DRIVE = False
OUT_DIR = '.'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT_DIR = '/content/drive/MyDrive/selfproject'
os.makedirs(OUT_DIR, exist_ok=True)
print('저장 위치:', OUT_DIR)
""")

md("## 2. 데이터 구성 — dev는 S1과 동일, 학습 응답은 dev와 분리")
code(r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources   = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
for r in responses:
    r['task_type'] = sources[r['source_id']]['task_type']

def build_reference(src):
    si, tt = src['source_info'], src['task_type']
    if tt == 'QA':
        return f"Question: {si.get('question','')}\n{si.get('passages','')}"
    if tt == 'Summary':
        return si if isinstance(si, str) else json.dumps(si, ensure_ascii=False)
    parts = []
    for k, v in (si.items() if isinstance(si, dict) else []):
        parts.append(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}")
    return '\n'.join(parts)

TASKS = ['QA', 'Summary', 'Data2txt']
train_pool = [r for r in responses if r['split'] == 'train']

# dev: S1과 같은 시드·순서
random.seed(SEED)
dev = []
for tt in TASKS:
    cand = [r for r in train_pool if r['task_type'] == tt]
    dev += random.sample(cand, 120)
dev_ids = {r['id'] for r in dev}
rest = [r for r in train_pool if r['id'] not in dev_ids]

# teacher 세트: MiniCheck 레이블을 붙일 응답 (과제 유형별 균등)
N_TEACHER_PER_TASK = 200          # 시간이 부족하면 120으로 낮출 것
rng = random.Random(11)
teacher_resp = []
for tt in TASKS:
    cand = [r for r in rest if r['task_type'] == tt]
    teacher_resp += rng.sample(cand, N_TEACHER_PER_TASK)

# teacher 세트의 15%를 검증용으로 분리 (응답 단위로 나눠 문장 누수 방지)
rng.shuffle(teacher_resp)
n_val = int(len(teacher_resp) * 0.15)
val_resp, teacher_train_resp = teacher_resp[:n_val], teacher_resp[n_val:]
used_ids = dev_ids | {r['id'] for r in teacher_resp}

# gold-large: 정답 레이블만 쓰는 대규모 학습 세트 (teacher 학습 응답 포함, dev·검증 제외)
N_GOLD_LARGE = 3000
gl_extra = rng.sample([r for r in rest if r['id'] not in used_ids],
                      N_GOLD_LARGE - len(teacher_train_resp))
gold_large_resp = teacher_train_resp + gl_extra

val_ids = {r['id'] for r in val_resp}
assert not (dev_ids & {r['id'] for r in teacher_resp}), 'dev와 teacher 세트 겹침'
assert not (dev_ids & {r['id'] for r in gold_large_resp}), 'dev와 gold-large 겹침'
assert not (val_ids & {r['id'] for r in gold_large_resp}), '검증과 gold-large 겹침'
print(f'dev {len(dev)} | teacher 학습 {len(teacher_train_resp)} | 검증 {len(val_resp)} | '
      f'gold-large {len(gold_large_resp)} 응답 (겹침 없음 확인)')
""")

code(r"""
seg = pysbd.Segmenter(language='en', clean=False, char_span=True)

def split_sentences(t):
    return [(s.start, s.end, t[s.start:s.end]) for s in seg.segment(t) if s.sent.strip()]

def to_sentence_df(resp_list):
    recs = []
    for r in resp_list:
        labels = r.get('labels') or []
        for i, (a, b, s) in enumerate(split_sentences(r['response'])):
            recs.append({'resp_id': str(r['id']), 'source_id': r['source_id'], 'task_type': r['task_type'],
                         'gen_model': r['model'], 'sent_idx': i, 'sentence': s,
                         'y': int(any(min(b, l['end']) > max(a, l['start']) for l in labels))})
    return pd.DataFrame(recs)

dev_df = to_sentence_df(dev)
tr_df  = to_sentence_df(teacher_train_resp)
val_df = to_sentence_df(val_resp)
gl_df  = to_sentence_df(gold_large_resp)
for n, d in [('dev', dev_df), ('teacher 학습', tr_df), ('검증', val_df), ('gold-large', gl_df)]:
    print(f'{n:12} 문장 {len(d):6,}  양성 {d.y.mean()*100:4.1f}%')
if len(dev_df) != 2629:
    print('주의: S1의 2,629문장과 다름 — dev 표본 재현 확인 필요')
""")

md(r"""## 3. 선생 레이블 — MiniCheck 점수

teacher 학습·검증 문장과 dev 문장에 MiniCheck 점수를 붙인다. 500문장마다 저장하므로
세션이 끊겨도 이 셀을 다시 실행하면 이어서 진행한다.""")
code(r"""
MC_MS = None

def minicheck_label(df, path, chunk=500):
    global MC_MS
    df = df.copy()
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['resp_id','sent_idx','mc_score'])
    done['resp_id'] = done.resp_id.astype(str); done['sent_idx'] = done.sent_idx.astype(int)
    key = set(zip(done.resp_id, done.sent_idx))
    todo = df[[ (a, b) not in key for a, b in zip(df.resp_id, df.sent_idx)]]
    if len(todo):
        from minicheck.minicheck import MiniCheck
        mc = MiniCheck(model_name='flan-t5-large', enable_prefix_caching=False)
        print(f'{path}: {len(done)}개 완료, {len(todo)}개 남음')
        for i in range(0, len(todo), chunk):
            part = todo.iloc[i:i+chunk]
            docs = [build_reference(sources[s]) for s in part.source_id]
            t0 = time.perf_counter()
            _, prob, _, _ = mc.score(docs=docs, claims=part.sentence.tolist())
            MC_MS = (time.perf_counter() - t0) * 1000 / len(part)
            new = pd.DataFrame({'resp_id': part.resp_id.values, 'sent_idx': part.sent_idx.values,
                                'mc_score': 1.0 - np.asarray(prob)})     # 클수록 환각
            done = pd.concat([done, new], ignore_index=True)
            done.to_csv(path, index=False)
            print(f'  {len(done)}/{len(df)} 저장 ({MC_MS:.0f} ms/문장)')
        del mc; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    done = done.drop_duplicates(['resp_id', 'sent_idx'])
    return df.merge(done[['resp_id', 'sent_idx', 'mc_score']], on=['resp_id', 'sent_idx'], how='left')

# dev: S1에서 저장한 파일이 있으면 재사용
dev_path = os.path.join(OUT_DIR, 'minicheck_dev_scores.csv')
if os.path.exists(dev_path):
    saved = pd.read_csv(dev_path)
    saved['resp_id'] = saved.resp_id.astype(str); saved['sent_idx'] = saved.sent_idx.astype(int)
    if 'mc_ms_per_sentence' in saved.columns:
        MC_MS_DEV = float(saved.mc_ms_per_sentence.iloc[0])
    dev_df = dev_df.merge(saved[['resp_id', 'sent_idx', 'mc_score']], on=['resp_id', 'sent_idx'], how='left')
    print('S1 dev MiniCheck 점수 재사용')
else:
    dev_df = minicheck_label(dev_df, dev_path)
    MC_MS_DEV = MC_MS

tr_df  = minicheck_label(tr_df,  os.path.join(OUT_DIR, 'minicheck_teacher_train.csv'))
val_df = minicheck_label(val_df, os.path.join(OUT_DIR, 'minicheck_teacher_val.csv'))
MC_MS_REF = MC_MS_DEV if 'MC_MS_DEV' in globals() and MC_MS_DEV else (MC_MS or 399.0)

for n, d in [('dev', dev_df), ('teacher 학습', tr_df), ('검증', val_df)]:
    assert d.mc_score.notna().all(), f'{n}에 MiniCheck 점수 누락'
MC_AUROC = roc_auc_score(dev_df.y, dev_df.mc_score)
print(f'선생(MiniCheck) dev AUROC {MC_AUROC:.3f} | {MC_MS_REF:.0f} ms/문장')
print(f'선생 점수와 정답의 일치 (teacher 학습 세트 AUROC): {roc_auc_score(tr_df.y, tr_df.mc_score):.3f}')
""")

md(r"""## 4. 근거 검색 — 조각을 미리 임베딩, 문장마다 top-k만 선택

근거 문서는 응답 생성 전에 이미 주어지므로 조각 임베딩은 미리 계산한다(서비스에서도 동일).
문장이 들어왔을 때는 문장 임베딩 1회와 내적 1회만 필요하다.""")
code(r"""
from sentence_transformers import SentenceTransformer
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)

CHUNK_WORDS, K_TRAIN, K_MAX = 40, 4, 8

def chunk_text(text, n=CHUNK_WORDS):
    w = text.split()
    return [' '.join(w[i:i+n]) for i in range(0, max(len(w), 1), n)] or [text]

ref_cache = {}
def get_chunks(sid):
    if sid not in ref_cache:
        ch = chunk_text(build_reference(sources[sid]))
        ref_cache[sid] = (ch, retriever.encode(ch, convert_to_tensor=True, normalize_embeddings=True,
                                               batch_size=128, show_progress_bar=False))
    return ref_cache[sid]

def attach_retrieval(df):
    for sid in df.source_id.unique(): get_chunks(sid)
    embs = retriever.encode(df.sentence.tolist(), convert_to_tensor=True, normalize_embeddings=True,
                            batch_size=128, show_progress_bar=False)
    ranks = []
    for sid, e in zip(df.source_id, embs):
        _, ce = get_chunks(sid)
        k = min(K_MAX, ce.shape[0])
        ranks.append(torch.topk(ce @ e, k).indices.tolist())
    df = df.copy(); df['ranked'] = ranks
    return df

def build_context(sid, ranked, k):
    ch, _ = get_chunks(sid)
    keep = sorted(ranked[:k])                  # 원문 순서 유지
    return ' ... '.join(ch[i] for i in keep)

t0 = time.perf_counter()
dev_df, tr_df, val_df, gl_df = [attach_retrieval(d) for d in (dev_df, tr_df, val_df, gl_df)]
print(f'근거 검색 준비 {time.perf_counter()-t0:.0f}초 | 근거 {len(ref_cache)}건')
ex = dev_df.iloc[0]
print('\n예시 문장:', ex.sentence[:150])
print('선택된 근거:', build_context(ex.source_id, ex.ranked, K_TRAIN)[:300], '...')
""")

md(r"""## 5. 학생 학습

학습 목표 t(0~1, 클수록 환각)에 대해 BCE 손실로 학습한다. mix 조건은 t = 0.5·정답 + 0.5·선생 점수이며,
BCE는 목표에 대해 선형이므로 두 손실을 반씩 더한 것과 같다. 매 에폭 뒤 검증 세트의 **정답 기준 AUROC**로
가장 좋은 시점을 고른다.""")
code(r"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

STUDENTS = {
    'MiniLM-L6':         'cross-encoder/nli-MiniLM2-L6-H768',
    'DeBERTa-v3-xsmall': 'cross-encoder/nli-deberta-v3-xsmall',
}
MAX_LEN, BS = 320, 32

def load_student(ckpt):
    tok = AutoTokenizer.from_pretrained(ckpt)
    m = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=1, ignore_mismatched_sizes=True)
    return tok, m.to(DEVICE)

def encode(tok, df, k):
    ctx = [build_context(s, r, k) for s, r in zip(df.source_id, df.ranked)]
    return tok(ctx, df.sentence.tolist(), truncation='only_first', max_length=MAX_LEN,
               padding=True, return_tensors='pt')

@torch.no_grad()
def predict(tok, m, df, k=K_TRAIN, bs=128):
    m.eval(); out = []
    for i in range(0, len(df), bs):
        enc = encode(tok, df.iloc[i:i+bs], k).to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
            out.append(m(**enc).logits.squeeze(-1).float().cpu())
    return torch.cat(out).numpy()

def train_student(ckpt, train_df, target, epochs, lr=3e-5):
    tok, m = load_student(ckpt)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_df) + BS - 1) // BS)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.06 * steps)), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    lossf = torch.nn.BCEWithLogitsLoss()
    best, best_state = -1, None
    for ep in range(epochs):
        m.train()
        order = np.random.permutation(len(train_df))
        for i in range(0, len(order), BS):
            b = train_df.iloc[order[i:i+BS]]
            enc = encode(tok, b, K_TRAIN).to(DEVICE)
            t = torch.tensor(target[order[i:i+BS]], dtype=torch.float32, device=DEVICE)
            with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
                logit = m(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), t)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        va = roc_auc_score(val_df.y, predict(tok, m, val_df))
        print(f'    epoch {ep+1}/{epochs}  검증 AUROC {va:.3f}')
        if va > best:
            best, best_state = va, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return tok, m, best
""")

code(r"""
CONDITIONS = {
    'A. gold':       (tr_df, lambda d: d.y.values.astype(float), 4),
    'B. teacher':    (tr_df, lambda d: d.mc_score.values, 4),
    'C. mix':        (tr_df, lambda d: 0.5 * d.y.values + 0.5 * d.mc_score.values, 4),
    'D. gold-large': (gl_df, lambda d: d.y.values.astype(float), 2),
}

results, dev_preds, trained = [], {}, {}
for sname, ckpt in STUDENTS.items():
    for cname, (d, tgt, ep) in CONDITIONS.items():
        print(f'[{sname} | {cname}] 학습 문장 {len(d):,}')
        t0 = time.perf_counter()
        tok, m, va = train_student(ckpt, d, tgt(d), ep)
        p = predict(tok, m, dev_df)
        dev_preds[(sname, cname)] = p
        results.append({'student': sname, 'condition': cname, 'n_train': len(d), 'val_AUROC': va,
                        'dev_AUROC': roc_auc_score(dev_df.y, p),
                        'dev_AUPRC': average_precision_score(dev_df.y, p),
                        'train_min': (time.perf_counter() - t0) / 60})
        print(f'    → dev AUROC {results[-1]["dev_AUROC"]:.3f} | AUPRC {results[-1]["dev_AUPRC"]:.3f}')
        trained[(sname, cname)] = (tok, m.cpu())
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

res = pd.DataFrame(results)
res.round(3)
""")

md(r"""## 6. 지연 측정 — 서비스와 같은 조건

문장이 하나씩 도착한다고 가정하고 배치 크기 1로 잰다. 문장 임베딩, 근거 조각 검색, 입력 구성,
학생 추론을 모두 포함한다. 같은 구조의 학생은 학습 조건과 무관하게 지연이 같으므로 학생별로 한 번 잰다.""")
code(r"""
@torch.no_grad()
def latency_ms(tok, m, df, k=K_TRAIN, n=200):
    m.eval().to(DEVICE)
    sample = df.sample(n, random_state=SEED)
    def one(row):
        e = retriever.encode([row.sentence], convert_to_tensor=True, normalize_embeddings=True,
                             show_progress_bar=False)[0]
        ch, ce = get_chunks(row.source_id)
        ranked = torch.topk(ce @ e, min(K_MAX, ce.shape[0])).indices.tolist()
        enc = tok([build_context(row.source_id, ranked, k)], [row.sentence], truncation='only_first',
                  max_length=MAX_LEN, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
            return m(**enc).logits
    for row in sample.head(10).itertuples(): one(row)          # 워밍업
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for row in sample.itertuples(): one(row)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n

LAT = {}
for sname in STUDENTS:
    tok, m = trained[(sname, 'A. gold')]
    LAT[sname] = latency_ms(tok, m, dev_df)
    m.cpu()
    print(f'{sname:18} {LAT[sname]:6.1f} ms/문장  (MiniCheck {MC_MS_REF:.0f} ms 대비 {MC_MS_REF/LAT[sname]:.1f}배 빠름)')
res['ms_per_sentence'] = res.student.map(LAT)
""")

md("## 7. 사전 기준 판정")
code(r"""
AUROC_MIN, MS_MAX, MS_STRETCH, DISTILL_GAIN = 0.78, 100.0, 30.0, 0.01

print(f'선생 MiniCheck: dev AUROC {MC_AUROC:.3f} | {MC_MS_REF:.0f} ms/문장\n')
for r in res.itertuples():
    ok = r.dev_AUROC >= AUROC_MIN and r.ms_per_sentence <= MS_MAX
    tag = '채택 가능' + (' (도전 목표 달성)' if ok and r.ms_per_sentence <= MS_STRETCH else '') if ok else '미달'
    print(f'{r.student:18} {r.condition:14} AUROC {r.dev_AUROC:.3f} ({r.dev_AUROC - MC_AUROC:+.3f})  '
          f'{r.ms_per_sentence:6.1f}ms  → {tag}')

print('\n=== 증류 효과 (같은 문장, 학습 목표만 다름) ===')
for sname, g in res.groupby('student'):
    a = g.set_index('condition').dev_AUROC
    gain = max(a['B. teacher'], a['C. mix']) - a['A. gold']
    print(f'{sname:18} gold {a["A. gold"]:.3f} | teacher {a["B. teacher"]:.3f} | mix {a["C. mix"]:.3f} '
          f'→ 선생 신호 {"유효" if gain >= DISTILL_GAIN else "효과 미미"} ({gain:+.3f})')

print('\n=== 비용 대비 효과 (MiniCheck 레이블 vs 정답 6배) ===')
for sname, g in res.groupby('student'):
    a = g.set_index('condition').dev_AUROC
    bc = max(a['B. teacher'], a['C. mix'])
    verdict = '정답만으로 충분 — MiniCheck 레이블링 불필요' if a['D. gold-large'] >= bc else 'MiniCheck 레이블이 더 효율적'
    print(f'{sname:18} gold-large {a["D. gold-large"]:.3f} vs teacher/mix 최고 {bc:.3f} → {verdict}')

adopt = res[(res.dev_AUROC >= AUROC_MIN) & (res.ms_per_sentence <= MS_MAX)]
BEST = (adopt if len(adopt) else res).sort_values('dev_AUROC', ascending=False).iloc[0]
print(f'\n최선 조합: {BEST.student} | {BEST.condition} (AUROC {BEST.dev_AUROC:.3f}, {BEST.ms_per_sentence:.1f}ms)')
if not len(adopt):
    print('결론: 채택 기준을 만족하는 학생 없음 — 가장 가까운 조합을 기준으로 개선 방향을 검토')
""")

md(r"""## 8. 근거를 몇 조각 읽을지 — 적응형 조절 변수 확인

최선 학생에게 근거 조각 수 k를 바꿔 넣어본다(재학습 없음). k가 정확도와 지연을 함께 움직인다면,
위험도에 따라 k를 달리 쓰는 것이 미들웨어의 조절 수단이 된다.""")
code(r"""
tok, m = trained[(BEST.student, BEST.condition)]
m.to(DEVICE)
krows = []
for k in [1, 2, 4, 8]:
    p = predict(tok, m, dev_df, k=k)
    krows.append({'k': k, 'dev_AUROC': roc_auc_score(dev_df.y, p),
                  'dev_AUPRC': average_precision_score(dev_df.y, p),
                  'ms_per_sentence': latency_ms(tok, m, dev_df, k=k, n=100)})
kres = pd.DataFrame(krows)
print(kres.round(3).to_string(index=False))
""")

md(r"""## 9. 학생을 게이트키퍼로 쓰면 — S1과 같은 기준으로 비교

S1에서 규칙 기반 게이트키퍼는 환각 95%를 MiniCheck까지 보내려면 호출 82%가 필요했다.
학생을 게이트키퍼 자리에 두면 이 비율이 얼마나 줄어드는지 본다. 학생이 단독으로 채택되지 못하더라도
이 구조로 쓸 수 있는지를 확인하는 보조 분석이다.""")
code(r"""
y, mc, s = dev_df.y.values, dev_df.mc_score.values, dev_preds[(BEST.student, BEST.condition)]
hit = None
for r in np.linspace(0.01, 1.0, 100):
    route = s >= (np.quantile(s, 1 - r) if r < 1 else -np.inf)
    if route[y == 1].mean() >= 0.95:
        hit = (r, roc_auc_score(y, np.where(route, mc, -1.0)))
        break
lat = LAT[BEST.student] + hit[0] * MC_MS_REF
print(f'환각 95% 커버리지에 필요한 MiniCheck 호출: {hit[0]:.0%} (S1 규칙 게이트키퍼 82%)')
print(f'이때 시스템 AUROC {hit[1]:.3f}, 예상 평균 지연 {lat:.0f} ms (전수 {MC_MS_REF:.0f} ms 대비 {1 - lat/MC_MS_REF:.0%} 감소)')
""")

md("## 10. 과적합 점검 — 부분집합별 성능")
code(r"""
a = dev_df.assign(student=dev_preds[(BEST.student, BEST.condition)])
rows = []
for col in ['task_type', 'gen_model']:
    for v, g in a.groupby(col):
        if g.y.nunique() < 2: continue
        rows.append({'group': f'{col}={v}', 'n': len(g), 'pos': g.y.mean(),
                     'student_AUROC': roc_auc_score(g.y, g.student),
                     'MiniCheck_AUROC': roc_auc_score(g.y, g.mc_score)})
sub = pd.DataFrame(rows)
sub['gap'] = sub.student_AUROC - sub.MiniCheck_AUROC
print(sub.round(3).to_string(index=False))
print('\n학생이 MiniCheck보다 0.05 이상 뒤처지는 부분집합:', ', '.join(sub[sub.gap < -0.05].group) or '없음')
""")

code(r"""
res.to_csv(os.path.join(OUT_DIR, 's2_student_results.csv'), index=False)
kres.to_csv(os.path.join(OUT_DIR, 's2_k_ablation.csv'), index=False)
sub.to_csv(os.path.join(OUT_DIR, 's2_subgroups.csv'), index=False)
out = dev_df[['resp_id', 'sent_idx', 'task_type', 'gen_model', 'y', 'mc_score']].copy()
for (sn, cn), p in dev_preds.items(): out[f'{sn}|{cn}'] = p
out.to_csv(os.path.join(OUT_DIR, 's2_dev_predictions.csv'), index=False)
save_dir = os.path.join(OUT_DIR, 'student_best')
tok.save_pretrained(save_dir); m.save_pretrained(save_dir)
print('저장 완료:', OUT_DIR, '| 최선 학생 모델:', save_dir)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 's2_distillation.ipynb'), 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
