import os
import json
cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.strip("\n")})

md(r"""# S2-b — 학생 판정기의 일반화 확인과 가변 k 학습

## 왜 하나

S2에서 MiniLM-L6 학생(정답 레이블 22,082문장, 근거 조각 k=4)이 RAGTruth dev에서 AUROC 0.864,
문장당 14.9ms로 MiniCheck(0.814, 443ms)를 앞섰다. 하지만 두 가지가 확인되지 않았다.

1. **일반화**: 학생은 RAGTruth로 학습·평가했고 MiniCheck는 RAGTruth를 본 적이 없다.
   다른 데이터셋에서도 학생이 버티는지 봐야 과적합 우려에 답할 수 있다.
   → 3주차에 미뤄둔 **HaluEval 길이 통제 재평가**에 학생을 함께 넣는다.
2. **k 곡선의 꺾임**: k=1→2→4에서 0.781→0.834→0.864로 오르다가 k=8에서 0.823으로 떨어졌다.
   k=4로만 학습했고 입력 길이 제한(320토큰)에서 뒤쪽 조각이 잘렸기 때문으로 추정된다.
   k가 미들웨어의 조절 변수가 되려면 곡선이 끝까지 올라가야 한다.
   → k를 {1,2,4,8}에서 무작위로 바꿔 학습하고 입력 길이를 512토큰으로 늘린다.

## 두 모델

| 모델 | 학습 k | 입력 길이 | 비고 |
|---|---|---|---|
| FIXED | 4 고정 | 320 | S2 최선 조합(D. gold-large) 재현. `student_best`가 있으면 불러옴 |
| VARK  | 배치마다 {1,2,4,8} 중 무작위 | 512 | 이번에 새로 학습 |

학습 데이터·분할·시드는 S2와 동일하다(dev·검증 응답과 겹치지 않음을 코드에서 확인).

## HaluEval 평가 방법

- 정상/환각 응답 쌍 중 길이 비율이 1/1.5~1.5배 안에 드는 쌍만 남기고, 다시 도메인마다
  '환각이 더 긴 쌍'과 '정상이 더 긴 쌍'을 같은 수로 뽑아 길이 신호를 제거한다.
  (비율 필터만으로는 환각 쪽이 일관되게 조금 더 길면 길이만으로 구분이 가능하다.)
- 미들웨어와 같은 방식으로 **응답을 문장으로 나눠 문장마다 판정하고, 응답 점수는 문장 점수의 최댓값**으로 한다.
  MiniCheck와 학생 모두 같은 방식이다.
- HaluEval의 QA·Summary는 RAGTruth와 과제가 비슷하고, Dialogue는 RAGTruth에 없는 과제다.

## 사전 정의 기준 (결과 보기 전에 확정)

- **길이 통제 성공**: 도메인별 길이 기준선 AUROC < 0.60. 미달 도메인은 해석에서 제외한다.
- **일반화 유지**: 길이 통제 성공 도메인에서 학생 AUROC가 MiniCheck − 0.05 이상.
  더 크게 뒤처지면 학생이 RAGTruth에 맞춰졌다고 보고, 학습 데이터 구성 보완을 다음 작업으로 둔다.
  단, MiniCheck 자체가 0.60 미만인 도메인은 비교 기준이 없으므로 판정에서 제외한다.
- **가변 k 채택**: VARK가 ① dev에서 k=4 AUROC가 FIXED − 0.01 이상이고
  ② k=1→2→4→8에서 AUROC가 줄지 않으면(허용 오차 0.005) 적응형 조절 변수로 채택한다.

> 런타임 **GPU(T4)**. 학생 학습 2회 약 20~30분, HaluEval MiniCheck 판정 약 20~30분.
> MiniCheck 판정 결과는 조각마다 저장되므로 끊겨도 다시 실행하면 이어서 진행한다.
> S2에서 저장한 `student_best/` 폴더가 `OUT_DIR`에 있으면 FIXED 학습을 건너뛴다.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install pysbd sentencepiece scikit-learn pandas matplotlib nltk sentence-transformers
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'
!git clone --depth 1 https://github.com/RUCAIBox/HaluEval.git 2>/dev/null || echo 'HaluEval already cloned'
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

USE_DRIVE = False          # S2의 student_best를 드라이브에 저장해뒀다면 True
OUT_DIR = '.'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT_DIR = '/content/drive/MyDrive/selfproject'
os.makedirs(OUT_DIR, exist_ok=True)
print('저장 위치:', OUT_DIR)
""")

md("## 2. RAGTruth 분할 — S2와 동일한 시드·순서로 재현")
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

random.seed(SEED)
dev = []
for tt in TASKS:
    cand = [r for r in train_pool if r['task_type'] == tt]
    dev += random.sample(cand, 120)
dev_ids = {r['id'] for r in dev}
rest = [r for r in train_pool if r['id'] not in dev_ids]

N_TEACHER_PER_TASK = 200
rng = random.Random(11)
teacher_resp = []
for tt in TASKS:
    cand = [r for r in rest if r['task_type'] == tt]
    teacher_resp += rng.sample(cand, N_TEACHER_PER_TASK)
rng.shuffle(teacher_resp)
n_val = int(len(teacher_resp) * 0.15)
val_resp, teacher_train_resp = teacher_resp[:n_val], teacher_resp[n_val:]
used_ids = dev_ids | {r['id'] for r in teacher_resp}

N_GOLD_LARGE = 3000
gl_extra = rng.sample([r for r in rest if r['id'] not in used_ids],
                      N_GOLD_LARGE - len(teacher_train_resp))
gold_large_resp = teacher_train_resp + gl_extra

val_ids = {r['id'] for r in val_resp}
assert not (dev_ids & {r['id'] for r in gold_large_resp}), 'dev와 gold-large 겹침'
assert not (val_ids & {r['id'] for r in gold_large_resp}), '검증과 gold-large 겹침'
assert not (dev_ids & val_ids), 'dev와 검증 겹침'

seg = pysbd.Segmenter(language='en', clean=False, char_span=True)
def split_sentences(t):
    out = [(s.start, s.end, t[s.start:s.end]) for s in seg.segment(t) if s.sent.strip()]
    return out or [(0, len(t), t)]

REFS = {}          # ref_key -> 근거 원문 (RAGTruth·HaluEval 공용)

def to_sentence_df(resp_list):
    recs = []
    for r in resp_list:
        key = f"rt:{r['source_id']}"
        REFS.setdefault(key, build_reference(sources[r['source_id']]))
        labels = r.get('labels') or []
        for i, (a, b, s) in enumerate(split_sentences(r['response'])):
            recs.append({'resp_id': str(r['id']), 'ref_key': key, 'task_type': r['task_type'],
                         'gen_model': r['model'], 'sent_idx': i, 'sentence': s,
                         'y': int(any(min(b, l['end']) > max(a, l['start']) for l in labels))})
    return pd.DataFrame(recs)

dev_df = to_sentence_df(dev)
val_df = to_sentence_df(val_resp)
gl_df  = to_sentence_df(gold_large_resp)
for n, d in [('dev', dev_df), ('검증', val_df), ('gold-large', gl_df)]:
    print(f'{n:10} 문장 {len(d):6,}  양성 {d.y.mean()*100:4.1f}%')
if (len(dev_df), len(gl_df)) != (2629, 22082):
    print('주의: S2와 문장 수가 다름 (S2: dev 2,629 / gold-large 22,082) — 분할 재현 확인 필요')
""")

md(r"""## 3. HaluEval — 길이 통제 쌍 선별 후 문장 단위로 전개

필터링 후 도메인당 최대 300쌍(방향별 150쌍)만 사용해 MiniCheck 판정 시간을 제한한다.
원본은 도메인당 1,500쌍에서 시작한다.""")
code(r"""
HB = 'HaluEval/data'
def load_raw_pairs(path, ref_key, right_key, hallu_key, n):
    lines = open(path).readlines()
    random.Random(SEED).shuffle(lines)
    rows = []
    for line in lines[:n]:
        d = json.loads(line)
        ref = d[ref_key] if isinstance(ref_key, str) else ref_key(d)
        rows.append({'reference': ref, 'right': d[right_key], 'hallu': d[hallu_key]})
    return pd.DataFrame(rows)

N_PER_DOMAIN, R, CAP = 1500, 1.5, 300
parts = [
    load_raw_pairs(f'{HB}/qa_data.json', 'knowledge', 'right_answer', 'hallucinated_answer', N_PER_DOMAIN).assign(domain='QA'),
    load_raw_pairs(f'{HB}/summarization_data.json', 'document', 'right_summary', 'hallucinated_summary', N_PER_DOMAIN).assign(domain='Summary'),
    load_raw_pairs(f'{HB}/dialogue_data.json', lambda d: d['knowledge'] + '\n' + d['dialogue_history'],
                   'right_response', 'hallucinated_response', N_PER_DOMAIN).assign(domain='Dialogue'),
]
pairs = pd.concat(parts, ignore_index=True)
pairs = pairs[(pairs.right.str.strip() != '') & (pairs.hallu.str.strip() != '')]
pairs['len_ratio'] = pairs.hallu.str.len() / pairs.right.str.len().clip(lower=1)
# ① 길이 비율 필터: 두 응답 길이가 1/R~R배 안
kept = pairs[(pairs.len_ratio >= 1/R) & (pairs.len_ratio <= R)].copy()
# ② 방향 균형: 비율 안에 들어도 환각 쪽이 '항상 조금 더 길면' 길이만으로 구분된다.
#    도메인마다 '환각이 더 긴 쌍'과 '정상이 더 긴 쌍'을 같은 수로 뽑아 길이 방향 신호를 없앤다.
kept['hallu_longer'] = kept.len_ratio > 1
bal = []
for d, g in kept.groupby('domain'):
    a, b = g[g.hallu_longer], g[~g.hallu_longer]
    n = min(len(a), len(b), CAP // 2)
    print(f'  {d:9} 비율 필터 통과 {len(g):4}쌍 (환각이 더 김 {len(a)}, 정상이 더 김 {len(b)}) → 균형 {n}+{n}쌍')
    bal += [a.head(n), b.head(n)]
kept = pd.concat(bal).reset_index(drop=True)
print('길이 통제 후 쌍 수:', kept.groupby('domain').size().to_dict())

items, sents = [], []
for pi, r in enumerate(kept.itertuples()):
    key = f'he:{pi}'
    REFS[key] = r.reference
    for y, text in [(0, r.right), (1, r.hallu)]:
        iid = f'{pi}-{y}'
        items.append({'item_id': iid, 'domain': r.domain, 'y': y, 'claim_len': len(text)})
        for si, (_, _, s) in enumerate(split_sentences(text)):
            sents.append({'item_id': iid, 'ref_key': key, 'domain': r.domain, 'sent_idx': si, 'sentence': s})
he_items = pd.DataFrame(items)
he_sent  = pd.DataFrame(sents)
print(f'HaluEval 응답 {len(he_items):,}개 → 문장 {len(he_sent):,}개')

print('\n=== 길이 통제 확인 (길이 기준선 AUROC < 0.60 목표) ===')
LEN_OK = {}
for d, g in he_items.groupby('domain'):
    a = roc_auc_score(g.y, g.claim_len)
    LEN_OK[d] = a < 0.60
    print(f'  {d:9} n={len(g):4}  길이 기준선 AUROC {a:.3f} → {"통제 성공" if LEN_OK[d] else "길이 신호 잔존 — 해석 제외"}')
""")

md(r"""## 4. 근거 검색 — S2와 동일 (40단어 조각, 임베딩 top-k)""")
code(r"""
from sentence_transformers import SentenceTransformer
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)

CHUNK_WORDS, K_MAX = 40, 8

def chunk_text(text, n=CHUNK_WORDS):
    w = text.split()
    return [' '.join(w[i:i+n]) for i in range(0, max(len(w), 1), n)] or [text]

chunk_cache = {}
def get_chunks(key):
    if key not in chunk_cache:
        ch = chunk_text(REFS[key])
        chunk_cache[key] = (ch, retriever.encode(ch, convert_to_tensor=True, normalize_embeddings=True,
                                                 batch_size=128, show_progress_bar=False))
    return chunk_cache[key]

def attach_retrieval(df):
    for key in df.ref_key.unique(): get_chunks(key)
    embs = retriever.encode(df.sentence.tolist(), convert_to_tensor=True, normalize_embeddings=True,
                            batch_size=128, show_progress_bar=False)
    ranks = []
    for key, e in zip(df.ref_key, embs):
        _, ce = get_chunks(key)
        ranks.append(torch.topk(ce @ e, min(K_MAX, ce.shape[0])).indices.tolist())
    df = df.copy(); df['ranked'] = ranks
    return df

def build_context(key, ranked, k):
    ch, _ = get_chunks(key)
    return ' ... '.join(ch[i] for i in sorted(ranked[:k]))

t0 = time.perf_counter()
dev_df, val_df, gl_df, he_sent = [attach_retrieval(d) for d in (dev_df, val_df, gl_df, he_sent)]
print(f'근거 검색 준비 {time.perf_counter()-t0:.0f}초 | 근거 {len(chunk_cache)}건')
""")

md(r"""## 5. 학생 학습 함수 — k를 배치마다 바꿀 수 있게 확장

`k_choices=[4]`이면 S2와 같은 고정 k 학습, `[1,2,4,8]`이면 가변 k 학습이다.
최적 시점은 검증 세트의 정답 기준 AUROC로 고르며, 가변 k 모델은 네 k의 평균으로 고른다.""")
code(r"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

CKPT = 'cross-encoder/nli-MiniLM2-L6-H768'
BS = 32

def load_student(ckpt):
    tok = AutoTokenizer.from_pretrained(ckpt)
    m = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=1, ignore_mismatched_sizes=True)
    return tok, m.to(DEVICE)

def encode(tok, df, k, max_len):
    ctx = [build_context(key, r, k) for key, r in zip(df.ref_key, df.ranked)]
    return tok(ctx, df.sentence.tolist(), truncation='only_first', max_length=max_len,
               padding=True, return_tensors='pt')

@torch.no_grad()
def predict(tok, m, df, k, max_len, bs=128):
    m.eval(); out = []
    for i in range(0, len(df), bs):
        enc = encode(tok, df.iloc[i:i+bs], k, max_len).to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
            out.append(m(**enc).logits.squeeze(-1).float().cpu())
    return torch.cat(out).numpy()

def train_student(train_df, epochs, k_choices, max_len, lr=3e-5):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tok, m = load_student(CKPT)
    target = train_df.y.values.astype(float)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_df) + BS - 1) // BS)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.06 * steps)), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    lossf = torch.nn.BCEWithLogitsLoss()
    krng = random.Random(SEED)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train()
        order = np.random.permutation(len(train_df))
        for i in range(0, len(order), BS):
            b = train_df.iloc[order[i:i+BS]]
            enc = encode(tok, b, krng.choice(k_choices), max_len).to(DEVICE)
            t = torch.tensor(target[order[i:i+BS]], dtype=torch.float32, device=DEVICE)
            with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
                logit = m(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), t)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        va = np.mean([roc_auc_score(val_df.y, predict(tok, m, val_df, k, max_len)) for k in k_choices])
        print(f'    epoch {ep+1}/{epochs}  검증 AUROC {va:.3f}')
        if va > best:
            best, best_state = va, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return tok, m, best
""")

md("## 6. 두 모델 준비 — FIXED(S2 재현), VARK(가변 k)")
code(r"""
MODELS = {}      # name -> (tok, model, max_len)

best_dir = os.path.join(OUT_DIR, 'student_best')
if os.path.isdir(best_dir):
    tok = AutoTokenizer.from_pretrained(best_dir)
    m = AutoModelForSequenceClassification.from_pretrained(best_dir).to(DEVICE)
    print('FIXED: S2의 student_best 불러옴')
else:
    print('FIXED: student_best 없음 → S2 D. gold-large 조건으로 재학습 (k=4, 320토큰, 2에폭)')
    tok, m, _ = train_student(gl_df, epochs=2, k_choices=[4], max_len=320)
MODELS['FIXED'] = (tok, m, 320)

print('VARK: 가변 k 학습 (k∈{1,2,4,8}, 512토큰, 2에폭)')
tok, m, _ = train_student(gl_df, epochs=2, k_choices=[1, 2, 4, 8], max_len=512)
MODELS['VARK'] = (tok, m, 512)

for name, (tok, m, ml) in MODELS.items():
    a = roc_auc_score(dev_df.y, predict(tok, m, dev_df, 4, ml))
    print(f'{name:6} dev AUROC (k=4) {a:.3f}' + ('   (S2 기록 0.864)' if name == 'FIXED' else ''))
""")

md(r"""## 7. RAGTruth dev에서 k 곡선 — 가변 k 학습이 꺾임을 해소하는가

k마다 정확도와 지연(배치 1, 검색 포함)을 함께 잰다.""")
code(r"""
@torch.no_grad()
def latency_ms(tok, m, df, k, max_len, n=150):
    m.eval()
    sample = df.sample(n, random_state=SEED)
    def one(row):
        e = retriever.encode([row.sentence], convert_to_tensor=True, normalize_embeddings=True,
                             show_progress_bar=False)[0]
        _, ce = get_chunks(row.ref_key)
        ranked = torch.topk(ce @ e, min(K_MAX, ce.shape[0])).indices.tolist()
        enc = tok([build_context(row.ref_key, ranked, k)], [row.sentence], truncation='only_first',
                  max_length=max_len, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
            return m(**enc).logits
    for row in sample.head(10).itertuples(): one(row)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for row in sample.itertuples(): one(row)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n

KS = [1, 2, 4, 8]
krows = []
for name, (tok, m, ml) in MODELS.items():
    for k in KS:
        p = predict(tok, m, dev_df, k, ml)
        krows.append({'model': name, 'k': k, 'dev_AUROC': roc_auc_score(dev_df.y, p),
                      'dev_AUPRC': average_precision_score(dev_df.y, p),
                      'ms_per_sentence': latency_ms(tok, m, dev_df, k, ml)})
kres = pd.DataFrame(krows)
print(kres.pivot(index='k', columns='model', values=['dev_AUROC', 'ms_per_sentence']).round(3))

f4 = kres.query("model=='FIXED' and k==4").dev_AUROC.item()
v = kres.query("model=='VARK'").set_index('k').dev_AUROC
cond1 = v[4] >= f4 - 0.01
cond2 = all(v[KS[i+1]] >= v[KS[i]] - 0.005 for i in range(len(KS) - 1))
print(f'\n① VARK k=4 {v[4]:.3f} vs FIXED k=4 {f4:.3f} → {"충족" if cond1 else "미달"}')
print(f'② VARK k 곡선 {" → ".join(f"{v[k]:.3f}" for k in KS)} → {"단조 증가" if cond2 else "꺾임 있음"}')
VARK_ADOPT = cond1 and cond2
print('판정:', 'VARK 채택 — k를 적응형 조절 변수로 사용' if VARK_ADOPT else 'VARK 미채택 — k 범위를 1~4로 제한하고 FIXED 유지')
""")

md(r"""## 8. HaluEval — MiniCheck 문장 단위 판정 (이어서 실행 가능)""")
code(r"""
from minicheck.minicheck import MiniCheck

def minicheck_sentences(df, path, chunk=400):
    df = df.drop(columns=['mc_score'], errors='ignore')      # 셀 재실행 대비
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['item_id', 'sent_idx', 'mc_score'])
    done['item_id'] = done.item_id.astype(str); done['sent_idx'] = done.sent_idx.astype(int)
    key = set(zip(done.item_id, done.sent_idx))
    todo = df[[(a, b) not in key for a, b in zip(df.item_id, df.sent_idx)]]
    ms = None
    if len(todo):
        mc = MiniCheck(model_name='flan-t5-large', enable_prefix_caching=False)
        print(f'{len(done)}개 완료, {len(todo)}개 남음')
        for i in range(0, len(todo), chunk):
            part = todo.iloc[i:i+chunk]
            t0 = time.perf_counter()
            _, prob, _, _ = mc.score(docs=[REFS[k] for k in part.ref_key], claims=part.sentence.tolist())
            ms = (time.perf_counter() - t0) * 1000 / len(part)
            new = pd.DataFrame({'item_id': part.item_id.values, 'sent_idx': part.sent_idx.values,
                                'mc_score': 1.0 - np.asarray(prob)})
            done = pd.concat([done, new], ignore_index=True)
            done.to_csv(path, index=False)
            print(f'  {len(done)}/{len(df)} 저장 ({ms:.0f} ms/문장)')
        del mc; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    done = done.drop_duplicates(['item_id', 'sent_idx'])
    return df.merge(done, on=['item_id', 'sent_idx'], how='left'), ms

he_sent, HE_MC_MS = minicheck_sentences(he_sent, os.path.join(OUT_DIR, 'minicheck_halueval_sent.csv'))
assert he_sent.mc_score.notna().all(), 'MiniCheck 점수 누락'
print('MiniCheck 판정 완료', f'({HE_MC_MS:.0f} ms/문장)' if HE_MC_MS else '(저장본 재사용)')
""")

md(r"""## 9. HaluEval 일반화 판정 — 학생 vs MiniCheck

응답 점수 = 문장 점수의 최댓값. 학생은 k=4(학습 조건)와 k=8(VARK의 최대 설정)로 잰다.""")
code(r"""
SCORERS = {'MiniCheck': he_sent.mc_score.values}
for name, (tok, m, ml) in MODELS.items():
    for k in ([4, 8] if name == 'VARK' else [4]):
        SCORERS[f'{name} k={k}'] = predict(tok, m, he_sent, k, ml)

agg = he_sent[['item_id']].copy()
for n, s in SCORERS.items(): agg[n] = s
agg = agg.groupby('item_id').max()
ev = he_items.set_index('item_id').join(agg)

rows = []
for d, g in ev.groupby('domain'):
    row = {'domain': d, 'n': len(g), 'len_baseline': roc_auc_score(g.y, g.claim_len), 'len_ok': LEN_OK[d]}
    for n in SCORERS: row[n] = roc_auc_score(g.y, g[n])
    rows.append(row)
he_res = pd.DataFrame(rows)
print(he_res.round(3).to_string(index=False))

print('\n=== 사전 기준: 학생 ≥ MiniCheck − 0.05 (길이 통제 성공 도메인만) ===')
GEN_OK = {}
for r in he_res.itertuples():
    if not r.len_ok:
        print(f'  {r.domain:9} 길이 신호 잔존 — 판정 제외'); continue
    if r.MiniCheck < 0.60:
        print(f'  {r.domain:9} MiniCheck도 {r.MiniCheck:.3f}로 판별 불가 — 비교 기준이 없어 판정 제외'); continue
    for n in SCORERS:
        if n == 'MiniCheck': continue
        gap = he_res.loc[r.Index, n] - r.MiniCheck
        ok = gap >= -0.05
        GEN_OK[(r.domain, n)] = ok
        print(f'  {r.domain:9} {n:10} {he_res.loc[r.Index, n]:.3f} vs MiniCheck {r.MiniCheck:.3f} '
              f'({gap:+.3f}) → {"일반화 유지" if ok else "RAGTruth 편향 의심"}')

main = 'VARK k=4' if VARK_ADOPT else 'FIXED k=4'
judged = [ok for (d, n), ok in GEN_OK.items() if n == main]
print(f'\n최종 후보({main}) 판정:',
      '판정 가능한 도메인 없음' if not judged else
      ('모든 도메인에서 일반화 유지 → S3 게이트웨이로 진행' if all(judged) else
       '일부 도메인에서 MiniCheck보다 크게 뒤처짐 → 학습 데이터 구성 보완 후 재확인'))
""")

code(r"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, g in kres.groupby('model'):
    axes[0].plot(g.k, g.dev_AUROC, 'o-', label=name)
axes[0].axhline(0.814, color='gray', ls='--', lw=1, label='MiniCheck 0.814')
axes[0].set_xscale('log', base=2); axes[0].set_xticks(KS); axes[0].set_xticklabels(KS)
axes[0].set_xlabel('k (evidence chunks)'); axes[0].set_ylabel('RAGTruth dev AUROC'); axes[0].legend()
axes[0].set_title('k curve')
cols = [c for c in SCORERS]
x = np.arange(len(he_res)); w = 0.8 / len(cols)
for i, c in enumerate(cols):
    axes[1].bar(x + i*w - 0.4 + w/2, he_res[c], w, label=c)
axes[1].plot(x, he_res.len_baseline, 'k_', ms=30, mew=2, label='length baseline')
axes[1].set_xticks(x); axes[1].set_xticklabels(he_res.domain); axes[1].set_ylim(0.4, 1)
axes[1].set_ylabel('HaluEval AUROC (length-controlled)'); axes[1].legend(fontsize=8)
axes[1].set_title('Generalization')
plt.tight_layout(); plt.show()
""")

code(r"""
kres.to_csv(os.path.join(OUT_DIR, 's2b_k_curve.csv'), index=False)
he_res.to_csv(os.path.join(OUT_DIR, 's2b_halueval.csv'), index=False)
ev.reset_index().to_csv(os.path.join(OUT_DIR, 's2b_halueval_items.csv'), index=False)
tok, m, _ = MODELS['VARK']
tok.save_pretrained(os.path.join(OUT_DIR, 'student_vark')); m.save_pretrained(os.path.join(OUT_DIR, 'student_vark'))
tok, m, _ = MODELS['FIXED']
if not os.path.isdir(best_dir):
    tok.save_pretrained(best_dir); m.save_pretrained(best_dir)
print('저장: s2b_k_curve.csv, s2b_halueval.csv, s2b_halueval_items.csv, student_vark/, student_best/')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 's2b_generalization_vark.ipynb'), 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
