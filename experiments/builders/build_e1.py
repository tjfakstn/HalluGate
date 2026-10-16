import json, os
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# E1 — 스트리밍 조기 탐지 평가 프로토콜과 기준선

## 왜 하나

현재 판정기(student_vark)는 **문장이 끝나야** 판단할 수 있어서, 환각이 문장 앞쪽에서 시작돼도
그 문장이 끝날 때까지 잡지 못한다. 이 때문에 pass 모드에서는 누출(약 126자)이, hold 모드에서는 보류 지연(첫 글자 +447ms)이 생긴다.
조기 탐지 모델(E2 이후)을 만들기 전에, 세 가지를 먼저 정해야 한다.

1. **평가 방법**: 선행 연구(Itkin 2026)와 같은 틀로 잰다. 발생 시점, 탐지 지연, 오경보 간격 ARL₀를 쓴다.
2. **기준선**: 현재 방식이 이 틀에서 어떤 수치를 내는지. 새 모델이 넘어야 할 선이다.
3. **개선 여지**: 환각이 문장 안 어디서 시작되는지. 문장 끝까지 기다리는 대신 발생 직후 잡으면 몇 토큰을 줄일 수 있는지.

## 무엇을 하나 (학습 없음)

| 점수기 | 동작 |
|---|---|
| **B1 문장 끝** | 문장이 끝난 시점에 그 문장을 판정 (현재 게이트웨이 방식) |
| **B2 앞부분 재판정 (s토큰마다)** | 문장이 진행되는 동안 s토큰마다 "지금까지의 문장 앞부분"을 같은 판정기로 판정. s = 2, 4, 8 |

B2는 완성 문장으로만 학습한 판정기를 앞부분에 그대로 쓰는 **단순한 조기 탐지**다. E2의 인과적 토큰 점수기는 이것을 넘어야 한다.

## 평가 프로토콜 (docs/RESEARCH_PLAN.md §4)

- 스트림 단위: `tiktoken` cl100k_base 토큰
- 발생 θ: RAGTruth 환각 구간과 처음 겹치는 토큰. 응답당 첫 환각만 센다.
- 경보: 판정 시점의 점수 ≥ 임계값. 경보 후에는 초기화하고 계속 센다.
- ARL₀: 환각 없는 응답의 토큰 수 ÷ 경보 수. **임계값은 검증 분할(90개 응답)에서 ARL₀가 목표값이 되도록 정하고**, 평가는 dev(360개 응답)에서 한다.
- 탐지(**엄격 기준, 주 지표**): 응답의 **첫 경보** τ가 θ 이후일 때만 탐지. θ 전에 경보가 나면 오경보로 끝난 것으로 보고 미탐지.
  재현율, 탐지 지연 τ−θ(탐지된 것만), 검열 EDD(미탐지는 남은 토큰 수). 관대 기준(θ 전 경보 무시)은 참고로만 함께 보고한다.
- **서비스 지표**: 환각 없는 응답 중 한 번이라도 표시되는 비율. 이 비율을 5·10·20%로 맞춘 임계값(검증 분할로 보정)에서의 재현율·지연도 보고한다.
- v2 변경 (v1 결과를 본 뒤): ① 주 지표를 엄격 기준으로 변경 ② 토큰을 공백 제외 내용 위치로 문장·구간에 배정 ③ 서비스 지표 추가.
  v1 수치는 SSOT에 폐기 사유와 함께 남긴다.
- 참고치 (선행 연구, RAGTruth **test**, ARL₀=100): 학습 CUSUM 11.5토큰/재현율 0.24, HistGBM 17.9토큰/0.40. dev 결과와의 비교는 참고용이다.

## 사전 기록 (결과 보기 전)

- 이 실험은 기준선을 재는 실험이라 합격/불합격 기준은 없다. 대신 **점검 항목**을 둔다.
  - B1의 문장 단위 AUROC가 S2-b의 dev 값(0.859)과 ±0.01 안에서 재현될 것
  - 검증 세트로 정한 임계값의 dev 실제 ARL₀가 목표의 0.5~2배 안일 것. 벗어나면 보정 방법을 E4에서 다시 설계한다.
- **예상 (가설 H1)**: B2는 B1보다 탐지 지연이 짧지만, 같은 ARL₀에서 재현율은 비슷하거나 낮을 것이다.
  판정기가 완성 문장으로만 학습돼, 앞부분을 보면 오경보가 늘기 때문이다. 이 가설이 맞으면 E2(앞부분 전용 학습)의 필요성이 입증된다.

> 런타임 **GPU(T4)**. 드라이브의 `selfproject/student_vark`가 필요하다. 약 15~25분.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install pysbd sentence-transformers tiktoken
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'

import json, time, random, os, gc
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
assert os.path.isdir(STUDENT_DIR), f'{STUDENT_DIR} 없음 — S2-b를 USE_DRIVE=True로 실행해 저장할 것'
print('device', DEVICE, '| 판정기', STUDENT_DIR)
""")

md("## 2. 데이터 — dev(평가)와 검증 분할(임계값 보정), S2와 같은 시드")
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
    return '\n'.join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}"
                     for k, v in (si.items() if isinstance(si, dict) else []))

TASKS = ['QA', 'Summary', 'Data2txt']
train_pool = [r for r in responses if r['split'] == 'train']          # test 분할은 사용하지 않음
random.seed(SEED)
dev = []
for tt in TASKS:
    dev += random.sample([r for r in train_pool if r['task_type'] == tt], 120)
dev_ids = {r['id'] for r in dev}
rest = [r for r in train_pool if r['id'] not in dev_ids]
rng = random.Random(11)
teacher_resp = []
for tt in TASKS:
    teacher_resp += rng.sample([r for r in rest if r['task_type'] == tt], 200)
rng.shuffle(teacher_resp)
val = teacher_resp[:int(len(teacher_resp) * 0.15)]                   # student 학습에 쓰지 않은 검증 분할
assert not (dev_ids & {r['id'] for r in val})
print(f'dev {len(dev)} 응답 | 검증 {len(val)} 응답')
""")

md(r"""## 3. 토큰 스트림과 환각 발생 위치

응답을 상용 API처럼 토큰 단위로 흘려보낸다고 보고, 각 토큰의 글자 위치, 환각 발생 토큰 θ, 문장 경계(토큰 위치)를 구한다.
문장 끝 판정(B1)은 문장의 마지막 토큰이 도착한 시점에 판정한다고 둔다. 실제 게이트웨이는 다음 문장 첫 토큰을 봐야
문장이 끝났음을 알기 때문에, B1에 약간 유리한 가정이다.""")
code(r"""
ENC = tiktoken.get_encoding('cl100k_base')
seg = pysbd.Segmenter(language='en', clean=False, char_span=True)

def tokenize_offsets(text):
    ids = ENC.encode(text)
    dec, starts = ENC.decode_with_offsets(ids)
    if dec != text:                       # 드문 경우: 멀티바이트 문자 경계 — 누적 디코드로 대체
        starts, pos = [], 0
        for i in range(len(ids)):
            starts.append(pos); pos = len(ENC.decode(ids[:i + 1]))
    ends = starts[1:] + [len(text)]
    return starts, ends

def prep(r):
    text = r['response']
    ts, te = tokenize_offsets(text)
    # 토큰의 '내용 시작' 위치: cl100k 토큰은 앞 공백을 포함하므로(" The"), 공백을 건너뛴 위치로 문장·환각 구간에 배정한다.
    # (v1에서는 공백 위치로 배정해 문장 경계 토큰이 앞 문장에 붙는 오류가 발생 — 발생 위치의 18%)
    cs = [a + (len(text[a:b]) - len(text[a:b].lstrip())) for a, b in zip(ts, te)]
    cs = [c if c < b else a for c, a, b in zip(cs, ts, te)]          # 공백뿐인 토큰
    spans = [(l['start'], l['end']) for l in (r.get('labels') or [])]
    bad = [j for j in range(len(ts)) if any(min(te[j], b) > max(cs[j], a) for a, b in spans)]
    sents = []
    for sp in seg.segment(text):
        if not sp.sent.strip(): continue
        a, b = sp.start, sp.end
        tok = [j for j in range(len(ts)) if a <= cs[j] < b]
        if not tok: continue
        y = int(any(min(b, e) > max(a, s) for s, e in spans))
        sents.append({'a': a, 'b': b, 't0': tok[0], 't1': tok[-1], 'y': y})
    sent_of = {t: sn['t0'] for sn in sents for t in range(sn['t0'], sn['t1'] + 1)}
    return {'id': str(r['id']), 'task_type': r['task_type'], 'gen_model': r['model'],
            'ref_key': r['source_id'], 'text': text, 'ts': ts, 'te': te, 'n_tok': len(ts),
            'onset': bad[0] if bad else None, 'sents': sents, 'sent_of': sent_of,
            'ends': {sn['t1'] for sn in sents}}

REFS = {}
def to_items(rs):
    out = []
    for r in rs:
        REFS.setdefault(r['source_id'], build_reference(sources[r['source_id']]))
        out.append(prep(r))
    return out

dev_items, val_items = to_items(dev), to_items(val)
for name, it in [('dev', dev_items), ('검증', val_items)]:
    n_h = sum(x['onset'] is not None for x in it)
    print(f'{name}: 응답 {len(it)} (환각 {n_h}, 없음 {len(it)-n_h}) | 토큰 {sum(x["n_tok"] for x in it):,} | '
          f'환각 없는 응답 토큰 {sum(x["n_tok"] for x in it if x["onset"] is None):,}')
""")

code(r"""
# 개선 여지: 환각은 문장 안 어디서 시작되는가?
rows = []
for x in dev_items:
    if x['onset'] is None: continue
    s = next(s for s in x['sents'] if s['t0'] <= x['onset'] <= s['t1']) if any(
        s['t0'] <= x['onset'] <= s['t1'] for s in x['sents']) else None
    if s is None: continue
    rows.append({'id': x['id'], 'task_type': x['task_type'], 'onset': x['onset'],
                 'sent_len_tok': s['t1'] - s['t0'] + 1, 'onset_in_sent': x['onset'] - s['t0'],
                 'to_sent_end': s['t1'] - x['onset'],
                 'onset_chars_to_end': s['b'] - x['ts'][x['onset']]})
onset_df = pd.DataFrame(rows)
print('환각 발생 위치 (dev, 응답당 첫 환각)')
print(onset_df[['sent_len_tok', 'onset_in_sent', 'to_sent_end', 'onset_chars_to_end']].describe().round(1).loc[['mean', '25%', '50%', '75%']])
print(f"\n문장 첫 토큰에서 시작하는 비율: {(onset_df.onset_in_sent == 0).mean():.2f}")
print(f"→ B1(문장 끝 판정)은 잡더라도 최소 {onset_df.to_sent_end.median():.0f}토큰 "
      f"({onset_df.onset_chars_to_end.median():.0f}자) 늦는다 (중앙값). 이것이 조기 탐지로 줄일 수 있는 최대 폭이다.")
""")

md("## 4. 판정기와 판정 지점")
code(r"""
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
tok = AutoTokenizer.from_pretrained(STUDENT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR).to(DEVICE).eval()
K, MAX_LEN, CHUNK_WORDS = 8, 512, 40

chunk_cache = {}
def get_chunks(key):
    if key not in chunk_cache:
        w = REFS[key].split()
        ch = [' '.join(w[i:i + CHUNK_WORDS]) for i in range(0, max(len(w), 1), CHUNK_WORDS)] or [REFS[key]]
        chunk_cache[key] = (ch, retriever.encode(ch, convert_to_tensor=True, normalize_embeddings=True,
                                                 batch_size=128, show_progress_bar=False))
    return chunk_cache[key]

@torch.no_grad()
def score_queries(queries, bs=128):
    # queries: [(ref_key, text)] → logit 배열. 텍스트마다 근거 조각을 새로 검색한다 (앞부분이면 앞부분 기준)
    out = np.zeros(len(queries), dtype=np.float32)
    order = sorted(range(len(queries)), key=lambda i: len(queries[i][1]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        texts = [queries[j][1] for j in idx]
        embs = retriever.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                                batch_size=bs, show_progress_bar=False)
        ctxs = []
        for j, e in zip(idx, embs):
            ch, ce = get_chunks(queries[j][0])
            top = torch.topk(ce @ e, min(K, ce.shape[0])).indices.tolist()
            ctxs.append(' ... '.join(ch[t] for t in sorted(top)))
        enc = tok(ctxs, texts, truncation='only_first', max_length=MAX_LEN, padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):
            out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out
""")

code(r"""
# 판정 지점 만들기. B2는 s=2 지점을 한 번에 계산하고 s=4, 8은 그 부분집합으로 쓴다.
def points(item, s):
    # 반환: [(토큰 위치 t, 판정 텍스트)] — 현재 문장의 앞부분
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

print(f'판정 질의 {len(queries):,}개 (중복 제거) 계산 중...')
t0 = time.perf_counter()
SCORES = score_queries(queries)
print(f'완료 {time.perf_counter()-t0:.0f}초')

def series(x, scorer):
    # 점수기별 (t, score) 목록
    ends = x['ends']
    if scorer == 'B1 문장 끝':
        return [(t, SCORES[q]) for t, q in x['pts2'] if t in ends]
    s = int(scorer.split('s=')[1].rstrip(')'))
    return [(t, SCORES[q]) for t, q in x['pts2'] if t in ends or (t - x['sent_of'][t] + 1) % s == 0]

SCORERS = ['B1 문장 끝', 'B2 앞부분 (s=8)', 'B2 앞부분 (s=4)', 'B2 앞부분 (s=2)']
""")

md("## 5. 점검 — B1 문장 단위 AUROC 재현")
code(r"""
ys, ss = [], []
for x in dev_items:
    for sn in x['sents']:
        q = dict(x['pts2'])[sn['t1']]
        ys.append(sn['y']); ss.append(SCORES[q])
auc_b1 = roc_auc_score(ys, ss)
print(f'B1 문장 단위 AUROC {auc_b1:.3f} (S2-b dev 0.859) → {"재현 OK" if abs(auc_b1 - 0.859) <= 0.01 else "차이 있음 — 문장 분할·토큰 경계 차이 확인"}')
""")

md(r"""## 6. 조기 탐지 평가 — ARL₀ 보정(검증) → 평가(dev)""")
code(r"""
def calibrate(items, scorer, arl0):
    clean = [x for x in items if x['onset'] is None]
    n_tok = sum(x['n_tok'] for x in clean)
    sc = np.sort(np.concatenate([[s for _, s in series(x, scorer)] for x in clean]))[::-1]
    n_alarm = max(1, int(n_tok / arl0))
    return float(sc[n_alarm - 1]) + 1e-6 if n_alarm <= len(sc) else float(sc[-1])

def evaluate(items, scorer, thr):
    # 엄격 기준(주 지표, 선행 연구와 같은 정의): 응답의 **첫 경보**가 발생 θ 이후여야 탐지로 인정.
    #   θ 전에 경보가 울리면 그 응답은 오경보로 끝난 것(미탐지)으로 본다.
    # 관대 기준(참고): θ 전 경보는 무시하고 θ 이후 첫 경보를 탐지로 인정 — v1의 주 지표였으나 재현율이 부풀려짐.
    clean = [x for x in items if x['onset'] is None]
    hall = [x for x in items if x['onset'] is not None]
    alarms = sum(sum(s >= thr for _, s in series(x, scorer)) for x in clean)
    arl = sum(x['n_tok'] for x in clean) / max(alarms, 1)
    clean_flag = np.mean([any(s >= thr for _, s in series(x, scorer)) for x in clean])
    st_det, st_d, st_ch, cedd, le_det, pre = 0, [], [], [], 0, 0
    for x in hall:
        ser = series(x, scorer)
        first = next((t for t, s in ser if s >= thr), None)
        le_hit = next((t for t, s in ser if t >= x['onset'] and s >= thr), None)
        le_det += le_hit is not None
        pre += first is not None and first < x['onset']
        if first is not None and first >= x['onset']:
            st_det += 1; st_d.append(first - x['onset']); cedd.append(first - x['onset'])
            st_ch.append(x['te'][first] - x['ts'][x['onset']])
        else:
            cedd.append(x['n_tok'] - x['onset'])
    return {'realized_ARL0_dev': arl, 'clean_resp_flagged': clean_flag,
            'recall': st_det / len(hall), 'delay_tok_mean': np.mean(st_d) if st_d else np.nan,
            'delay_tok_median': np.median(st_d) if st_d else np.nan,
            'delay_chars_median': np.median(st_ch) if st_ch else np.nan,
            'censored_EDD_tok': np.mean(cedd), 'pre_onset_alarm_rate': pre / len(hall),
            'recall_lenient': le_det / len(hall),
            'calls_per_resp': np.mean([len(series(x, scorer)) for x in items])}

def calibrate_flag(items, scorer, rate):
    # 서비스 기준 보정: 환각 없는 응답 중 표시되는 비율이 rate 이하가 되는 임계값 (응답별 최대 점수의 분위수)
    mx = np.array([max([s for _, s in series(x, scorer)] or [-1e9]) for x in items if x['onset'] is None])
    return float(np.quantile(mx, 1 - rate)) + 1e-6

rows = []
for arl0 in [50, 100, 200, 500]:
    for sc in SCORERS:
        thr = calibrate(val_items, sc, arl0)
        rows.append({'target_ARL0': arl0, 'target_clean_flag': None, 'scorer': sc, 'thr': thr, **evaluate(dev_items, sc, thr)})
for rate in [0.05, 0.10, 0.20]:
    for sc in SCORERS:
        thr = calibrate_flag(val_items, sc, rate)
        rows.append({'target_ARL0': None, 'target_clean_flag': rate, 'scorer': sc, 'thr': thr, **evaluate(dev_items, sc, thr)})
e1 = pd.DataFrame(rows)
pd.set_option('display.width', 200)
cols = ['scorer', 'thr', 'recall', 'delay_tok_mean', 'delay_tok_median', 'delay_chars_median', 'clean_resp_flagged',
        'realized_ARL0_dev', 'pre_onset_alarm_rate', 'recall_lenient', 'calls_per_resp']
print('=== ARL₀=100 (선행 연구와 같은 조건, 엄격 기준) ===')
print(e1[e1.target_ARL0 == 100][cols].round(3).to_string(index=False))
print('\n=== 서비스 기준: 환각 없는 응답 표시 ≤ 10% (검증으로 보정) ===')
print(e1[e1.target_clean_flag == 0.10][cols].round(3).to_string(index=False))
print('\n참고 (선행 연구, RAGTruth test, ARL₀=100): 학습 CUSUM 11.5토큰/재현율 0.24, HistGBM 17.9토큰/0.40')
""")

code(r"""
print('=== 점검: 검증으로 정한 임계값의 dev 실제 ARL₀ (목표의 0.5~2배 안이면 OK) ===')
for r in e1[e1.target_ARL0.notna()].itertuples():
    ratio = r.realized_ARL0_dev / r.target_ARL0
    flag = 'OK' if 0.5 <= ratio <= 2 else '벗어남'
    print(f'  목표 {r.target_ARL0:4} | {r.scorer:16} 실제 {r.realized_ARL0_dev:7.1f} ({ratio:.2f}배) {flag}')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for sc, g in e1[e1.target_ARL0.notna()].groupby('scorer', sort=False):
    ax[0].plot(g.target_ARL0, g.recall, 'o-', label=sc)
    ax[1].plot(g.recall, g.delay_tok_median, 'o-', label=sc)
ax[0].set_xscale('log'); ax[0].set_xlabel('target ARL0 (tokens per false alarm)'); ax[0].set_ylabel('recall')
ax[0].axhline(0.40, color='gray', ls=':', lw=1); ax[0].axhline(0.24, color='gray', ls=':', lw=1)
ax[1].set_xlabel('recall'); ax[1].set_ylabel('median detection delay (tokens)')
ax[1].axhline(onset_df.to_sent_end.median(), color='gray', ls='--', lw=1, label='median tokens onset→sentence end')
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md("## 7. 비용 — 재판정 1회 지연(배치 1)과 응답당 연산량")
code(r"""
@torch.no_grad()
def one_ms(n=150):
    sample = random.Random(SEED).sample(queries, min(n, len(queries)))
    for q in sample[:10]: score_queries([q], bs=1)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for q in sample: score_queries([q], bs=1)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / len(sample)

MS = one_ms()
cost = e1[e1.target_ARL0 == 100][['scorer', 'calls_per_resp']].copy()
cost['gpu_ms_per_resp'] = cost.calls_per_resp * MS
print(f'재판정 1회 {MS:.1f} ms (배치 1, 검색 포함)')
print(cost.round(1).to_string(index=False))
""")

md("## 8. 과제별 (ARL₀=100)")
code(r"""
rows = []
for tt in TASKS:
    its = [x for x in dev_items if x['task_type'] == tt]
    for sc in SCORERS:
        thr = e1[(e1.target_ARL0 == 100) & (e1.scorer == sc)].thr.item()
        r = evaluate(its, sc, thr)
        rows.append({'task': tt, 'scorer': sc, 'recall': r['recall'], 'delay_tok_median': r['delay_tok_median'],
                     'clean_resp_flagged': r['clean_resp_flagged'],
                     'n_hall': sum(x['onset'] is not None for x in its)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
""")

md("## 9. 가설 H1 판정과 저장")
code(r"""
a = e1[e1.target_ARL0 == 100].set_index('scorer')
b1, b2 = a.loc['B1 문장 끝'], a.loc['B2 앞부분 (s=4)']
print('(엄격 기준, ARL₀=100)')
print(f"B1: 재현율 {b1.recall:.3f}, 지연 중앙값 {b1.delay_tok_median:.1f}토큰")
print(f"B2(s=4): 재현율 {b2.recall:.3f}, 지연 중앙값 {b2.delay_tok_median:.1f}토큰")
shorter = b2.delay_tok_median < b1.delay_tok_median
not_better_recall = b2.recall <= b1.recall + 0.02
print('H1 (B2가 지연은 짧고 재현율은 비슷하거나 낮다):', '지지' if shorter and not_better_recall else '기각',
      '→ 앞부분 전용 학습(E2) 필요성' + (' 입증' if shorter and not_better_recall else ' 재검토'))

os.makedirs(os.path.join(OUT_DIR, 'e1'), exist_ok=True)
e1.to_csv(os.path.join(OUT_DIR, 'e1', 'e1_summary.csv'), index=False)
onset_df.to_csv(os.path.join(OUT_DIR, 'e1', 'e1_onset_stats.csv'), index=False)
pts = [{'id': x['id'], 'split': sp, 'task_type': x['task_type'], 'onset': x['onset'], 'n_tok': x['n_tok'],
        't': t, 'score': float(SCORES[q]), 'is_sent_end': t in {s['t1'] for s in x['sents']}}
       for sp, its in [('dev', dev_items), ('val', val_items)] for x in its for t, q in x['pts2']]
pd.DataFrame(pts).to_csv(os.path.join(OUT_DIR, 'e1', 'e1_points.csv'), index=False)
print('저장:', os.path.join(OUT_DIR, 'e1'), '(e1_summary.csv, e1_onset_stats.csv, e1_points.csv)')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 'e1_streaming_protocol_baselines.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
