import os
import json

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})

md("""# S1-2 — 게이트키퍼 실현 가능성 실험

## 목적

MiniCheck는 AUROC 0.814로 정확도 기준을 만족했지만 문장당 379ms로 실시간 검증에 쓸 수 없다.
모든 문장을 MiniCheck에 보내는 대신, **수 ms 안에 계산되는 가벼운 신호로 위험한 문장만 골라
MiniCheck에 보내면** 평균 지연을 줄일 수 있다. 이 실험은 그 구조가 실제로 가능한지 확인한다.

## 구조

```
문장 → [게이트키퍼: 수 ms] → 위험하면 → [MiniCheck: 379ms] → 차단 여부
                         → 안전하면 → 바로 통과
```

게이트키퍼가 통과시킨 환각은 영영 못 잡으므로, **환각 문장을 놓치지 않는 비율(커버리지)**이 핵심이다.

## 가벼운 신호 후보

| 신호 | 의미 | 비용 |
|---|---|---|
| 새 단어 비율 | 문장의 내용어 중 근거에 없는 단어의 비율 | 1ms 미만 |
| 근거에 없는 숫자 개수 | 수치 오류는 대표적인 충돌형 환각 | 1ms 미만 |
| 근거에 없는 고유명사 개수 | 인명·지명·상호 날조 | 1ms 미만 |
| 임베딩 유사도 | 문장과 가장 비슷한 근거 문장의 코사인 유사도 | 수 ms |
| 문장 길이·위치·과제 유형 | 보조 신호 | 무시 가능 |

## 과적합 방지 설계

- 게이트키퍼는 **평가 표본과 겹치지 않는 별도의 train 응답**으로 학습한다.
- 평가는 직전 실험과 같은 360개 응답(2,629문장)에서 수행하여 MiniCheck 결과와 직접 비교한다.
- test 분할은 열지 않는다.

## 사전 정의 기준 (결과 보기 전에 확정)

- **통과**: 환각 커버리지 95%를 유지하면서 MiniCheck 호출 비율이 **50% 이하**
  → 경량 게이트키퍼 방향으로 S2 진행
- **미달**: 호출 비율이 50%를 넘으면 가벼운 신호만으로는 부족 → S2를 MiniCheck 증류 방향으로 전환
- 보조 기준: 그 지점에서 시스템 전체 AUROC가 MiniCheck 전수 검증 대비 **0.02 이내**로 유지

> 런타임 유형 **GPU(T4)**. MiniCheck 점수 산출에 약 15~20분이 걸리며, 결과를 CSV로 저장하므로
> 구글 드라이브를 연결해두면 다음 실험에서 재사용할 수 있다.
""")

md("## 1. 환경 준비")
code("""!pip -q install pysbd sentencepiece scikit-learn pandas matplotlib nltk sentence-transformers
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'already cloned'
!pip -q install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main" 2>&1 | tail -2

import nltk
for pkg in ['punkt', 'punkt_tab']:
    try: nltk.download(pkg, quiet=True)
    except Exception as e: print(pkg, 'download 실패:', e)

import json, time, random, re, os, gc
import numpy as np, pandas as pd, torch, pysbd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEVICE)""")

code("""# (선택) 구글 드라이브에 결과를 저장하려면 USE_DRIVE = True
USE_DRIVE = False
OUT_DIR = '.'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT_DIR = '/content/drive/MyDrive/selfproject'
    os.makedirs(OUT_DIR, exist_ok=True)
print('저장 위치:', OUT_DIR)""")

md("## 2. 데이터 — 평가 표본은 직전 실험과 동일, 게이트키퍼 학습 표본은 별도")
code("""BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources   = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
for r in responses:
    r['task_type'] = sources[r['source_id']]['task_type']

def build_reference(src):
    si, tt = src['source_info'], src['task_type']
    if tt == 'QA':
        return f"Question: {si.get('question','')}\\n{si.get('passages','')}"
    if tt == 'Summary':
        return si if isinstance(si, str) else json.dumps(si, ensure_ascii=False)
    parts = []
    for k, v in (si.items() if isinstance(si, dict) else []):
        parts.append(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}")
    return '\\n'.join(parts)

# 평가 표본: 직전 실험(task1b, task1c)과 동일한 시드·순서로 추출
N_PER_TASK = 120
dev_pool = [r for r in responses if r['split'] == 'train']
random.seed(SEED)
dev = []
for tt in ['QA', 'Summary', 'Data2txt']:
    cand = [r for r in dev_pool if r['task_type'] == tt]
    dev += random.sample(cand, min(N_PER_TASK, len(cand)))
dev_ids = {r['id'] for r in dev}

# 게이트키퍼 학습 표본: 평가 표본과 겹치지 않는 train 응답
N_GATE_TRAIN_PER_TASK = 200
rng_gate = random.Random(7)
gate_train = []
for tt in ['QA', 'Summary', 'Data2txt']:
    cand = [r for r in dev_pool if r['task_type'] == tt and r['id'] not in dev_ids]
    gate_train += rng_gate.sample(cand, min(N_GATE_TRAIN_PER_TASK, len(cand)))

assert not ({r['id'] for r in gate_train} & dev_ids), '표본이 겹침'
print(f'평가 표본 {len(dev)}개 응답 | 게이트키퍼 학습 표본 {len(gate_train)}개 응답 (겹침 없음)')""")

code("""seg = pysbd.Segmenter(language='en', clean=False, char_span=True)

def split_sentences(t):
    return [(s.start, s.end, t[s.start:s.end]) for s in seg.segment(t) if s.sent.strip()]

def to_sentence_df(resp_list):
    recs = []
    for r in resp_list:
        labels = r.get('labels') or []
        for i, (a, b, s) in enumerate(split_sentences(r['response'])):
            recs.append({'resp_id': r['id'], 'source_id': r['source_id'], 'task_type': r['task_type'],
                         'sent_idx': i, 'sentence': s,
                         'y': int(any(min(b, l['end']) > max(a, l['start']) for l in labels))})
    df = pd.DataFrame(recs)
    df['reference'] = df.source_id.map(lambda sid: build_reference(sources[sid]))
    return df

dev_df   = to_sentence_df(dev)
train_df = to_sentence_df(gate_train)
print(f'평가 문장 {len(dev_df):,}개 (양성 {dev_df.y.mean()*100:.1f}%) | '
      f'학습 문장 {len(train_df):,}개 (양성 {train_df.y.mean()*100:.1f}%)')
if len(dev_df) != 2629:
    print('주의: 직전 실험의 2,629문장과 다름 — 표본 재현 여부 확인 필요')""")

md("""## 3. MiniCheck 기준선 — 평가 표본 전 문장

직전 실험에서는 점수를 파일로 저장하지 않았으므로 한 번 더 산출하고, 이번에는 CSV로 남겨
이후 실험(문맥 범위 재실험, 증류 등)에서 재사용한다. 저장된 파일이 있으면 불러온다.""")
code("""MC_PATH = os.path.join(OUT_DIR, 'minicheck_dev_scores.csv')

if os.path.exists(MC_PATH):
    saved = pd.read_csv(MC_PATH)
    assert len(saved) == len(dev_df), '저장된 파일의 표본이 다름 — 파일을 지우고 다시 실행'
    dev_df['mc_score'] = saved.mc_score.values
    MC_MS = float(saved.mc_ms_per_sentence.iloc[0])
    print(f'저장된 MiniCheck 점수 사용 ({MC_MS:.1f} ms/문장)')
else:
    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name='flan-t5-large', enable_prefix_caching=False)
    t0 = time.perf_counter()
    _, raw_prob, _, _ = mc.score(docs=dev_df.reference.tolist(), claims=dev_df.sentence.tolist())
    MC_MS = (time.perf_counter() - t0) * 1000 / len(dev_df)
    dev_df['mc_score'] = 1.0 - np.asarray(raw_prob)     # 클수록 환각
    pd.DataFrame({'resp_id': dev_df.resp_id, 'sent_idx': dev_df.sent_idx,
                  'mc_score': dev_df.mc_score, 'mc_ms_per_sentence': MC_MS}).to_csv(MC_PATH, index=False)
    del mc; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    print(f'MiniCheck 산출 완료 ({MC_MS:.1f} ms/문장), 저장: {MC_PATH}')

MC_AUROC = roc_auc_score(dev_df.y, dev_df.mc_score)
MC_AUPRC = average_precision_score(dev_df.y, dev_df.mc_score)
print(f'MiniCheck 전수 검증 기준선: AUROC {MC_AUROC:.3f} | AUPRC {MC_AUPRC:.3f}')""")

md("""## 4. 가벼운 신호 추출

실제 미들웨어에서 근거 문서는 응답 생성 전에 이미 주어지므로, **근거 쪽 처리(문장 분할,
임베딩)는 미리 계산해 캐시**할 수 있다. 따라서 문장당 지연에는 생성된 문장 쪽 계산만 포함한다.""")
code("""from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)

TOKEN_RE = re.compile(r"[a-z]+|\\d+(?:[.,]\\d+)*")
NUM_RE   = re.compile(r"\\d+(?:[.,]\\d+)*")
CAP_RE   = re.compile(r"\\b[A-Z][a-zA-Z]+\\b")
STOP     = set(ENGLISH_STOP_WORDS)

def norm_num(x): return x.replace(',', '')

# 근거 쪽 캐시 (응답 생성 전에 준비 가능한 부분)
ref_cache = {}
def get_ref(sid):
    if sid not in ref_cache:
        text = build_reference(sources[sid])
        sents = [s for _, _, s in split_sentences(text)] or [text]
        ref_cache[sid] = {
            'tokens': set(TOKEN_RE.findall(text.lower())),
            'nums':   {norm_num(n) for n in NUM_RE.findall(text)},
            'emb':    emb_model.encode(sents, convert_to_tensor=True, normalize_embeddings=True,
                                       show_progress_bar=False),
        }
    return ref_cache[sid]

def sentence_features(sent, sid, emb=None):
    ref = get_ref(sid)
    toks  = [t for t in TOKEN_RE.findall(sent.lower()) if t not in STOP and len(t) > 1]
    novel = [t for t in toks if t not in ref['tokens']]
    nums  = [norm_num(n) for n in NUM_RE.findall(sent)]
    caps  = CAP_RE.findall(sent)
    if caps and sent.lstrip().startswith(caps[0]):
        caps = caps[1:]                                     # 문장 첫 단어의 대문자는 제외
    if emb is None:
        emb = emb_model.encode([sent], convert_to_tensor=True, normalize_embeddings=True,
                               show_progress_bar=False)[0]
    sim = float((ref['emb'] @ emb).max())
    return {
        'novel_ratio':  len(novel) / len(toks) if toks else 0.0,   # 클수록 위험
        'novel_tokens': len(novel),
        'novel_nums':   sum(n not in ref['nums'] for n in nums),
        'novel_caps':   sum(c.lower() not in ref['tokens'] for c in caps),
        'sim_gap':      1.0 - sim,                                  # 클수록 위험
        'n_chars':      len(sent),
    }

def build_features(df):
    for sid in df.source_id.unique(): get_ref(sid)          # 근거 캐시 선계산
    embs = emb_model.encode(df.sentence.tolist(), convert_to_tensor=True, normalize_embeddings=True,
                            batch_size=64, show_progress_bar=False)
    rows = [sentence_features(s, sid, e) for s, sid, e in zip(df.sentence, df.source_id, embs)]
    F = pd.DataFrame(rows, index=df.index)
    F['sent_idx'] = df.sent_idx.clip(upper=8).values
    for tt in ['QA', 'Summary', 'Data2txt']:
        F[f'task_{tt}'] = (df.task_type == tt).astype(int).values
    return F

t0 = time.perf_counter(); X_train = build_features(train_df); print(f'학습 특징 {time.perf_counter()-t0:.0f}초')
t0 = time.perf_counter(); X_dev   = build_features(dev_df);   print(f'평가 특징 {time.perf_counter()-t0:.0f}초')
X_dev.describe().T[['mean','std','max']].round(3)""")

md("""## 5. 게이트키퍼 지연 측정 — 실제 스트리밍처럼 문장 하나씩

배치로 재면 실제보다 빠르게 나오므로, 문장이 하나씩 도착하는 상황을 가정해 배치 크기 1로 잰다.""")
code("""sample = dev_df.sample(200, random_state=SEED)
for _ in range(10): sentence_features(sample.sentence.iloc[0], sample.source_id.iloc[0])   # 워밍업
if DEVICE == 'cuda': torch.cuda.synchronize()
t0 = time.perf_counter()
for s, sid in zip(sample.sentence, sample.source_id):
    sentence_features(s, sid)
if DEVICE == 'cuda': torch.cuda.synchronize()
GATE_MS = (time.perf_counter() - t0) * 1000 / len(sample)
print(f'게이트키퍼 신호 산출: {GATE_MS:.2f} ms/문장 (MiniCheck {MC_MS:.0f} ms 대비 {MC_MS/GATE_MS:.0f}배 빠름)')""")

md("## 6. 신호별 판별력 — 각 신호가 혼자서 환각을 얼마나 가려내는가")
code("""SIGNALS = ['novel_ratio', 'novel_tokens', 'novel_nums', 'novel_caps', 'sim_gap', 'n_chars']
rows = []
for s in SIGNALS:
    rows.append({'signal': s,
                 'AUROC': roc_auc_score(dev_df.y, X_dev[s]),
                 'AUPRC': average_precision_score(dev_df.y, X_dev[s])})
sig_res = pd.DataFrame(rows).sort_values('AUROC', ascending=False).reset_index(drop=True)
print(f'참고 — MiniCheck AUROC {MC_AUROC:.3f}, 무작위 AUPRC {dev_df.y.mean():.3f}')
sig_res.round(3)""")

md("## 7. 게이트키퍼 학습 — 신호 결합 (별도 표본으로 학습, 평가 표본에서 측정)")
code("""gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
gate.fit(X_train, train_df.y)
dev_df['gate_score'] = gate.predict_proba(X_dev)[:, 1]

GATE_AUROC = roc_auc_score(dev_df.y, dev_df.gate_score)
print(f'게이트키퍼 단독 AUROC {GATE_AUROC:.3f} | AUPRC {average_precision_score(dev_df.y, dev_df.gate_score):.3f}')
coef = pd.Series(gate[-1].coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
print('\\n신호별 가중치 (표준화 기준, 절댓값 순)'); print(coef.round(3))""")

md("""## 8. 핵심 — 호출 비율 대비 환각 커버리지와 시스템 성능

호출 비율 r만큼 게이트키퍼 점수가 높은 문장을 MiniCheck로 보낸다. 보내지 않은 문장은
'통과'로 처리되므로 시스템 점수를 최저로 둔다.

- **커버리지**: 실제 환각 문장 중 MiniCheck까지 도달한 비율
- **시스템 AUROC**: 게이트키퍼 + MiniCheck 전체 구조의 판별력
- **예상 지연**: 게이트키퍼 지연 + r × MiniCheck 지연
- **무작위 라우팅**: 게이트키퍼 없이 무작위로 r만큼 보낼 때 (비교 기준)""")
code("""y  = dev_df.y.values
mc = dev_df.mc_score.values
g  = dev_df.gate_score.values

def system_eval(route_mask):
    sys_score = np.where(route_mask, mc, -1.0)
    return (route_mask[y == 1].mean(),
            roc_auc_score(y, sys_score),
            average_precision_score(y, sys_score))

rows = []
rng = np.random.default_rng(SEED)
for r in np.round(np.arange(0.05, 1.0001, 0.05), 2):
    thr = np.quantile(g, 1 - r) if r < 1 else -np.inf
    cov, au, ap = system_eval(g >= thr)
    rand = [system_eval(rng.random(len(y)) < r) for _ in range(20)]
    rows.append({'route_rate': r, 'coverage': cov, 'sys_AUROC': au, 'sys_AUPRC': ap,
                 'rand_coverage': np.mean([x[0] for x in rand]),
                 'rand_AUROC': np.mean([x[1] for x in rand]),
                 'exp_latency_ms': GATE_MS + r * MC_MS})
curve = pd.DataFrame(rows)
curve.round(3)""")

code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

ax[0].plot(curve.route_rate, curve.coverage, 'o-', label='gatekeeper')
ax[0].plot(curve.route_rate, curve.rand_coverage, '--', color='gray', label='random routing')
ax[0].axhline(0.95, color='r', ls=':', label='coverage 95%')
ax[0].axvline(0.50, color='orange', ls=':', label='route rate 50%')
ax[0].set_xlabel('MiniCheck call rate'); ax[0].set_ylabel('hallucination coverage')
ax[0].set_title('How many hallucinations reach MiniCheck'); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(curve.exp_latency_ms, curve.sys_AUROC, 'o-', label='gatekeeper + MiniCheck')
ax[1].plot(curve.exp_latency_ms, curve.rand_AUROC, '--', color='gray', label='random + MiniCheck')
ax[1].axhline(MC_AUROC, color='g', ls=':', label=f'MiniCheck only ({MC_AUROC:.3f})')
ax[1].axhline(MC_AUROC - 0.02, color='r', ls=':', label='-0.02 tolerance')
ax[1].set_xlabel('expected latency per sentence (ms)'); ax[1].set_ylabel('system AUROC')
ax[1].set_title('Accuracy vs latency'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()""")

md("## 9. 사전 기준 판정")
code("""COV_TARGET, RATE_MAX, AUROC_TOL = 0.95, 0.50, 0.02

# 커버리지 95%를 달성하는 최소 호출 비율을 촘촘하게 탐색
fine = []
for r in np.linspace(0.01, 1.0, 100):
    thr = np.quantile(g, 1 - r) if r < 1 else -np.inf
    cov, au, ap = system_eval(g >= thr)
    fine.append((r, cov, au, ap))
fine = pd.DataFrame(fine, columns=['route_rate', 'coverage', 'sys_AUROC', 'sys_AUPRC'])
hit = fine[fine.coverage >= COV_TARGET].iloc[0]

lat = GATE_MS + hit.route_rate * MC_MS
print(f'MiniCheck 전수 검증 : AUROC {MC_AUROC:.3f} | {MC_MS:.0f} ms/문장')
print(f'게이트키퍼 신호     : {GATE_MS:.2f} ms/문장')
print(f'커버리지 {COV_TARGET:.0%} 달성 지점')
print(f'  MiniCheck 호출 비율 : {hit.route_rate:.0%}')
print(f'  시스템 AUROC       : {hit.sys_AUROC:.3f}  (전수 대비 {hit.sys_AUROC - MC_AUROC:+.3f})')
print(f'  예상 평균 지연      : {lat:.0f} ms/문장  (전수 대비 {1 - lat/MC_MS:.0%} 감소)')
print()
ok_rate  = hit.route_rate <= RATE_MAX
ok_auroc = hit.sys_AUROC >= MC_AUROC - AUROC_TOL
print(f'호출 비율 ≤ {RATE_MAX:.0%}   : {"통과" if ok_rate else "미달"}')
print(f'AUROC 손실 ≤ {AUROC_TOL} : {"통과" if ok_auroc else "미달"}')
print()
if ok_rate and ok_auroc:
    print('결론: 경량 게이트키퍼 구조가 성립 → S2에서 게이트키퍼 고도화 진행')
elif ok_rate:
    print('결론: 호출은 충분히 줄었으나 판별력 손실이 큼 → 게이트키퍼 신호 보강 필요')
else:
    print('결론: 가벼운 신호만으로는 호출을 충분히 줄일 수 없음 → S2를 MiniCheck 증류 방향으로 전환')""")

md("## 10. 오류 분석 — 게이트키퍼가 놓친 환각은 어떤 것인가")
code("""thr_hit = np.quantile(g, 1 - hit.route_rate)
missed = dev_df[(dev_df.y == 1) & (dev_df.gate_score < thr_hit)]
print(f'놓친 환각 {len(missed)}개 / 전체 환각 {int(dev_df.y.sum())}개')
print('\\n과제 유형별 놓친 비율')
print((dev_df[dev_df.y == 1].assign(missed=lambda d: d.gate_score < thr_hit)
       .groupby('task_type').missed.mean().round(3)))
pd.set_option('display.max_colwidth', 160)
missed[['task_type', 'sentence', 'gate_score', 'mc_score']].head(10)""")

code("""curve.to_csv(os.path.join(OUT_DIR, 'gatekeeper_curve.csv'), index=False)
sig_res.to_csv(os.path.join(OUT_DIR, 'gatekeeper_signals.csv'), index=False)
dev_df[['resp_id', 'sent_idx', 'task_type', 'y', 'mc_score', 'gate_score']].to_csv(
    os.path.join(OUT_DIR, 'gatekeeper_dev_scores.csv'), index=False)
print('저장 완료:', OUT_DIR)""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 's1_gatekeeper_feasibility.ipynb'), 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
