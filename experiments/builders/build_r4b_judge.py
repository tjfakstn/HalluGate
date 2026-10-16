"""R4b-A 노트북 생성: Claude API 대신 Colab GPU에서 공개 LLM(vLLM)으로 학습 데이터(gold-large 3,000응답)를 재판정한다.
쓰기 전에 cal-L 400응답에서 R4의 Opus 판정과 일치도를 재는 관문을 통과해야 한다.
결과(labels_train.json)는 R4b-B(재학습) 노트북이 읽는다. 데이터·문장 분할 코드는 R1b 노트북에서 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
r1b = json.load(open(os.path.join(NB_DIR, 'r1b_sentence_local_labels.ipynb')))
C = [''.join(c['source']) for c in r1b['cells'] if c['cell_type'] == 'code']
def find(prefix):
    return next(c for c in C if c.startswith(prefix))
DATA = find("BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find("# 학습 응답: S2/S2-b의 gold-large")
PREP = find("ENC = tiktoken.get_encoding")

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R4b-A — 공개 LLM으로 학습 데이터 재판정 (Claude API 비용 없음)

## 왜 하나

R4b는 학습 데이터(gold-large 3,000응답)의 정답 누락을 고쳐 다시 학습하는 실험이다. 처음 계획은 Claude Sonnet API(약 $10)였지만 비용 때문에 **Colab GPU에서 공개 LLM을 직접 돌려** 재판정한다.
공개 모델은 Opus보다 약할 수 있으므로, **쓰기 전에 품질을 잰다.** R4에서 cal-L 1,608응답은 이미 Opus가 판정했으므로, 그중 400응답을 공개 모델로도 판정해 Opus와 비교한다.

## 무엇을 하나

1. 재판정자: GPU가 80GB면 `Qwen/Qwen2.5-72B-Instruct-AWQ`, 40GB면 `Qwen/Qwen2.5-32B-Instruct-AWQ` (4비트 양자화, vLLM, 탐욕 생성). 1차 `Qwen2.5-14B-Instruct`는 관문 미달이었고, 이것이 사전에 정한 재도전 1회다. R4와 **같은 지시문**(RAGTruth 환각 정의, 문장별 S/H/A, 구절, JSON).
2. **관문(cal-L 400응답, 시드 5)**: Opus 판정과 비교.
3. 관문 통과 시 학습 3,000응답 전체를 판정해 드라이브 `r4b/labels_train.json`에 저장 → R4b-B 노트북이 읽는다.
   평가 정답(C1)을 만든 Opus와 다른 모델이 학습 정답을 만들게 되어, 판정기가 평가자 기준을 외우는 순환도 줄어든다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **관문 G**: cal-L 400응답에서 (a) Opus와 문장 단위 κ(H 대 나머지) ≥ 0.50, 그리고 (b) 원래 정답 환각 문장에 H 또는 A ≥ 0.70. (참고: R4에서 Sonnet–Opus κ 0.778)
- 관문 미달이면 학습 데이터 판정을 하지 않고 R4b를 보류한다(결과는 보고만).
- 판정 실패(파싱 실패·길이 초과)는 원래 정답을 유지하고 개수를 보고한다.

> **런타임: A100** (80GB면 72B, 40GB면 32B를 자동 선택). 모델 다운로드 약 19~39GB, 판정 시간은 처음 200건 속도로 남은 시간을 출력한다.
> vLLM 설치 후 **런타임 재시작을 요구하면 재시작하고 1절부터 다시 실행**한다(설치는 건너뜀).
> 판정 결과는 200건마다 드라이브 `r4b/judge_cache_*.jsonl`에 저장되고, 끊겨도 이어서 판정한다.
> 드라이브에 `r0/r0_items.pkl`, `r4/labels_claude-opus-5-5.json`이 있어야 한다.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install vllm pysbd tiktoken
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'

import json, time, random, os, re, pickle
import numpy as np, pandas as pd, pysbd, tiktoken
from sklearn.metrics import cohen_kappa_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
USE_DRIVE = True
OUT_DIR = '/content/drive/MyDrive/selfproject' if USE_DRIVE else '.'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
R0_DIR, R4_DIR, R4B_DIR = (os.path.join(OUT_DIR, d) for d in ['r0', 'r4', 'r4b'])
os.makedirs(R4B_DIR, exist_ok=True)
OPUS_P = os.path.join(R4_DIR, 'labels_claude-opus-5-5.json')
assert os.path.exists(OPUS_P), 'R4 Opus 판정(r4/labels_claude-opus-5-5.json)이 있어야 한다'

# 1차 'Qwen/Qwen2.5-14B-Instruct'는 관문 미달(κ 0.472, H·A 0.652, 10/14). 재도전 1회:
# GPU 메모리 70GB 이상(A100 80GB)이면 72B AWQ(약 39GB), 아니면(A100 40GB) 32B AWQ(약 19GB). 어느 쪽이든 관문은 1회만.
import torch
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
LLM_ID = 'Qwen/Qwen2.5-72B-Instruct-AWQ' if GPU_GB >= 70 else 'Qwen/Qwen2.5-32B-Instruct-AWQ'
print(f'GPU {GPU_GB:.0f}GB → 재판정자 {LLM_ID}')
MODEL_TAG = LLM_ID.split('/')[-1]
MAX_MODEL_LEN, MAX_NEW = 12288, 2048
""")

md("## 2. 데이터 — 학습 3,000응답 (R1b와 같은 분할·문장 분할)과 관문용 cal-L")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
code(r"""
TRAIN_ITEMS = to_items(train)
ITEMS = {f'tr-{x["id"]}': x for x in TRAIN_ITEMS}

# 관문: cal-L 400응답 (R0 문장 분할 그대로, 원문은 RAGTruth에서)
RESP = {str(r['id']): r for r in responses}
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
CAL = R0P['items']['calL']
for x in CAL:
    r = RESP[x['id']]; assert r['split'] == 'train'
    x['text'] = r['response']; x['ref_key'] = r['source_id']
    REFS.setdefault(r['source_id'], build_reference(sources[r['source_id']]))
OPUS = {k: {int(i): v for i, v in d.items()} for k, d in json.load(open(OPUS_P)).items()}
GATE_IDS = [f'calL-{x["id"]}' for x in random.Random(5).sample(CAL, min(400, len(CAL))) if f'calL-{x["id"]}' in OPUS]
ITEMS.update({f'calL-{x["id"]}': x for x in CAL})
print(f'학습 {len(TRAIN_ITEMS):,}응답 | 관문 cal-L {len(GATE_IDS)}응답 (Opus 판정 있음)')
""")

md("## 3. 지시문 (R4와 같음)")
code(r"""
SYSTEM = '''You are an expert annotator for RAGTruth, a benchmark of hallucinations in retrieval-augmented generation.
You receive a REFERENCE that a language model was given, and the model's RESPONSE split into numbered sentences.
Label EVERY sentence against the REFERENCE only (not world knowledge).

A sentence contains a hallucination if any part of it is:
- Evident conflict: clearly contradicts the reference (wrong fact, number, name, date, negation).
- Subtle conflict: deviates from the reference in a way that changes meaning (altered term, scope, time, attribution).
- Evident baseless info: introduces information that is not in the reference, even if it may be true in the real world.
- Subtle baseless info: adds unsupported inference, opinion, or embellishment (e.g. "popular", "cozy", "family-friendly") that the reference does not support.

Not hallucinations: faithful paraphrase; reasonable summarization; statements that the reference lacks the information or declining to answer;
meta phrases such as "Based on the passages,". For Data2txt (business data + user reviews), claims supported by the structured data or by what reviewers say are supported.

Labels: "S" = no hallucination. "H" = contains a hallucination. "A" = ambiguous, careful annotators could reasonably disagree.
For every H or A sentence also give: "type" (one of: evident_conflict, subtle_conflict, evident_baseless, subtle_baseless),
"phrase" (the shortest span copied EXACTLY, character for character, from that sentence that is hallucinated), and "reason" (at most 20 words).

Return ONLY a JSON object, no other text:
{"sentences": [{"i": 1, "label": "S"}, {"i": 2, "label": "H", "type": "evident_baseless", "phrase": "...", "reason": "..."}]}
with exactly one entry per numbered sentence, in order.'''

TASK_DESC = {'QA': 'QA — answer the question using the passages',
             'Summary': 'Summary — summarize the news article',
             'Data2txt': 'Data2txt — write an overview of a business from structured data and user reviews'}

def sent_texts(x):
    return [x['text'][sn['a']:sn['b']].strip() for sn in x['sents']]

def user_msg(x):
    lines = '\n'.join(f'[{i + 1}] {s}' for i, s in enumerate(sent_texts(x)))
    return f"TASK: {TASK_DESC[x['task_type']]}\n\nREFERENCE:\n<<<\n{REFS[x['ref_key']]}\n>>>\n\nRESPONSE ({len(x['sents'])} numbered sentences):\n{lines}"

def parse(text, n):
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        labs = {}
        for e in d.get('sentences', []):
            i, lab = int(e.get('i', 0)), str(e.get('label', '')).strip().upper()[:1]
            if 1 <= i <= n and lab and lab in 'SHA':
                labs[i] = {'label': lab, 'type': e.get('type', ''), 'phrase': e.get('phrase', '') or '', 'reason': e.get('reason', '')}
        return labs, len(labs) == n
    except Exception:
        return {}, False
""")

md("## 4. 모델 불러오기 (vLLM)")
code(r"""
# 코랩(Jupyter)에서는 sys.stdout.fileno()가 없어 vLLM 엔진 시작이 실패한다 → 엔진을 같은 프로세스에서 돌리고 fileno를 보충
import sys, contextlib
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
for _s, _real in ((sys.stdout, sys.__stdout__), (sys.stderr, sys.__stderr__)):
    try:
        _s.fileno()
    except Exception:
        _s.fileno = _real.fileno
from vllm import LLM, SamplingParams
try:
    import vllm.utils.system_utils as _su, vllm.distributed.parallel_state as _ps
    _su.suppress_stdout = _ps.suppress_stdout = contextlib.nullcontext
except Exception as e:
    print('suppress_stdout 패치 건너뜀:', type(e).__name__)
llm = LLM(model=LLM_ID, dtype='auto' if 'AWQ' in LLM_ID else 'bfloat16', max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90, seed=0)
tok = llm.get_tokenizer()
GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
RETRY = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_NEW, seed=1)

def prompt_of(cid):
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_msg(ITEMS[cid])}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def judge(cids, cache_name, chunk=200):
    # 200건씩 판정해 드라이브 jsonl에 이어 쓴다. 이미 있는 것은 건너뛴다.
    path = os.path.join(R4B_DIR, f'judge_cache_{MODEL_TAG}_{cache_name}.jsonl')    # 모델별로 따로 저장
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[r['cid']] = r
    todo = [c for c in cids if c not in done]
    print(f'[{cache_name}] 완료 {len(done):,} | 남음 {len(todo):,}')
    t0 = time.perf_counter(); n_done = 0
    for k in range(0, len(todo), chunk):
        part = todo[k:k + chunk]
        prompts, keep, skipped = [], [], []
        for c in part:
            p = prompt_of(c)
            (keep if len(tok.encode(p)) <= MAX_MODEL_LEN - MAX_NEW else skipped).append(c)
            if keep and keep[-1] == c: prompts.append(p)
        outs = llm.generate(prompts, GREEDY, use_tqdm=False) if prompts else []
        recs, bad = [], []
        for c, o in zip(keep, outs):
            labs, ok = parse(o.outputs[0].text, len(ITEMS[c]['sents']))
            (recs.append({'cid': c, 'ok': True, 'labs': {str(i): v for i, v in labs.items()}}) if ok else bad.append(c))
        if bad:                                       # 파싱 실패는 샘플링으로 1회 더
            outs2 = llm.generate([prompt_of(c) for c in bad], RETRY, use_tqdm=False)
            for c, o in zip(bad, outs2):
                labs, ok = parse(o.outputs[0].text, len(ITEMS[c]['sents']))
                recs.append({'cid': c, 'ok': ok, 'labs': {str(i): v for i, v in labs.items()} if ok else {}})
        recs += [{'cid': c, 'ok': False, 'labs': {}, 'too_long': True} for c in skipped]
        with open(path, 'a') as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + '\n'); done[r['cid']] = r
        n_done += len(part); el = time.perf_counter() - t0
        print(f'  {n_done:,}/{len(todo):,} | {el/60:.1f}분, 남은 예상 {el / n_done * (len(todo) - n_done) / 60:.0f}분 | 실패 누적 {sum(not r["ok"] for r in done.values())}', flush=True)
    return {c: {int(i): v for i, v in done[c]['labs'].items()} for c in cids if done.get(c, {}).get('ok')}
""")

md(r"""## 5. 관문 — cal-L 400응답에서 Opus와 비교""")
code(r"""
G = judge(GATE_IDS, 'gate_calL')
rows = []
for cid in GATE_IDS:
    if cid not in G: continue
    x = ITEMS[cid]
    for i, sn in enumerate(x['sents']):
        rows.append((cid, i + 1, sn['y'], OPUS[cid][i + 1]['label'], G[cid][i + 1]['label'], x['task_type']))
J = pd.DataFrame(rows, columns=['cid', 'i', 'y', 'opus', 'open', 'task'])
kappa_H = cohen_kappa_score(J.opus == 'H', J.open == 'H')
kappa_3 = cohen_kappa_score(J.opus, J.open)
sens_HA = J[J.y == 1].open.isin(['H', 'A']).mean()
cover_g = len(G) / len(GATE_IDS)
print(f'판정 확보 {len(G)}/{len(GATE_IDS)} | 문장 {len(J):,}')
print(f'Opus와 κ(H 대 나머지) {kappa_H:.3f} (≥ 0.50), κ(3분류) {kappa_3:.3f} | 원래 환각 문장 H·A {sens_HA:.3f} (≥ 0.70)')
print(f'H 비율: 공개 모델 {(J.open == "H").mean():.3f} vs Opus {(J.opus == "H").mean():.3f}')
print(pd.crosstab(J.opus, J.open, rownames=['Opus'], colnames=[LLM_ID.split('/')[-1]]))
for tt in TASKS:
    s = J[J.task == tt]
    print(f'  {tt:9} κ(H) {cohen_kappa_score(s.opus == "H", s.open == "H"):.3f} | 원래 환각 문장 H·A {s[s.y == 1].open.isin(["H", "A"]).mean():.3f}')
GATE = kappa_H >= 0.50 and sens_HA >= 0.70
print(f"\n=== 관문 G: {'통과 → 6절로' if GATE else '미달 → 학습 데이터 판정을 하지 않음 (R4b 보류)'} ===")
json.dump({'llm': LLM_ID, 'gate': bool(GATE), 'kappa_H': kappa_H, 'kappa_3': kappa_3, 'sens_HA': sens_HA, 'coverage': cover_g,
           'n_resp': len(G), 'n_sent': len(J), 'h_rate_open': float((J.open == 'H').mean()), 'h_rate_opus': float((J.opus == 'H').mean())},
          open(os.path.join(R4B_DIR, 'judge_gate.json'), 'w'), indent=1)
json.dump(json.load(open(os.path.join(R4B_DIR, 'judge_gate.json'))), open(os.path.join(R4B_DIR, f'judge_gate_{MODEL_TAG}.json'), 'w'), indent=1)
""")

md(r"""## 6. 학습 3,000응답 판정 (관문 통과 시에만) → `r4b/labels_train.json`""")
code(r"""
assert GATE, '관문 미달 — 학습 데이터 판정을 하지 않는다'
TR_IDS = [f'tr-{x["id"]}' for x in TRAIN_ITEMS]
LAB = judge(TR_IDS, 'train')
json.dump({k: {str(i): v for i, v in d.items()} for k, d in LAB.items()}, open(os.path.join(R4B_DIR, 'labels_train.json'), 'w'), ensure_ascii=False)
meta = json.load(open(os.path.join(R4B_DIR, 'judge_gate.json')))
meta.update({'train_coverage': len(LAB) / len(TR_IDS), 'train_n': len(LAB)})
json.dump(meta, open(os.path.join(R4B_DIR, 'judge_gate.json'), 'w'), indent=1)
print(f'학습 판정 확보 {len(LAB):,}/{len(TR_IDS):,} → 드라이브 r4b/labels_train.json 저장. 이제 런타임을 끊고 R4b-B 노트북을 실행한다.')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r4b_a_open_judge.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
