"""R4b-A(Claude) 노트북 생성: 학습 데이터(gold-large 3,000응답)를 Claude Sonnet API(배치)로 재판정한다.
관문은 R4에서 이미 받은 Sonnet·Opus 판정(cal-L 부분)으로 계산하므로 추가 비용이 없다.
결과(r4b/labels_train.json, judge_gate.json)는 R4b-B(재학습) 노트북이 읽는다."""
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

md(r"""# R4b-A (Claude) — Sonnet API로 학습 데이터 재판정

공개 LLM 두 개(Qwen 14B, 72B AWQ)는 Opus와의 일치도 관문을 넘지 못했다. 사용자 결정으로 **처음 계획대로 `claude-sonnet-5-5`(배치)** 를 쓴다.

- **관문 G**(같은 기준): cal-L에서 Opus와 κ(H 대 나머지) ≥ 0.50, 원래 환각 문장 H·A ≥ 0.70. R4에서 이미 받은 Sonnet 판정(cal-L 부분)으로 계산하므로 **추가 비용 없음**.
- 비용(추정): 3,000응답 약 $10 (R4 실측 응답당 토큰 기준). **크레딧 $15 이상** 필요.
- 결과는 드라이브 `r4b/labels_train.json`, `r4b/judge_gate.json`. 끝나면 R4b-B(`r4b_b_retrain.ipynb`)를 실행한다.
- 런타임: CPU로 충분.
""")
md("## 1. 환경 준비")
code(r"""
!pip -q install anthropic pysbd tiktoken
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'

import json, time, random, os, pickle
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
""")
md("## 2. 데이터 — 학습 3,000응답 (R1b와 같은 분할·문장 분할)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
md("## 3. 관문 G — R4의 Sonnet·Opus 판정 중 cal-L 부분으로 계산 (비용 없음)")
code(r"""
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
CAL = {f'calL-{x["id"]}': x for x in R0P['items']['calL']}
load = lambda f: {k: {int(i): v for i, v in d.items()} for k, d in json.load(open(os.path.join(R4_DIR, f))).items()}
OPUS, SON = load('labels_claude-opus-5-5.json'), load('labels_claude-sonnet-5-5.json')
ids = [c for c in SON if c in CAL and c in OPUS]
rows = [(CAL[c]['sents'][i]['y'], OPUS[c][i + 1]['label'], SON[c][i + 1]['label'], CAL[c]['task_type'])
        for c in ids for i in range(len(CAL[c]['sents']))]
J = pd.DataFrame(rows, columns=['y', 'opus', 'son', 'task'])
kappa_H = cohen_kappa_score(J.opus == 'H', J.son == 'H'); sens_HA = J[J.y == 1].son.isin(['H', 'A']).mean()
GATE = kappa_H >= 0.50 and sens_HA >= 0.70
print(f'cal-L {len(ids)}응답·{len(J):,}문장 | Opus와 κ(H) {kappa_H:.3f} (≥ 0.50) | 원래 환각 문장 H·A {sens_HA:.3f} (≥ 0.70) | H 비율 Sonnet {(J.son == "H").mean():.3f} vs Opus {(J.opus == "H").mean():.3f}')
print(f"=== 관문 G: {'통과' if GATE else '미달 — 재판정하지 않음'} ===")
json.dump({'llm': 'claude-sonnet-5-5', 'gate': bool(GATE), 'kappa_H': kappa_H, 'sens_HA': float(sens_HA), 'n_resp': len(ids), 'n_sent': len(J),
           'h_rate_open': float((J.son == 'H').mean()), 'h_rate_opus': float((J.opus == 'H').mean()), 'source': 'R4 labels (cal-L part)'},
          open(os.path.join(R4B_DIR, 'judge_gate.json'), 'w'), indent=1)
assert GATE, '관문 미달'
""")
md(r"""## 4. 학습 데이터 재판정 요청 (R4와 같은 지시문)""")
code(r"""
import anthropic

def get_key(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v
    except Exception:
        pass
    from getpass import getpass
    return getpass(f'{name}: ')

JUDGE = 'claude-sonnet-5-5'
PRICE = {'claude-sonnet-5-5': (1.0, 5.0), 'claude-opus-5-5': (2.0, 10.0)}        # 배치 가격 $/백만 토큰, 2026-10
TOK_PER_RESP = {'claude-sonnet-5-5': (1857, 280), 'claude-opus-5-5': (1821, 453)}  # R4 실측 응답당 (입력, 출력) 토큰

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

TRAIN_ITEMS = to_items(train)
ITEMS = {f'tr-{x["id"]}': x for x in TRAIN_ITEMS}
assert len(ITEMS) == len(TRAIN_ITEMS)

def sent_texts(x):
    return [x['text'][sn['a']:sn['b']].strip() for sn in x['sents']]

def user_msg(x):
    lines = '\n'.join(f'[{i + 1}] {s}' for i, s in enumerate(sent_texts(x)))
    return f"TASK: {TASK_DESC[x['task_type']]}\n\nREFERENCE:\n<<<\n{REFS[x['ref_key']]}\n>>>\n\nRESPONSE ({len(x['sents'])} numbered sentences):\n{lines}"

def params(cid):
    return {'model': JUDGE, 'max_tokens': 4096, 'system': SYSTEM, 'messages': [{'role': 'user', 'content': user_msg(ITEMS[cid])}]}

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

n_resp = len(ITEMS)
tin, tout = TOK_PER_RESP[JUDGE]
cost = n_resp * (tin * PRICE[JUDGE][0] + tout * PRICE[JUDGE][1]) / 1e6
print(f'학습 응답 {n_resp:,} (환각 {sum(x["onset"] is not None for x in TRAIN_ITEMS):,}) | 문장 {sum(len(x["sents"]) for x in TRAIN_ITEMS):,}')
print(f'{JUDGE} 배치 예상 비용 약 ${cost:.1f} (R4 실측 응답당 입력 {tin}, 출력 {tout} 토큰 기준)')
""")

md(r"""### 4-1. 3건만 바로 보내 형식 확인 (비용 수 센트)""")
code(r"""
A = anthropic.Anthropic(api_key=get_key('ANTHROPIC_API_KEY'), max_retries=2)
def call_now(cid):
    msg = A.messages.create(**params(cid))
    return ''.join(b.text for b in msg.content if getattr(b, 'type', '') == 'text'), msg.usage
for cid in list(ITEMS)[:3]:
    txt, u = call_now(cid)
    labs, ok = parse(txt, len(ITEMS[cid]['sents']))
    print(f"{cid}: 문장 {len(ITEMS[cid]['sents'])} | 파싱 {'성공' if ok else '실패'} | 입력 {u.input_tokens} / 출력 {u.output_tokens} |", {k: v['label'] for k, v in labs.items()})
""")

md(r"""### 4-2. 배치 제출 (400건씩, 배치 번호는 드라이브 `r4b/batches.json`) → 결과 받기

다시 실행하면 이미 제출한 묶음·받은 결과는 건너뛴다. 결과는 `r4b/labels_train.json`.""")
code(r"""
CHUNK = 400
BATCH_P, LAB_P = os.path.join(R4B_DIR, 'batches.json'), os.path.join(R4B_DIR, 'labels_train.json')
if os.path.exists(LAB_P):
    LABELS = {k: {int(i): v for i, v in d.items()} for k, d in json.load(open(LAB_P)).items()}
    print(f'저장된 재판정 {len(LABELS):,}건')
else:
    BATCHES = json.load(open(BATCH_P)) if os.path.exists(BATCH_P) else {'chunks': []}
    sent = {c for ch in BATCHES['chunks'] for c in ch['cids']}
    todo = [c for c in ITEMS if c not in sent]
    print(f'이미 제출 {len(sent):,} | 이번에 제출 {len(todo):,}')
    for k in range(0, len(todo), CHUNK):
        part = todo[k:k + CHUNK]
        try:
            b = A.messages.batches.create(requests=[{'custom_id': c, 'params': params(c)} for c in part])
        except Exception as e:
            print(f'  ✗ 제출 실패 ({len(part)}건): {type(e).__name__}: {str(e)[:300]}'); raise
        BATCHES['chunks'].append({'id': b.id, 'cids': part}); json.dump(BATCHES, open(BATCH_P, 'w'))
        print(f'  제출 {b.id} ({len(part)}건)', flush=True)
    LABELS, bad, tin, tout = {}, [], 0, 0
    for ch in BATCHES['chunks']:
        while True:
            b = A.messages.batches.retrieve(ch['id']); rc = b.request_counts
            print(f'  {time.strftime("%H:%M:%S")} {ch["id"][-8:]} {b.processing_status} | 처리 중 {rc.processing}, 성공 {rc.succeeded}, 오류 {rc.errored}', flush=True)
            if b.processing_status == 'ended': break
            time.sleep(60)
        for r in A.messages.batches.results(ch['id']):
            if r.result.type == 'succeeded':
                msg = r.result.message; tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
                labs, ok = parse(''.join(x.text for x in msg.content if getattr(x, 'type', '') == 'text'), len(ITEMS[r.custom_id]['sents']))
                if ok: LABELS[r.custom_id] = labs; continue
            bad.append(r.custom_id)
    for cid in bad[:100]:                                    # 실패분은 바로 요청 1회 (크레딧 부족 등으로 실패하면 원래 정답 유지)
        try:
            txt, u = call_now(cid); labs, ok = parse(txt, len(ITEMS[cid]['sents']))
            if ok: LABELS[cid] = labs
        except Exception as e:
            print(f'  {cid}: {type(e).__name__} {str(e)[:100]}'); break
    USAGE = {'input_tokens': tin, 'output_tokens': tout, 'cost_usd': tin / 1e6 * PRICE[JUDGE][0] + tout / 1e6 * PRICE[JUDGE][1]}
    print(f'재판정 확보 {len(LABELS):,}/{len(ITEMS):,} | 배치 사용량 입력 {tin/1e6:.2f}M, 출력 {tout/1e6:.2f}M → 약 ${USAGE["cost_usd"]:.1f}')
    json.dump({k: {str(i): v for i, v in d.items()} for k, d in LABELS.items()}, open(LAB_P, 'w'), ensure_ascii=False)
    json.dump(USAGE, open(os.path.join(R4B_DIR, 'usage_train.json'), 'w'))
""")


nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r4b_a_claude_judge.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
