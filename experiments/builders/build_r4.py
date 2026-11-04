"""R4 노트북 생성: RAGTruth 정답 누락을 LLM 재판정으로 보정하고, 엄격 재현율 상한(약 0.50)이
정답 누락 때문인지 다시 잰다. GPU 불필요(저장된 점수만 사용). Claude API(Message Batches) 사용."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R4 — 정답 재검증: 엄격 재현율 상한은 정답 누락 때문인가

## 왜 하나

경보 규칙(R2, 52개), 학습 목표(R1b), 판정기 크기(R1c, 0.08B → 0.44B) 어느 쪽을 바꿔도 dev-L ARL₀=100 엄격 재현율은 약 0.50에서 막혔다.
반면 판별력(응답 AUROC 0.78 → 0.83)과 서비스 조건 재현율(0.33 → 0.49)은 판정기를 키우면 올랐다. 엄격 지표만 안 움직인다.

엄격 지표는 RAGTruth 정답에 가장 민감하다. **첫 경보가 정답 환각 시작보다 앞이면 실패**이고, 멀쩡한 응답(정답 표시 없음)의 경보는 오경보 예산을 쓴다.
R2 사례 검토(50건)에서 이른 경보의 30~47%, 멀쩡한 응답 경보의 15~35%가 **정답에 표시되지 않은 환각**이었다. 정답 누락은 두 군데서 재현율을 깎는다.
1. 환각 응답: 표시 안 된 앞 환각에 울린 경보가 "발생 전 경보 = 실패"로 집계된다.
2. 멀쩡한 응답: 사실은 환각인 응답이 오경보 예산을 써서 임계값을 올린다.

50건으로는 크기를 알 수 없다. 이번에는 **cal-L·dev-L 3,216응답의 모든 문장**을 다시 판정해 상한을 다시 잰다.

## 무엇을 하나

1. **재판정자**: `claude-opus-5-5`가 응답마다 근거 문서와 번호 붙인 문장 전체를 보고 문장마다 S(근거 있음) / H(환각) / A(애매)를 단다. RAGTruth의 환각 정의(명백·미묘한 모순, 명백·미묘한 근거 없는 정보)를 그대로 준다.
   - 재판정자는 **정답도, 판정기 점수도 보지 않는다.** 판정기가 경보를 울린 문장만 골라 보면 그 판정기에 유리하게 기울기 때문에 모든 응답의 모든 문장을 본다.
   - 일치도 확인용으로 400응답(cal-L 200 + dev-L 200)은 `claude-sonnet-5-5`도 독립적으로 판정한다.
   - Message Batches API(50% 할인)로 보낸다. 끊겨도 배치 번호를 드라이브에 저장해 이어서 받는다.
2. **보정 정답** (결과 보기 전 고정)
   - **C1(주)**: 원래 정답 ∪ 재판정 H. 원래 정답은 **지우지 않는다**(추가만).
   - **C2(민감도)**: 원래 정답 ∪ 재판정 H ∪ A.
   - 추가된 문장의 시작 토큰: 재판정자가 적은 환각 구절을 문장 안에서 찾으면 그 첫 토큰, 못 찾으면 문장 첫 토큰. 응답의 환각 시작 = 원래 시작과 추가된 시작 중 이른 것.
3. **다시 잰다**: R0·R1b·R1c의 저장된 점수(B, E2, R1b V0b·V1b, mini, base, large)를 그대로 쓰고 정답만 바꾼다.
   - 원래: cal-L 원래 정답으로 보정 → dev-L 원래 정답으로 평가 (R1c와 같아야 함)
   - 인정 효과: 원래 임계값 그대로 → dev-L 보정 정답으로 평가 (발생 전 경보가 탐지로 바뀌는 효과만)
   - **보정 전체**: cal-L 보정 정답으로 다시 보정 → dev-L 보정 정답으로 평가 (주 지표)
4. **위약 대조(placebo)**: 정답을 **추가하기만 해도** 엄격 재현율은 기계적으로 오른다(환각 시작이 앞당겨지면 첫 경보가 시작 이후일 확률이 커지고, 멀쩡한 응답이 줄면 임계값이 내려간다). 무작위 점수·무작위 라벨로 한 smoke test에서도 +0.11이 나왔다.
   그래서 재판정이 추가한 문장 수를 **응답 단위로 그대로 유지한 채 같은 과제의 다른 응답의 무작위 문장으로 옮긴** 가짜 보정 정답을 20번 만들어 같은 계산을 하고, 그만큼을 빼고 본다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **R4-0 재판정 타당성**: (a) 원래 정답에서 환각인 문장에 재판정자가 H 또는 A를 단 비율 ≥ 0.70, 그리고 (b) 400응답에서 Opus–Sonnet 문장 단위 일치도(H 대 나머지, Cohen κ) ≥ 0.50. 미달이면 R4-1·R4-2는 "LLM 정답 기준 참고값"으로만 보고하고 상한 주장을 하지 않는다.
- **R4-1 정답 상한(주)**: B 문장 끝의 dev-L ARL₀=100 엄격 재현율에서 **(C1 − 원래) − 위약 평균 이득 ≥ +0.05**, 그리고 C1 이득이 위약 20회 모두보다 큼(p < 0.05).
- **R4-2 가려진 용량 효과**: C1에서 4토큰 판정의 base 시드 평균 − mini 시드 평균 ≥ +0.03 (R1c 원래 정답에서는 +0.027).
- **보고(판정 없음)**: C1 이득의 부트스트랩 구간, R1b V1b 4토큰·large·E2의 변화, 인정 효과와 재보정 효과 분해, 발생 전 경보율 변화, C2, 서비스 10% 재현율, 응답·문장 AUROC, 판정기 순위가 유지되는지(Kendall τ), 과제별, 멀쩡 → 환각으로 바뀐 응답 수.
- 해석 규칙: R4-0 충족 + R4-1 충족이면 "상한의 상당 부분은 정답 누락 → 이후 주 평가에 보정 정답을 함께 보고하고, 학습 데이터 정제(R4b)로 진행". R4-1 미달이면 "정답 누락은 상한의 작은 부분 → 상한은 과제 자체(진짜 오경보) 쪽"으로 기록.

> **런타임: CPU로 충분**(GPU 불필요). 드라이브에 `r0/`(r0_items.pkl, r0_scores.npz), `r1b/`, `r1c/`의 scores.npy가 있어야 한다.
> **API 키**: 코랩 왼쪽 열쇠 아이콘(보안 비밀)에 `ANTHROPIC_API_KEY`를 등록하거나 실행 중 입력한다.
> **비용(추정, 2026-10 배치 가격)**: Opus 3,216응답 약 $15~20 + Sonnet 400응답 약 $1~2. 4절에서 실제 입력 길이로 다시 계산해 보여 준다.
> 배치는 보통 1시간 안에 끝나지만 최대 24시간 걸릴 수 있다. 6절은 끝날 때까지 기다리며, 코랩이 끊기면 6절부터 다시 실행하면 된다(재제출하지 않음).
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install anthropic
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'

import json, time, random, os, re, pickle, copy
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.stats import kendalltau

USE_DRIVE = True
OUT_DIR = '/content/drive/MyDrive/selfproject' if USE_DRIVE else '.'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
R0_DIR, R1B_DIR, R1C_DIR = (os.path.join(OUT_DIR, d) for d in ['r0', 'r1b', 'r1c'])
R4_DIR = os.path.join(OUT_DIR, 'r4'); os.makedirs(R4_DIR, exist_ok=True)

def get_key(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v
    except Exception:
        pass
    from getpass import getpass
    return getpass(f'{name}: ')

JUDGE, JUDGE2 = 'claude-opus-5-5', 'claude-sonnet-5-5'
PRICE = {JUDGE: (2.0, 10.0), JUDGE2: (1.0, 5.0)}      # 배치 가격 $/백만 토큰 (입력, 출력), 2026-10 기준
""")

md("## 2. 데이터 — R0 평가 세트(cal-L, dev-L)와 저장된 점수")
code(r"""
BASE = 'RAGTruth/dataset'
RESP = {str(json.loads(l)['id']): json.loads(l) for l in open(f'{BASE}/response.jsonl')}
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}

def build_reference(src):
    si, tt = src['source_info'], src['task_type']
    if tt == 'QA':
        return f"Question: {si.get('question','')}\n{si.get('passages','')}"
    if tt == 'Summary':
        return si if isinstance(si, str) else json.dumps(si, ensure_ascii=False)
    return '\n'.join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}"
                     for k, v in (si.items() if isinstance(si, dict) else []))

R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
CAL, DEV = R0P['items']['calL'], R0P['items']['devL']
for x in CAL + DEV:
    r = RESP[x['id']]
    assert r['split'] == 'train', 'test 분할 응답이 섞임'            # test 분할은 열지 않는다
    x['text'] = r['response']
    assert x['te'][-1] == len(x['text']), f"토큰 오프셋 불일치 {x['id']}"
TASKS = ['QA', 'Summary', 'Data2txt']
print(f"cal-L {len(CAL)} (환각 {sum(x['onset'] is not None for x in CAL)}) | dev-L {len(DEV)} (환각 {sum(x['onset'] is not None for x in DEV)}) | "
      f"문장 {sum(len(x['sents']) for x in CAL + DEV):,}")

SC = dict(np.load(os.path.join(R0_DIR, 'r0_scores.npz')))
SEEDS = [42, 7, 2026]
FAM = {'B': ['B'], 'E2': ['E2']}
for d, pre, names in [(R1B_DIR, 'R1b', ['V0b', 'V1b']), (R1C_DIR, '', ['mini', 'base', 'large'])]:
    for n in names:
        for s in SEEDS:
            p = os.path.join(d, f'{n}_s{s}', 'scores.npy')
            if os.path.exists(p):
                SC[f'{pre}{n}_s{s}'] = np.load(p); FAM.setdefault(f'{pre}{n}', []).append(f'{pre}{n}_s{s}')
print({k: v for k, v in FAM.items()})
assert all(k in FAM for k in ['R1bV1b', 'mini', 'base']), 'R1b·R1c 점수가 드라이브에 있어야 한다'
""")

md("## 3. 평가 함수 (R1c와 같음)")
code(r"""
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

def calibrate(items, scorer, arl0=100):
    clean = [x for x in items if x['onset'] is None]
    n_tok = sum(x['n_tok'] for x in clean)
    sc = np.sort(np.concatenate([[s for _, s in series(x, scorer)] for x in clean]))[::-1]
    n_alarm = max(1, int(n_tok / arl0))
    return float(sc[n_alarm - 1]) + 1e-6 if n_alarm <= len(sc) else float(sc[-1])

def calibrate_flag(items, scorer, rate=0.10):
    mx = np.array([max([s for _, s in series(x, scorer)] or [-1e9]) for x in items if x['onset'] is None])
    return float(np.quantile(mx, 1 - rate)) + 1e-6

def first_alarm(x, scorer, thr):
    return next((t for t, s in series(x, scorer) if s >= thr), None)

def evaluate(items, scorer, thr):
    hall = [x for x in items if x['onset'] is not None]
    det, pre, d = 0, 0, []
    for x in hall:
        f = first_alarm(x, scorer, thr)
        if f is not None and f >= x['onset']: det += 1; d.append(f - x['onset'])
        elif f is not None: pre += 1
    return {'recall': det / len(hall), 'delay_med': float(np.median(d)) if d else np.nan,
            'pre_onset': pre / len(hall), 'n_hall': len(hall)}

def resp_auc(items, m):
    return roc_auc_score([int(x['onset'] is not None) for x in items],
                         [max([SC[m][q] for _, q in x['pts2']] or [-1e9]) for x in items])

def sent_auc(items, m):
    ys, ss = [], []
    for x in items:
        d = dict(x['pts2'])
        for sn in x['sents']:
            if sn['t1'] in d: ys.append(sn['y']); ss.append(SC[m][d[sn['t1']]])
    return roc_auc_score(ys, ss)

for sc, ref in [('B|end', 0.5007), ('B|4', 0.4809), ('E2|4', 0.4506)]:
    got = evaluate(DEV, sc, calibrate(CAL, sc))['recall']
    print(f'{sc:6} R0 {ref:.4f} | 지금 {got:.4f} | {"일치" if abs(got - ref) <= 0.002 else "불일치 — 확인 필요"}')
    assert abs(got - ref) <= 0.002
""")

md(r"""## 4. 재판정 요청 만들기와 비용 추정

재판정자에게는 과제 종류, 근거 문서(판정기와 같은 `build_reference`), 번호 붙인 문장만 준다. 정답·판정기 점수는 주지 않는다.""")
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
    ref = build_reference(sources[RESP[x['id']]['source_id']])
    lines = '\n'.join(f'[{i + 1}] {s}' for i, s in enumerate(sent_texts(x)))
    return f"TASK: {TASK_DESC[x['task_type']]}\n\nREFERENCE:\n<<<\n{ref}\n>>>\n\nRESPONSE ({len(x['sents'])} numbered sentences):\n{lines}"

ITEMS = {f'calL-{x["id"]}': x for x in CAL}
ITEMS.update({f'devL-{x["id"]}': x for x in DEV})
assert len(ITEMS) == len(CAL) + len(DEV)
_r = random.Random(4)
SUB = [f'calL-{x["id"]}' for x in _r.sample(CAL, 200)] + [f'devL-{x["id"]}' for x in _r.sample(DEV, 200)]
JOBS = {JUDGE: list(ITEMS), JUDGE2: SUB}

USE_TEMP = False                              # 이 모델·SDK는 temperature를 받지 않음(10/14 확인) → 기본값으로 판정
def params(model, cid):
    p = {'model': model, 'max_tokens': 4096, 'system': SYSTEM, 'messages': [{'role': 'user', 'content': user_msg(ITEMS[cid])}]}
    if USE_TEMP: p['temperature'] = 0
    return p

est = {}
for m, ids in JOBS.items():
    tin = sum(len(SYSTEM) + len(user_msg(ITEMS[c])) for c in ids) / 3.6
    tout = sum(12 * len(ITEMS[c]['sents']) + 40 for c in ids) * 1.6        # H·A 문장은 구절·이유가 붙음 (여유 1.6배)
    est[m] = (tin, tout, tin / 1e6 * PRICE[m][0] + tout / 1e6 * PRICE[m][1])
    print(f'{m}: 요청 {len(ids):,} | 입력 약 {tin/1e6:.2f}M 토큰, 출력 약 {tout/1e6:.2f}M 토큰 → 약 ${est[m][2]:.1f} (배치 가격)')
print(f'합계 약 ${sum(v[2] for v in est.values()):.1f}')
print('\n예시 요청:\n' + user_msg(ITEMS[SUB[0]])[:1500])
""")

md("### 4-1. 3건만 바로 보내 응답 형식 확인 (비용 수 센트)")
code(r"""
import anthropic
A = anthropic.Anthropic(api_key=get_key('ANTHROPIC_API_KEY'), max_retries=2)

def parse(text, n):
    # 응답 JSON에서 문장별 판정을 꺼낸다. 문장 수가 맞지 않으면 ok=False
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        labs = {}
        for e in d.get('sentences', []):
            i, lab = int(e.get('i', 0)), str(e.get('label', '')).strip().upper()[:1]
            if 1 <= i <= n and lab in 'SHA' and lab:
                labs[i] = {'label': lab, 'type': e.get('type', ''), 'phrase': e.get('phrase', '') or '', 'reason': e.get('reason', '')}
        return labs, len(labs) == n
    except Exception:
        return {}, False

def call_now(model, cid):
    global USE_TEMP
    try:
        msg = A.messages.create(**params(model, cid))
    except (anthropic.BadRequestError, TypeError) as e:
        if USE_TEMP and 'temperature' in str(e).lower():
            print('  temperature 미지원 → 기본값으로'); USE_TEMP = False
            msg = A.messages.create(**params(model, cid))
        else:
            raise
    return ''.join(b.text for b in msg.content if getattr(b, 'type', '') == 'text'), msg.usage

for cid in SUB[:3]:
    txt, u = call_now(JUDGE, cid)
    labs, ok = parse(txt, len(ITEMS[cid]['sents']))
    print(f"{cid}: 문장 {len(ITEMS[cid]['sents'])} | 파싱 {'성공' if ok else '실패'} | 입력 {u.input_tokens} / 출력 {u.output_tokens} 토큰 |",
          {k: v['label'] for k, v in labs.items()})
print('\n마지막 응답 원문:\n' + txt[:800])
""")

md(r"""## 5. 배치 제출 (400건씩 나눠 제출 — 배치 번호는 제출할 때마다 드라이브 `r4/batches.json`에 저장)

한 번에 3,216건(약 50MB)을 보내면 요청이 너무 커서 실패할 수 있어 나눠 보낸다. 중간에 실패해도 다시 실행하면 **이미 제출한 묶음은 건너뛰고** 남은 것만 보낸다.""")
code(r"""
CHUNK = 400
BATCH_P = os.path.join(R4_DIR, 'batches.json')
BATCHES = json.load(open(BATCH_P)) if os.path.exists(BATCH_P) else {}
for m, ids in JOBS.items():
    BATCHES.setdefault(m, {'chunks': [], 'use_temp': USE_TEMP})
    sent = {c for ch in BATCHES[m]['chunks'] for c in ch['cids']}
    todo = [c for c in ids if c not in sent]
    print(f'{m}: 전체 {len(ids):,} | 이미 제출 {len(sent):,} | 이번에 제출 {len(todo):,}')
    for k in range(0, len(todo), CHUNK):
        part = todo[k:k + CHUNK]
        try:
            b = A.messages.batches.create(requests=[{'custom_id': c, 'params': params(m, c)} for c in part])
        except Exception as e:
            print(f'  ✗ 제출 실패 ({len(part)}건): {type(e).__name__}: {str(e)[:300]}')
            print('  → 이 셀을 다시 실행하면 남은 것만 보낸다. 같은 오류가 반복되면 오류 메시지를 알려 줄 것'); break
        BATCHES[m]['chunks'].append({'id': b.id, 'cids': part, 'submitted': time.strftime('%Y-%m-%d %H:%M:%S')})
        json.dump(BATCHES, open(BATCH_P, 'w'))
        print(f'  제출 {b.id} ({len(part)}건)', flush=True)
print('\n제출 완료:', {m: sum(len(ch['cids']) for ch in v['chunks']) for m, v in BATCHES.items()})
""")

md(r"""## 6. 결과 받기 (끝날 때까지 기다림) → 파싱 → 실패분은 바로 다시 요청

결과는 `r4/labels_<모델>.json`에 저장된다. 코랩이 끊기면 1·2·3·4절, 4-1절의 `A`·`parse` 정의 후 이 절부터 다시 실행한다.""")
code(r"""
def wait(bid, every=60):
    while True:
        b = A.messages.batches.retrieve(bid)
        rc = b.request_counts
        print(f'  {time.strftime("%H:%M:%S")} {b.processing_status} | 처리 중 {rc.processing}, 성공 {rc.succeeded}, 오류 {rc.errored}, 만료 {rc.expired}', flush=True)
        if b.processing_status == 'ended': return
        time.sleep(every)

LABELS, USAGE = {}, {}
for m, ids in JOBS.items():
    lp = os.path.join(R4_DIR, f'labels_{m}.json')
    if os.path.exists(lp):
        LABELS[m] = {k: {int(i): v for i, v in d.items()} for k, d in json.load(open(lp)).items()}
        print(f'{m}: 저장된 결과 {len(LABELS[m]):,}건'); continue
    USE_TEMP = BATCHES[m].get('use_temp', USE_TEMP)
    assert sum(len(ch['cids']) for ch in BATCHES[m]['chunks']) == len(ids), f'{m}: 아직 다 제출되지 않음 — 5절을 다시 실행'
    got, bad, tin, tout = {}, [], 0, 0
    def all_results():
        for ch in BATCHES[m]['chunks']:
            wait(ch['id'])
            yield from A.messages.batches.results(ch['id'])
    for r in all_results():
        cid = r.custom_id
        if r.result.type == 'succeeded':
            msg = r.result.message
            tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
            labs, ok = parse(''.join(b.text for b in msg.content if getattr(b, 'type', '') == 'text'), len(ITEMS[cid]['sents']))
            if ok: got[cid] = labs; continue
        bad.append(cid)
    print(f'{m}: 배치 성공·파싱 {len(got):,} | 다시 요청 {len(bad)}')
    for cid in bad:                                     # 실패분: 바로 요청(최대 2회)
        for _ in range(2):
            try:
                txt, u = call_now(m, cid); tin += u.input_tokens * 2; tout += u.output_tokens * 2   # 실시간은 배치의 2배 가격
                labs, ok = parse(txt, len(ITEMS[cid]['sents']))
                if ok: got[cid] = labs; break
            except Exception as e:
                print(f'  {cid}: {type(e).__name__} {str(e)[:120]}')
    missing = [c for c in ids if c not in got]
    print(f'{m}: 최종 {len(got):,}/{len(ids):,} (판정 없음 {len(missing)} → 원래 정답 유지)')
    USAGE[m] = {'input_tokens': tin, 'output_tokens': tout, 'cost_usd': tin / 1e6 * PRICE[m][0] + tout / 1e6 * PRICE[m][1], 'missing': missing}
    print(f'   사용량: 입력 {tin/1e6:.2f}M, 출력 {tout/1e6:.2f}M → 약 ${USAGE[m]["cost_usd"]:.1f}')
    json.dump({k: {str(i): v for i, v in d.items()} for k, d in got.items()}, open(lp, 'w'), ensure_ascii=False)
    json.dump(USAGE[m], open(os.path.join(R4_DIR, f'usage_{m}.json'), 'w'), indent=1)
    LABELS[m] = got
""")

md("## 7. R4-0 재판정 타당성")
code(r"""
def flat(m, ids=None):
    # (응답 키, 문장 번호, 원래 정답 y, 재판정 라벨, 과제)
    rows = []
    for cid, labs in LABELS[m].items():
        if ids is not None and cid not in ids: continue
        x = ITEMS[cid]
        for i, sn in enumerate(x['sents']):
            rows.append((cid, i + 1, sn['y'], labs[i + 1]['label'], x['task_type']))
    return pd.DataFrame(rows, columns=['cid', 'i', 'y', 'lab', 'task'])

F1 = flat(JUDGE)
pos, neg = F1[F1.y == 1], F1[F1.y == 0]
sens_H, sens_HA = (pos.lab == 'H').mean(), pos.lab.isin(['H', 'A']).mean()
print(f'원래 정답 환각 문장 {len(pos):,}개: 재판정 H {sens_H:.3f}, H 또는 A {sens_HA:.3f}')
print(f'원래 정답 멀쩡한 문장 {len(neg):,}개: 재판정 H {(neg.lab == "H").mean():.3f}, A {(neg.lab == "A").mean():.3f}')
print('\n과제별 (원래 환각 문장의 H·A 비율 / 원래 멀쩡한 문장의 H 비율)')
for tt in TASKS:
    p, n = pos[pos.task == tt], neg[neg.task == tt]
    print(f'  {tt:9} {p.lab.isin(["H","A"]).mean():.3f} (n={len(p)}) / {(n.lab == "H").mean():.3f} (n={len(n)})')

both = sorted(set(LABELS[JUDGE]) & set(LABELS[JUDGE2]) & set(SUB))
F2s = flat(JUDGE2, set(both)).set_index(['cid', 'i']); F1s = F1[F1.cid.isin(both)].set_index(['cid', 'i'])
J = F1s.join(F2s[['lab']], rsuffix='2')
kappa_H = cohen_kappa_score(J.lab == 'H', J.lab2 == 'H')
kappa_3 = cohen_kappa_score(J.lab, J.lab2)
sens2 = J[J.y == 1].lab2.isin(['H', 'A']).mean()
print(f'\nOpus–Sonnet ({len(both)}응답, 문장 {len(J):,}): κ(H 대 나머지) {kappa_H:.3f}, κ(3분류) {kappa_3:.3f} | Sonnet의 원래 환각 문장 H·A {sens2:.3f}')
print(pd.crosstab(J.lab, J.lab2, rownames=['Opus'], colnames=['Sonnet']))

R4_0 = sens_HA >= 0.70 and kappa_H >= 0.50
print(f"\n=== R4-0: 원래 환각 문장 H·A {sens_HA:.3f} ≥ 0.70, κ {kappa_H:.3f} ≥ 0.50 → {'충족' if R4_0 else '미달 — 이후 수치는 LLM 정답 기준 참고값'} ===")
""")

md("## 8. 보정 정답 만들기 (C1 주, C2 민감도)")
code(r"""
def content_starts(x):
    t = x['text']
    cs = [a + (len(t[a:b]) - len(t[a:b].lstrip())) for a, b in zip(x['ts'], x['te'])]
    return [c if c < b else a for c, a, b in zip(cs, x['ts'], x['te'])]

def locate(x, sn, phrase, cs):
    # 구절의 첫 토큰. 못 찾으면 문장 첫 토큰
    seg = x['text'][sn['a']:sn['b']]
    ph = (phrase or '').strip()
    k = seg.find(ph) if ph else -1
    if k < 0 and ph: k = seg.lower().find(ph.lower())
    if k < 0: return sn['t0'], False
    a, b = sn['a'] + k, sn['a'] + k + len(ph)
    for t in range(sn['t0'], sn['t1'] + 1):
        if min(x['te'][t], b) > max(cs[t], a): return t, True
    return sn['t0'], False

def corrected(items, split, add):
    out, stat = [], {'added_sent': 0, 'flip_clean': 0, 'earlier_onset': 0, 'phrase_found': 0, 'no_label': 0}
    for x in items:
        cid = f'{split}-{x["id"]}'; labs = LABELS[JUDGE].get(cid)
        y = copy.copy(x); y['sents'] = [dict(sn) for sn in x['sents']]; y['added'] = []
        if labs is None:
            stat['no_label'] += 1; out.append(y); continue
        cs, cand = None, []
        for i, sn in enumerate(y['sents']):
            if sn['y'] == 0 and labs[i + 1]['label'] in add:
                cs = cs or content_starts(x)
                t, found = locate(x, sn, labs[i + 1]['phrase'], cs)
                sn['y'] = 1; cand.append(t); y['added'].append(i + 1)
                stat['added_sent'] += 1; stat['phrase_found'] += found
        if cand:
            new = min(cand + ([x['onset']] if x['onset'] is not None else []))
            if x['onset'] is None: stat['flip_clean'] += 1
            elif new < x['onset']: stat['earlier_onset'] += 1
            y['onset'] = new
        out.append(y)
    return out, stat

V = {'orig': (CAL, DEV)}
STAT = {}
for name, add in [('C1', {'H'}), ('C2', {'H', 'A'})]:
    cal_c, s1 = corrected(CAL, 'calL', add); dev_c, s2 = corrected(DEV, 'devL', add)
    V[name] = (cal_c, dev_c); STAT[name] = {'calL': s1, 'devL': s2}
    for sp, its, s in [('cal-L', cal_c, s1), ('dev-L', dev_c, s2)]:
        nh = sum(x['onset'] is not None for x in its)
        print(f"{name} {sp}: 환각 응답 {nh} (원래 {sum(x['onset'] is not None for x in (CAL if sp == 'cal-L' else DEV))}) | "
              f"추가 문장 {s['added_sent']} (구절 위치 찾음 {s['phrase_found']}) | 멀쩡 → 환각 {s['flip_clean']} | 시작이 앞당겨짐 {s['earlier_onset']} | 판정 없음 {s['no_label']}")

# 위약 대조: C1이 응답마다 추가한 문장 수(k)를 같은 과제 안에서 응답끼리 섞어 다른 응답에 배정하고,
# 그 응답의 원래 멀쩡한 문장 k개를 무작위로 골라 환각으로 바꾼다. 문장 안 시작 위치는 실제 추가분의 상대 위치 분포에서 뽑는다.
FRAC = []
for its in V['C1']:
    for x in its:
        for i in x['added']:
            sn = x['sents'][i - 1]
            if x['onset'] is not None and sn['t0'] <= x['onset'] <= sn['t1']:
                FRAC.append((x['onset'] - sn['t0']) / max(1, sn['t1'] - sn['t0']))
FRAC = FRAC or [0.0]

def placebo(items, split, rng):
    ks = {x['id']: len(c['added']) for x, c in zip(items, V['C1'][0 if split == 'calL' else 1])}
    out, n_flip = [copy.copy(x) for x in items], 0
    for tt in TASKS:
        idx = [j for j, x in enumerate(items) if x['task_type'] == tt]
        kv = [ks[items[j]['id']] for j in idx]; rng.shuffle(kv)
        for j, k in zip(idx, kv):
            if k == 0: continue
            x = items[j]; negs = [i for i, sn in enumerate(x['sents']) if sn['y'] == 0]
            pick = rng.sample(negs, min(k, len(negs)))
            if not pick: continue
            ts_ = [x['sents'][i]['t0'] + int(round(rng.choice(FRAC) * (x['sents'][i]['t1'] - x['sents'][i]['t0']))) for i in pick]
            new = min(ts_ + ([x['onset']] if x['onset'] is not None else []))
            n_flip += x['onset'] is None
            out[j]['onset'] = new
    return out, n_flip

PLACEBO, PSTAT = [], []
for r in range(20):
    rng = random.Random(1000 + r)
    cl, f1 = placebo(CAL, 'calL', rng); dv, f2 = placebo(DEV, 'devL', rng)
    PLACEBO.append((cl, dv)); PSTAT.append({'flip_calL': f1, 'flip_devL': f2})
print(f"위약 20회: dev-L 멀쩡 → 환각 평균 {np.mean([p['flip_devL'] for p in PSTAT]):.0f} (실제 C1 {STAT['C1']['devL']['flip_clean']}) | "
      f"시작 위치 상대값 표본 {len(FRAC)}개")
""")

md("## 9. 다시 재기 — 원래 / 인정 효과(임계값 그대로) / 보정 전체(재보정)")
code(r"""
KINDS = ['end', '4']
MODELS = [m for f in FAM.values() for m in f]
rows = []
for m in MODELS:
    for k in KINDS:
        sc = f'{m}|{k}'
        thr_o = calibrate(CAL, sc)
        r_o = evaluate(DEV, sc, thr_o)
        r_o10 = evaluate(DEV, sc, calibrate_flag(CAL, sc))
        row = {'model': m, 'kind': k, 'orig': r_o['recall'], 'orig_delay': r_o['delay_med'], 'orig_pre': r_o['pre_onset'], 'orig_svc': r_o10['recall']}
        for c in ['C1', 'C2']:
            cal_c, dev_c = V[c]
            r_cr = evaluate(dev_c, sc, thr_o)
            thr_c = calibrate(cal_c, sc); r_c = evaluate(dev_c, sc, thr_c)
            row.update({f'{c}_credit': r_cr['recall'], c: r_c['recall'], f'{c}_delay': r_c['delay_med'], f'{c}_pre': r_c['pre_onset'],
                        f'{c}_svc': evaluate(dev_c, sc, calibrate_flag(cal_c, sc))['recall'], f'{c}_thr_shift': thr_c - thr_o})
        rows.append(row)
    for c, (_, dev_c) in [('orig', V['orig']), ('C1', V['C1'])]:
        rows[-1][f'{c}_resp_auc'] = resp_auc(dev_c, m); rows[-1][f'{c}_sent_auc'] = sent_auc(dev_c, m)
res = pd.DataFrame(rows)

def fam_mean(fam, k, col):
    return float(np.mean([res[(res.model == m) & (res.kind == k)][col].iloc[0] for m in FAM[fam]]))
agg = pd.DataFrame([{'family': f, 'n_seeds': len(FAM[f]), 'kind': k,
                     **{c: fam_mean(f, k, c) for c in ['orig', 'C1_credit', 'C1', 'C2', 'orig_delay', 'C1_delay', 'orig_pre', 'C1_pre', 'orig_svc', 'C1_svc']}}
                    for f in FAM for k in KINDS])
agg['gain_C1'] = agg.C1 - agg.orig
pd.set_option('display.width', 250)
print(agg.round(3).to_string(index=False))
assert abs(res[(res.model == 'B') & (res.kind == 'end')].orig.iloc[0] - 0.5007) <= 0.002
""")

md("## 10. 사전 기준 판정")
code(r"""
def per_item(items, sc, thr):
    # 응답마다 (환각 응답인가, 엄격 탐지했나)
    h = np.array([x['onset'] is not None for x in items])
    d = np.array([x['onset'] is not None and (lambda f: f is not None and f >= x['onset'])(first_alarm(x, sc, thr)) for x in items])
    return h, d

def boot_gain(sc, c='C1', n=2000, seed=0):
    ho, do = per_item(DEV, sc, calibrate(CAL, sc))
    cal_c, dev_c = V[c]; hc, dc = per_item(dev_c, sc, calibrate(cal_c, sc))
    rng = np.random.default_rng(seed); N = len(DEV); g = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        g.append(dc[i].sum() / hc[i].sum() - do[i].sum() / ho[i].sum())
    return dc.sum() / hc.sum() - do.sum() / ho.sum(), *np.percentile(g, [2.5, 97.5])

g, lo, hi = boot_gain('B|end')
PL = {}
for sc in ['B|end', 'B|4', 'R1bV1b_s42|4', 'base_s42|4', 'large_s42|4']:
    if sc.split('|')[0] in SC:
        o = evaluate(DEV, sc, calibrate(CAL, sc))['recall']
        PL[sc] = [evaluate(dv, sc, calibrate(cl, sc))['recall'] - o for cl, dv in PLACEBO]
pm = float(np.mean(PL['B|end']))
R4_1 = g - pm >= 0.05 and g > max(PL['B|end'])
print(f"B 문장 끝: C1 이득 {g:+.3f} [{lo:+.3f}, {hi:+.3f}] | 위약 이득 평균 {pm:+.3f} (범위 {min(PL['B|end']):+.3f} ~ {max(PL['B|end']):+.3f})")
print(f"=== R4-1 정답 상한: 위약을 뺀 이득 {g - pm:+.3f} (≥ +0.05), C1 > 위약 최대 → {'충족' if R4_1 else '미달'} ===")

cap_o = fam_mean('base', '4', 'orig') - fam_mean('mini', '4', 'orig')
cap_c = fam_mean('base', '4', 'C1') - fam_mean('mini', '4', 'C1')
R4_2 = cap_c >= 0.03
print(f"=== R4-2 가려진 용량 효과: base − mini (4토큰) 원래 {cap_o:+.3f} → C1 {cap_c:+.3f} (≥ +0.03) → {'충족' if R4_2 else '미달'} ===")
if not R4_0:
    print('※ R4-0 미달: 위 두 판정은 LLM 정답 기준 참고값')

print('\n(보고) 다른 점수기의 C1 이득 [부트스트랩 95%]')
BOOT = {'B|end': (g, lo, hi)}
for sc in ['B|4', 'E2|4', 'R1bV1b_s42|4', 'base_s42|4', 'large_s42|4']:
    if sc.split('|')[0] in SC:
        BOOT[sc] = boot_gain(sc)
        plc = f' | 위약 {np.mean(PL[sc]):+.3f} → 순이득 {BOOT[sc][0] - np.mean(PL[sc]):+.3f}' if sc in PL else ''
        print(f'  {sc:14} {BOOT[sc][0]:+.3f} [{BOOT[sc][1]:+.3f}, {BOOT[sc][2]:+.3f}]{plc}')

sub = res[res.kind == '4']
tau_C1 = kendalltau(sub.orig, sub.C1)
print(f'\n(보고) 4토큰 점수기 {len(sub)}개의 재현율 순위 유지: Kendall τ(원래, C1) = {tau_C1.statistic:.3f}')
bend = res[(res.model == 'B') & (res.kind == 'end')].iloc[0]
print(f"(보고) B 문장 끝 분해: 원래 {bend.orig:.3f} → 인정 효과 {bend.C1_credit:.3f} → 재보정 {bend.C1:.3f} | 발생 전 경보 {bend.orig_pre:.3f} → {bend.C1_pre:.3f} | 서비스 10% {bend.orig_svc:.3f} → {bend.C1_svc:.3f}")
print(f"(보고) C2(애매 포함) B 문장 끝 {bend.C2:.3f}")
""")

md("## 11. 과제별과 사람 점검용 표본 (보고)")
code(r"""
tr = []
for tt in TASKS:
    for c in ['orig', 'C1']:
        cal_c, dev_c = V[c]; its = [x for x in dev_c if x['task_type'] == tt]
        row = {'task': tt, 'labels': c, 'n_hall': sum(x['onset'] is not None for x in its)}
        for sc in ['B|end', 'B|4', 'R1bV1b_s42|4', 'base_s42|4', 'large_s42|4']:
            if sc.split('|')[0] in SC:
                row[sc] = evaluate(its, sc, calibrate(cal_c, sc))['recall']
        tr.append(row)
task_df = pd.DataFrame(tr)
print(task_df.round(3).to_string(index=False))

# 사람 점검용: dev-L에서 재판정이 새로 환각이라고 한 문장 60개 무작위 (교수님 보고·정밀도 점검용, 선택)
add_rows = []
for x in V['C1'][1]:
    for i in x['added']:
        lab = LABELS[JUDGE][f'devL-{x["id"]}'][i]
        add_rows.append({'ragtruth_id': x['id'], 'task': x['task_type'], 'sent_no': i, 'sentence': sent_texts(x)[i - 1],
                         'type': lab['type'], 'phrase': lab['phrase'], 'reason': lab['reason'], 'human_verdict(H/S/A)': ''})
audit = pd.DataFrame(add_rows).sample(min(60, len(add_rows)), random_state=0) if add_rows else pd.DataFrame()
print(f'\n새로 추가된 환각 문장 {len(add_rows)}개 중 {len(audit)}개를 r4_human_audit.csv로 저장')
print(pd.DataFrame(add_rows).groupby(['task', 'type']).size().unstack(fill_value=0) if add_rows else '')
""")

md("## 12. 저장")
code(r"""
corr = {c: {sp: {x['id']: {'onset': x['onset'], 'added': x['added']} for x in its}
            for sp, its in zip(['calL', 'devL'], V[c])} for c in ['C1', 'C2']}
summ = {'R4_0': bool(R4_0), 'sens_HA': sens_HA, 'sens_H': sens_H, 'kappa_H': kappa_H, 'kappa_3': kappa_3, 'n_both': len(both),
        'R4_1': bool(R4_1), 'Bend_gain': BOOT['B|end'], 'placebo': PL, 'placebo_stat': PSTAT, 'R4_2': bool(R4_2), 'cap_orig': cap_o, 'cap_C1': cap_c,
        'boot': BOOT, 'kendall_tau_4': tau_C1.statistic, 'stat': STAT, 'usage': USAGE, 'judge': JUDGE, 'judge2': JUDGE2, 'use_temp': USE_TEMP}
json.dump(summ, open('r4_summary.json', 'w'), default=float, ensure_ascii=False, indent=1)
json.dump(corr, open(os.path.join(R4_DIR, 'r4_corrected_labels.json'), 'w'))
import shutil
out = {'r4_devL.csv': res, 'r4_family.csv': agg, 'r4_task.csv': task_df, 'r4_human_audit.csv': audit}
for f, d in out.items():
    d.to_csv(f, index=False); d.to_csv(os.path.join(R4_DIR, f), index=False)
shutil.copy('r4_summary.json', os.path.join(R4_DIR, 'r4_summary.json'))
print('저장:', list(out) + ['r4_summary.json', 'r4_corrected_labels.json'], '→', R4_DIR)
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r4_label_audit.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
