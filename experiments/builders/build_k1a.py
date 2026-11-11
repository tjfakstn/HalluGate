"""K1a 노트북 생성: 직접 모순 합성 학습 데이터 (SSOT §6 K1).
gold-large 학습 응답(train 분할)의 멀쩡한 문장을 Claude Sonnet 5.5가 ① 최소 수정 모순 ② 충실한 바꿔쓰기로 고쳐 쓰고,
그 문장만 바꾼 응답을 R4 지시문으로 따로 판정해 ①은 H, ②는 S인 것만 남긴다. 학습은 K1b 노트북.
데이터·학습 정답 코드는 R4c 노트북, 배치 코드는 M1 노트북, 정답 지시문은 R4(build_r4b_claude.py)에서 그대로 가져온다.
cal-L·dev-L 근거와 test 분할은 쓰지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')

def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    m = importlib.util.module_from_spec(spec); argv = sys.argv; spec.loader.exec_module(m); sys.argv = argv
    return m
M1 = _load('build_m1')

def cells_of(name):
    nb = json.load(open(os.path.join(NB_DIR, name)))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
R4C = cells_of('r4c_clean_labels_base.ipynb')
def find(C, prefix): return next(c for c in C if c.startswith(prefix))
DATA = find(R4C, "BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find(R4C, "# 학습 응답: S2/S2-b의 gold-large")
PREP = find(R4C, "ENC = tiktoken.get_encoding")
LOADLAB = find(R4C, "TRAIN_ITEMS = to_items(train)")
ADD = find(R4C, "def content_starts(x):")

# ---- K1A_* 셀은 smoke_k1.py가 가짜 데이터·가짜 API 결과로 그대로 실행해 확인한다 ----
K1A_SELECT = r"""
# 고쳐 쓸 문장 고르기: 보정 학습 정답(R4b Sonnet) 기준 멀쩡한 문장(원래 정답 y=0, 추가 환각 아님, 판정 S), 8~60토큰
N_PER_TASK, MAX_PER_RESP, MIN_TOK, MAX_TOK = 500, 2, 8, 60
HOLD = {x['ref_key'] for k in ['calL', 'devL'] for x in R0P['items'][k]}
assert not HOLD & {x['ref_key'] for x in TRAIN_ITEMS}, '학습 응답에 cal-L·dev-L 근거가 섞임'

def sent_text(x, sn): return x['text'][sn['a']:sn['b']].strip()

def candidates(x):
    labs = LABELS.get(f'tr-{x["id"]}')
    if not labs: return []
    return [i for i, sn in enumerate(x['sents'])
            if sn['y'] == 0 and (x['id'], i) not in ADD_ONSET and labs[i + 1]['label'] == 'S'
            and MIN_TOK <= sn['t1'] - sn['t0'] + 1 <= MAX_TOK]

rs = random.Random(2028)
PICK = {}
for tt in TASKS:
    pool = [x for x in TRAIN_ITEMS if x['task_type'] == tt and candidates(x)]
    for x in rs.sample(pool, min(N_PER_TASK, len(pool))):
        c = candidates(x)
        PICK[x['id']] = sorted(rs.sample(c, min(MAX_PER_RESP, len(c))))
BY_ID = {x['id']: x for x in TRAIN_ITEMS}
print(f'고쳐 쓸 응답 {len(PICK)}개, 문장 {sum(len(v) for v in PICK.values())}개 |',
      {t: sum(BY_ID[i]['task_type'] == t for i in PICK) for t in TASKS})
"""

K1A_GEN = r"""
SYN_MODEL = 'claude-sonnet-5-5'
SYN_SYSTEM = '''You create training data for a hallucination detector used in retrieval-augmented generation.
You receive a REFERENCE (the only source the model was given), a model RESPONSE split into numbered sentences, and TARGET sentence numbers.
Every target sentence is faithful to the reference. For EACH target sentence write two new versions of it:

1. "contradiction": a MINIMAL EDIT of the sentence that now DIRECTLY CONFLICTS with the reference.
   - Keep most of the original words and the sentence structure. Change one relation or detail so a careful reader who checks the reference finds it wrong.
   - It must contradict the reference, not merely add unsupported information. It must stay fluent and plausible.
   - Preferred kinds by task:
     Data2txt: attribution (which reviewer / how many reviewers said it, e.g. "another reviewer" when it was the same one, "several" when only one),
               flipped review detail (who did what, what was praised or criticized), mismatch with the business fields (hours, attributes, category).
     Summary: swapped roles or entities (who did what to whom, which person held which title), changed cause/effect, time or order, quantity.
     QA: changed step, condition or detail in the procedure or fact; negation; wrong item from the passages.
   - "changed": the shortest span copied EXACTLY from your contradiction sentence that carries the conflict.
   - "kind": one short label for what you changed (e.g. attribution, role_swap, detail_flip, negation, number, time, field_mismatch).
2. "paraphrase": a faithful rewording of the ORIGINAL sentence with the same meaning, fully supported by the reference.
   Reword noticeably (different wording or order) but add or drop no information.

Return ONLY a JSON object, no other text:
{"items": [{"i": 3, "contradiction": "...", "changed": "...", "kind": "...", "paraphrase": "..."}]}
with exactly one entry per target sentence.'''

def numbered(sents):
    return '\n'.join(f'[{k + 1}] {s}' for k, s in enumerate(sents))

def syn_msg(x, targets):
    sents = [sent_text(x, sn) for sn in x['sents']]
    return (f"TASK: {TASK_DESC[x['task_type']]}\n\nREFERENCE:\n<<<\n{REFS[x['ref_key']]}\n>>>\n\n"
            f"RESPONSE ({len(sents)} numbered sentences):\n{numbered(sents)}\n\nTARGET sentences: {', '.join(str(i + 1) for i in targets)}")

def syn_params(x, targets):
    return {'model': SYN_MODEL, 'max_tokens': 3000, 'system': SYN_SYSTEM,
            'messages': [{'role': 'user', 'content': syn_msg(x, targets)}]}

def words(s): return [w.strip('.,;:!?"\'()').lower() for w in s.split() if w.strip('.,;:!?"\'()')]

def overlap(a, b):
    # 고쳐 쓴 문장이 원래 단어를 얼마나 유지했나 (원래 단어 중 남은 비율)
    wa, wb = words(a), set(words(b))
    return sum(w in wb for w in wa) / max(len(wa), 1)

def parse_syn(text, x, targets):
    # 후보 목록: (문장 번호 0부터, 종류 contra/para, 새 문장, 바뀐 구절, 바꾼 종류). 형식이 틀린 항목은 버린다
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
    except Exception:
        return [], 'json'
    out = []
    for e in d.get('items', []):
        try: i = int(e.get('i', 0)) - 1
        except Exception: continue
        if i not in targets: continue
        orig = sent_text(x, x['sents'][i])
        c, ch, p = str(e.get('contradiction', '')).strip(), str(e.get('changed', '')).strip(), str(e.get('paraphrase', '')).strip()
        if c and ch and c != orig and (ch in c or ch.lower() in c.lower()) and overlap(orig, c) >= 0.5:
            out.append((i, 'contra', c, ch, str(e.get('kind', ''))[:30]))
        if p and p != orig:
            out.append((i, 'para', p, '', ''))
    return out, 'ok'
"""

K1A_VERIFY = r"""
VERIFY_MODEL = 'claude-sonnet-5-5'

def verify_msg(x, i, new):
    sents = [sent_text(x, sn) for sn in x['sents']]
    sents[i] = new
    return (f"TASK: {TASK_DESC[x['task_type']]}\n\nREFERENCE:\n<<<\n{REFS[x['ref_key']]}\n>>>\n\n"
            f"RESPONSE ({len(sents)} numbered sentences):\n{numbered(sents)}\n\n"
            f"This time label ONLY sentence [{i + 1}]. Return ONLY "
            '{"sentences": [{"i": ' + str(i + 1) + ', "label": "S"}]} with that single entry (add type, phrase and reason if H or A).')

def verify_params(x, i, new):
    # R4와 같은 정답 지시문(SYSTEM), 대상 문장 하나만
    return {'model': VERIFY_MODEL, 'max_tokens': 400, 'system': SYSTEM,
            'messages': [{'role': 'user', 'content': verify_msg(x, i, new)}]}

def parse_verify(text, i):
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        for e in d.get('sentences', []):
            if int(e.get('i', 0)) == i + 1:
                return str(e.get('label', '')).strip().upper()[:1] or None
    except Exception:
        pass
    return None

def keep(kind, label):
    return (kind == 'contra' and label == 'H') or (kind == 'para' and label == 'S')
"""

K1A_EXPORT = r"""
N_MAX = 1500
def build_synth(cands, vres):
    # cands: vid -> 후보 dict, vres: vid -> 판정 결과. 통과한 ①·② 중 같은 수(최대 각 N_MAX)
    ok = {'contra': [], 'para': []}
    for vid, c in cands.items():
        if vid not in vres: continue
        lab = parse_verify(vres[vid]['text'], c['i'])
        c = dict(c, verify=lab)
        if keep(c['kind'], lab): ok[c['kind']].append(c)
    n = min(len(ok['contra']), len(ok['para']), N_MAX)
    r = random.Random(2029)
    out = r.sample(ok['contra'], n) + r.sample(ok['para'], n)
    return out, {k: len(v) for k, v in ok.items()}, n

def check_sample(synth, k=40):
    # K1-0 가린 점검용: 통과한 모순 40개 (근거, 원래 문장, 고친 문장, 바뀐 구절)
    con = [c for c in synth if c['kind'] == 'contra']
    sel = random.Random(2030).sample(con, min(k, len(con)))
    return [{'qid': f'q{n + 1:02d}', 'task': c['task'], 'reference': REFS[c['ref_key']], 'original': c['orig'],
             'rewritten': c['sentence'], 'changed': c['phrase']} for n, c in enumerate(sel)]
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# K1a — 직접 모순 합성 학습 데이터

## 왜 하나

M1a에서 최신 LLM(Claude Haiku·Sonnet 5.5) 환각을 판정기가 놓친 원인을 보니, **놓친 진짜 환각의 68%가 직접 모순**이었다.
근거에 있는 단어를 그대로 쓰면서 *누가·무엇을*만 바꾼 문장이다(같은 리뷰어의 말을 "다른 리뷰어도"로, 인물 역할 뒤바꿈). 모순 46문장 중 경보는 1개.
학습 데이터(RAGTruth, 2023년 무렵 LLM 응답)에 이런 미세한 모순이 적었을 가능성이 크다 → **이런 모순을 합성해 F1 base를 한 번 더 학습**한다(K1b).

## 무엇을 하나

1. gold-large 학습 응답(train 분할, cal-L·dev-L 근거 제외)에서 과제마다 500개, 응답당 멀쩡한 문장 최대 2개를 고른다
2. `claude-sonnet-5-5`(배치)가 문장마다 **① 최소 수정 모순**(바뀐 구절 표시)과 **② 충실한 바꿔쓰기**(대조용)를 만든다
3. 그 문장만 바꾼 응답을 R4와 같은 정답 지시문으로 `claude-sonnet-5-5`가 **따로** 판정 → ①은 H, ②는 S인 것만 남긴다
4. 통과한 ①·②를 같은 수(최대 각 1,500)로 `k1/synth_train.json`에 저장, 가린 점검용 모순 40개를 `k1_check.json`으로 내보낸다

## 사전 정의 기준 (SSOT §6 K1, 실행 전 고정)

- **K1-0(이 노트북 부분)**: 검증을 통과한 모순 ≥ 300개. 그리고 `k1_check.json` 40개를 하위 에이전트가 가린 채 점검해 진짜 모순 ≥ 85% (점검은 Claude Code에서)
- 주 기준 K1-1(새 생성 환각 응답 탐지율 +0.10)과 안전 기준 K1-2·K1-3은 K1b 노트북에서 판정

## 실행 방법

1. 왼쪽 열쇠 아이콘(보안 비밀)에 `ANTHROPIC_API_KEY`, 노트북 접근 켜기
2. 런타임: **CPU로 충분**(GPU 필요 없음)
3. **모두 실행**. 배치 두 번(생성 → 검증), 보통 1~2시간. 끊기면 다시 모두 실행 — 드라이브 `k1/`에서 이어서 한다
4. 비용(추정, 실행 중 출력): 생성 약 $7 + 검증 약 $15
5. 끝나면 `k1a_results.zip`이 내려받아진다 → Claude Code에 올리기 (`k1_check.json` 점검)

> 드라이브에 `r0/r0_items.pkl`, `r4b/labels_train.json`이 있어야 한다. 이 노트북과 별도로 `m1_fresh_eval.ipynb`(새 평가 세트)를 돌려 둔다.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install anthropic pysbd tiktoken
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'
import json, time, random, os, gc, copy, pickle
import numpy as np, pandas as pd, pysbd, tiktoken
from google.colab import drive
drive.mount('/content/drive')
SEED = 42
random.seed(SEED); np.random.seed(SEED)
OUT_DIR = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
R0_DIR, R4B_DIR, K1_DIR = (os.path.join(OUT_DIR, d) for d in ['r0', 'r4b', 'k1'])
os.makedirs(K1_DIR, exist_ok=True)
for p in [os.path.join(R4B_DIR, 'labels_train.json'), os.path.join(R0_DIR, 'r0_items.pkl')]:
    assert os.path.exists(p), f'{p} 없음'
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
""")
md("## 2. 데이터와 보정 학습 정답 (R4c·F1과 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
code(LOADLAB)
code(ADD)
md("## 3. 고쳐 쓸 문장 고르기")
code(K1A_SELECT)
md("## 4. Claude API (M1과 같은 배치 코드)")
code(M1.M1_API)
md("## 5. ① 모순 · ② 바꿔쓰기 생성")
code(M1.LABEL_PROMPT + K1A_GEN + r"""
# 형식 확인 1건 (수 센트)
_x = BY_ID[next(iter(PICK))]
_g = parse_syn(text_of(A.messages.create(**syn_params(_x, PICK[_x['id']]))), _x, PICK[_x['id']])
for i, k, s, ch, kd in _g[0]: print(f"  [{i + 1}] {k:6} {kd:14} {s[:150]!r} | 바뀐 구절 {ch!r}")
print(f"예상 비용: 생성 약 ${len(PICK) * (2500 * 1.0 + 700 * 5.0) / 1e6:.1f} + 검증 약 ${len(PICK) * 4 * (2600 * 1.0 + 80 * 5.0) / 1e6:.1f} (토큰 추정)")

GEN = run_batch({f'syn-{xid}': syn_params(BY_ID[xid], t) for xid, t in PICK.items()}, os.path.join(K1_DIR, 'syn_gen.json'))
CANDS, bad = {}, 0
for xid, t in PICK.items():
    g = GEN.get(f'syn-{xid}')
    if g is None: continue
    out, st = parse_syn(g['text'], BY_ID[xid], t)
    bad += st != 'ok'
    for i, kind, s, ch, kd in out:
        x = BY_ID[xid]
        CANDS[f'v-{xid}-{i}-{kind}'] = {'rid': xid, 'ref_key': x['ref_key'], 'task': x['task_type'], 'i': i, 'kind': kind,
                                        'orig': sent_text(x, x['sents'][i]), 'sentence': s, 'phrase': ch, 'gen_kind': kd}
print(f"후보: 모순 {sum(c['kind'] == 'contra' for c in CANDS.values())}, 바꿔쓰기 {sum(c['kind'] == 'para' for c in CANDS.values())} | JSON 파싱 실패 {bad}")
""")
md("## 6. 따로 판정해 검증 (R4 지시문, 대상 문장만)")
code(K1A_VERIFY + r"""
VRES = run_batch({vid: verify_params(BY_ID[c['rid']], c['i'], c['sentence']) for vid, c in CANDS.items()},
                 os.path.join(K1_DIR, 'syn_verify.json'))
""")
md("## 7. 저장 · K1-0(개수) 확인 · 점검 표본")
code(K1A_EXPORT + r"""
SYN, PASSED, N_EACH = build_synth(CANDS, VRES)
by = pd.DataFrame([dict(c, verify=parse_verify(VRES[v]['text'], c['i']) if v in VRES else None) for v, c in CANDS.items()])
print('검증 결과 (종류 × 판정)'); print(pd.crosstab([by.task, by.kind], by.verify.fillna('없음')))
print('모순 종류(통과분)', pd.Series([c['gen_kind'] for c in SYN if c['kind'] == 'contra']).value_counts().head(12).to_dict())
K1_0_count = PASSED['contra'] >= 300
print(f"=== K1-0(개수): 통과한 모순 {PASSED['contra']} (≥ 300) → {'충족' if K1_0_count else '미달 → 보류'} | 학습에 쓸 수 각 {N_EACH} ===")
save_json(SYN, os.path.join(K1_DIR, 'synth_train.json'))
LOCAL = '/content/k1a_local'; os.makedirs(LOCAL, exist_ok=True)
CHECK = check_sample(SYN)
json.dump(CHECK, open(os.path.join(LOCAL, 'k1_check.json'), 'w'), ensure_ascii=False, indent=1)
usage = {f: round(cost(load_json(os.path.join(K1_DIR, f))), 2) for f in ['syn_gen.json', 'syn_verify.json']}
summ = {'n_pick_resp': len(PICK), 'n_pick_sent': sum(len(v) for v in PICK.values()), 'cands': by.kind.value_counts().to_dict(),
        'passed': PASSED, 'n_each': N_EACH, 'K1_0_count': bool(K1_0_count), 'cost_usd': usage,
        'by_task': {t: {k: int(((by.task == t) & (by.kind == k) & by.apply(lambda r: keep(r['kind'], r['verify']), axis=1)).sum()) for k in ['contra', 'para']} for t in TASKS},
        'gen_kinds': pd.Series([c['gen_kind'] for c in SYN if c['kind'] == 'contra']).value_counts().to_dict()}
json.dump(summ, open(os.path.join(LOCAL, 'k1a_summary.json'), 'w'), ensure_ascii=False, indent=1)
print('API 사용량(추정 $)', usage)
import shutil
zp = shutil.make_archive('/content/k1a_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'k1a_contradiction_synth.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
