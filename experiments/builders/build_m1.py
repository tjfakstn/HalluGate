"""M1 노트북 생성: 최신 LLM·컬렉션 조건 점검 — 배포 판정기(I3 조합)의 임계값이 최신 Claude 응답과 게이트웨이 컬렉션 RAG 흐름에서도
멀쩡한 응답 표시율을 지키는지 잰다. 학습 없음. 사전 기준은 SSOT §6 M1.
게이트웨이 코드(hallugate/judge.py·store.py·context.py)를 그대로 품고, 판정 지점·조합식은 L3와 같은 코드를 쓴다.
정답(Opus) 지시문과 파싱은 R4/R4b와 같은 코드를 build_r4b_claude.py에서 그대로 가져온다. RAGTruth test 분할은 열지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
SRC = {n: open(os.path.join(ROOT, 'hallugate', f'{n}.py'), encoding='utf-8').read() for n in ['judge', 'store', 'context']}

_spec = importlib.util.spec_from_file_location('build_l3', os.path.join(HERE, 'build_l3.py'))
L3 = importlib.util.module_from_spec(_spec); _argv = sys.argv; _spec.loader.exec_module(L3); sys.argv = _argv
COMBO = L3.COMBO

# R4와 같은 정답 지시문(SYSTEM, TASK_DESC)과 파싱(parse)
_r4b = open(os.path.join(HERE, 'build_r4b_claude.py'), encoding='utf-8').read()
def _between(s, a, b):
    i = s.index(a); j = s.index(b, i)
    return s[i:j]
LABEL_PROMPT = _between(_r4b, "SYSTEM = '''", "TRAIN_ITEMS = to_items(train)").rstrip() + '\n'
LABEL_PARSE = _between(_r4b, "def parse(text, n):", "n_resp = len(ITEMS)").rstrip() + '\n'
assert 'Subtle baseless info' in LABEL_PROMPT and "'Data2txt'" in LABEL_PROMPT and 'return labs, len(labs) == n' in LABEL_PARSE

# ---- 아래 M1_* 셀은 smoke_m1.py가 맥 CPU에서 가짜 API 결과·실제 가중치로 그대로 실행해 확인한다 ----
M1_FLOW = r"""
# 생성 요청 만들기: QA = 컬렉션 RAG(게이트웨이 store·context 그대로), Summary·Data2txt = 근거 직접 전달(RAGTruth 원 프롬프트)
GENS = {'haiku': 'claude-haiku-5-5', 'sonnet': 'claude-sonnet-5-5'}
LABELER = 'claude-opus-5-5'
TOP_K, CHUNK_W, OVERLAP = 4, 120, 20          # 게이트웨이 기본값 (config.py)
GEN_MAX_TOKENS = 8000

def build_collection(store, coll, sids, sources):
    n = 0
    for sid in sids:
        if sources[sid]['task_type'] != 'QA': continue
        text = sources[sid]['source_info'].get('passages', '')
        if text.strip():
            store.add(coll, text, title=sid, doc_id=sid); n += 1
    return n

def make_cases(split, sids, sources, store, coll):
    cases = {}
    for sid in sids:
        src = sources[sid]; tt = src['task_type']
        if tt == 'QA':
            q = src['source_info'].get('question', '')
            ev = store.search(coll, q, TOP_K)
            msgs = rag_messages([{'role': 'user', 'content': q}], ev)
            system, messages = msgs[0]['content'], msgs[1:]          # OpenAI 형식 시스템 메시지 → Messages API system
            ref_j = rag_reference(ev)
            extra = {'flow': 'collection', 'gold_in': any(e.doc_id == sid for e in ev),
                     'ref_label': f'Question: {q}\n{ref_j}'}
        else:
            system, messages = None, [{'role': 'user', 'content': src['prompt']}]
            ref_j = build_reference(src)
            extra = {'flow': 'direct', 'gold_in': None, 'ref_label': ref_j}
        for g, model in GENS.items():
            cases[f'{split}-{g}-{sid}'] = {'sid': sid, 'task': tt, 'gen': g, 'model': model, 'system': system,
                                           'messages': messages, 'ref_judge': ref_j, **extra}
    return cases

def gen_params(c):
    p = {'model': c['model'], 'max_tokens': GEN_MAX_TOKENS, 'messages': c['messages']}
    if c['system']: p['system'] = c['system']
    return p
"""

M1_LABEL = LABEL_PROMPT + r"""
def sentences(text):
    # (시작, 끝, 문장) — 판정 지점(points_of)과 같은 pysbd 분할
    out = []
    for sp in SEG.segment(text):
        if not sp.sent.strip(): continue
        a = sp.start + (len(sp.sent) - len(sp.sent.lstrip()))
        out.append((a, sp.start + len(sp.sent.rstrip()), sp.sent.strip()))
    return out

def label_msg(c, text):
    ss = sentences(text)
    lines = '\n'.join(f'[{i + 1}] {s}' for i, (_, _, s) in enumerate(ss))
    return f"TASK: {TASK_DESC[c['task']]}\n\nREFERENCE:\n<<<\n{c['ref_label']}\n>>>\n\nRESPONSE ({len(ss)} numbered sentences):\n{lines}"

def label_params(c, text):
    # R4와 같은 요청 (모델·max_tokens·지시문)
    return {'model': LABELER, 'max_tokens': 4096, 'system': SYSTEM, 'messages': [{'role': 'user', 'content': label_msg(c, text)}]}

""" + LABEL_PARSE

M1_SCORE = r"""
def score_response(jF, jI, refs, text, fn):
    # 판정 지점(문장마다 8토큰째 + 문장 끝)마다 (문장 시작, 지점 끝, F1 base, I2, 조합)
    out = []
    for a, cut in points_of(text):
        t = text[a:cut].strip()
        if not t: continue
        sf, si = fn(jF, jI, refs[0], refs[1], {'text': t, 'prior': text[:a]})
        out.append((a, cut, sf, si, combine(sf, si)))
    return out

def gen_status(g):
    if g is None: return 'nogen'
    if g['stop'] == 'refusal': return 'refusal'
    if not g['text'].strip() or not sentences(g['text']): return 'empty'
    return 'valid'
"""

M1_METRICS = r"""
import pandas as pd

def wilson(k, n, z=1.96):
    if n == 0: return float('nan'), float('nan'), float('nan')
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h

def conformal(maxes, alpha=0.10):
    # cal-L 노트북(conformal_flag)과 같은 등각 보정: 멀쩡한 응답 최대 점수의 ceil((n+1)(1-α))번째 + 1e-6
    mx = np.sort(np.asarray(maxes, float)); n = len(mx); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(mx[k - 1]) + 1e-6 if k <= n else float('inf')

def response_rows(cases, gens, labels, scores, thr_c, thr_f):
    rows = []
    for cid, c in cases.items():
        g = gens.get(cid); st = gen_status(g)
        r = {'cid': cid, 'sid': c['sid'], 'task': c['task'], 'gen': c['gen'], 'flow': c['flow'], 'gold_in': c['gold_in'],
             'stop': g['stop'] if g else None, 'status': st}
        if st == 'valid':
            ss = sentences(g['text']); lab = labels.get(cid)
            r['status'] = 'ok' if lab is not None else 'nolabel'
            pts = scores.get(cid, [])
            r['n_sent'], r['n_pts'] = len(ss), len(pts)
            r['max_c'] = max([p[4] for p in pts], default=-np.inf)
            r['max_f'] = max([p[2] for p in pts], default=-np.inf)
            r['flag_c'], r['flag_f'] = bool(r['max_c'] >= thr_c), bool(r['max_f'] >= thr_f)
            first = next((p[0] for p in pts if p[4] >= thr_c), None)
            r['first_alarm_a'] = first
            if lab is not None:
                ls = [lab[i + 1]['label'] for i in range(len(ss))]
                r['n_H'], r['n_A'] = ls.count('H'), ls.count('A')
                r['clean'], r['strict_clean'] = r['n_H'] == 0, all(l == 'S' for l in ls)
                hs = [ss[i][0] for i, l in enumerate(ls) if l == 'H']
                r['first_H_a'] = hs[0] if hs else None
                r['strict_det'] = bool(hs and first is not None and first >= hs[0])
        rows.append(r)
    return pd.DataFrame(rows)

def rate_line(df, col='flag_c'):
    k, n = int(df[col].sum()), len(df)
    p, lo, hi = wilson(k, n)
    return {'k': k, 'n': n, 'rate': p, 'lo': lo, 'hi': hi}

def fmt(d):
    return f"{d['rate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] ({d['k']}/{d['n']})" if d['n'] else '—'

def report(df, thr_name=''):
    ok = df[df.status == 'ok'].copy()
    for col in ['clean', 'strict_clean', 'flag_c', 'flag_f', 'strict_det']:
        ok[col] = ok[col].astype(bool)
    clean, hal = ok[ok.clean], ok[~ok.clean]
    out = {'status': df.status.value_counts().to_dict(), 'max_tokens_stops': int((df.stop == 'max_tokens').sum()),
           'clean_combo': rate_line(clean), 'clean_f1': rate_line(clean, 'flag_f'),
           'strict_clean_combo': rate_line(ok[ok.strict_clean]),
           'hal_any': rate_line(hal), 'hal_strict': rate_line(hal, 'strict_det'),
           'flag_precision_H': rate_line(ok[ok.flag_c], 'clean').copy(),
           'by_gen': {g: rate_line(d) for g, d in clean.groupby('gen')},
           'by_task': {t: rate_line(d) for t, d in clean.groupby('task')},
           'by_gen_task': {f'{g}|{t}': rate_line(d) for (g, t), d in clean.groupby(['gen', 'task'])},
           'hal_rate_by_gen': {g: float((~d.clean).mean()) for g, d in ok.groupby('gen')}}
    fp = out['flag_precision_H']                     # 표시된 응답 중 멀쩡한 비율 → H 비율로 바꿈
    if fp['n']: fp.update(k=fp['n'] - fp['k'], rate=1 - fp['rate'], lo=1 - fp['hi'], hi=1 - fp['lo'])
    qa = df[(df.flow == 'collection') & (df.gen == 'haiku')]
    out['qa_gold_in'] = float(qa.gold_in.astype(float).mean()) if len(qa) else None
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# M1 — 최신 LLM·컬렉션 조건 점검 (배포 판정기 임계값)

**왜 하나**: 배포 판정기(I3 조합: F1 base + I2, 임계값 2.992)는 RAGTruth의 **2023년 무렵 LLM 응답**으로 "멀쩡한 응답 10%만 표시"되게 맞췄다. 실제 서비스는 최신 LLM이 답하고, 게이트웨이가 **컬렉션에서 검색한 조각**을 근거로 쓴다. 이 조건에서 멀쩡한 응답에 경고가 너무 많이 뜨면 서비스로 쓸 수 없다.

**무엇을 하나** (학습 없음)
- 근거: **dev-L 근거 문서 전부**(RAGTruth train, 학습·cal-L과 근거가 겹치지 않음)
- 생성: `claude-haiku-5-5`·`claude-sonnet-5-5`(배치, 기본 설정), 근거마다 모델당 1개
  - **QA = 컬렉션 RAG**: dev-L QA 근거를 컬렉션 하나에 등록 → 질문으로 검색(top-4, 120단어 조각) → 게이트웨이 시스템 지시문으로 생성 → 판정 근거 = 검색된 조각
  - **Summary·Data2txt = 근거 직접 전달**: RAGTruth 원 프롬프트, 판정 근거 = 원 근거
- 정답: `claude-opus-5-5`가 R4와 같은 지시문으로 문장마다 S/H/A. **멀쩡한 응답 = H 문장 없음**(C1 규칙)
- 판정: 게이트웨이 `judge.py`로 문장마다 8토큰 + 문장 끝, 조합 점수가 임계값 이상인 지점이 하나라도 있으면 "표시"

**사전 정의 기준** (SSOT §6 M1, 실행 전 고정)
- **M1-0 유효성**: 정답 확보 응답 ≥ 90%, 멀쩡한 응답 ≥ 200개. 미달이면 판정 보류
- **M1-1 (주)**: 두 모델·세 과제를 합친 **멀쩡한 응답 표시율 ≤ 15%**
- **M1-2 (M1-1 미달일 때만 자동 실행)**: cal-L 근거로 같은 흐름 → 그 멀쩡한 응답으로 서비스 10% 임계값 재보정 → dev-L 멀쩡한 응답 표시율 ≤ 15%면 "재보정 임계값 후보"(교체는 별도 결정)
- 보고만: 모델·과제별 표시율(95% 구간), F1 base 단독, 엄격 멀쩡(모든 문장 S), 환각 응답 탐지율, 표시된 응답 중 실제 환각 비율, QA 검색에 원래 근거가 들어간 비율

**실행 방법**
1. 왼쪽 열쇠 아이콘(보안 비밀)에 `ANTHROPIC_API_KEY`를 넣고 노트북 접근을 켠다
2. 런타임 유형: **GPU**(A100·L4·T4 아무거나. 지연은 재지 않음)
3. **모두 실행**. 배치는 보통 수십 분~1시간. 끊기면 다시 모두 실행 — 제출한 배치·받은 결과·판정 점수는 드라이브 `m1/`에서 이어서 한다
4. 비용(추정, 실행 중 출력): dev-L 약 $5~8. M1-1 미달이면 cal-L도 돌아 약 2배
5. 끝나면 `m1_results.zip`이 내려받아진다
""")

md("## 1. 환경 준비")
code("SRC = " + repr(SRC) + r"""
import os, sys, json, time, random, subprocess, shutil, pickle
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'anthropic', 'sentence-transformers', 'tiktoken', 'pysbd', '-U', 'transformers>=4.48'], check=True)
subprocess.run('git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || true', shell=True)
import numpy as np, torch
from google.colab import drive
drive.mount('/content/drive')
# 게이트웨이 코드 그대로 (패키지 __init__은 FastAPI 앱을 불러오므로 비워 둔다)
os.makedirs('/content/hg/hallugate', exist_ok=True)
open('/content/hg/hallugate/__init__.py', 'w').write('')
for n, s in SRC.items(): open(f'/content/hg/hallugate/{n}.py', 'w').write(s)
sys.path.insert(0, '/content/hg')
from hallugate.judge import StudentJudge, read_config, input_mode
from hallugate.store import SQLiteStore
from hallugate.context import rag_messages, rag_reference

OUT = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
F_DIR, I_DIR = os.path.join(OUT, 'final', 'judge_base'), os.path.join(OUT, 'i2', 'judge_i2')
R0_DIR, M1_DIR = os.path.join(OUT, 'r0'), os.path.join(OUT, 'm1')
os.makedirs(M1_DIR, exist_ok=True)
for d in [F_DIR, I_DIR]:
    assert os.path.exists(os.path.join(d, 'judge_config.json')), f'{d}/judge_config.json 없음'
assert os.path.exists(os.path.join(R0_DIR, 'r0_items.pkl')), 'r0/r0_items.pkl 없음 (cal-L·dev-L 분할)'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'
COMBO = """ + repr(COMBO) + r"""
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
print('device', GPU, '| 조합 상수(I3)', COMBO)
""")

md("## 2. 근거 문서 — dev-L(평가), cal-L(M1-2 재보정용)")
code(L3.L3_INPUTS + r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
SIDS = {sp: sorted({x['ref_key'] for x in R0P['items'][k]}) for sp, k in [('devL', 'devL'), ('calL', 'calL')]}
assert not set(SIDS['devL']) & set(SIDS['calL']), 'dev-L·cal-L 근거가 겹침'
split_of = {}
for r in responses: split_of.setdefault(r['source_id'], set()).add(r['split'])
for sp, ss in SIDS.items():
    assert all(split_of[s] == {'train'} for s in ss), f'{sp}에 test 분할 근거가 섞임'
    print(sp, f'근거 {len(ss)}개 |', {t: sum(sources[s]['task_type'] == t for s in ss) for t in ['QA', 'Summary', 'Data2txt']})
""")

md("## 3. 판정기·생성 요청 준비")
code(L3.L3_CORE + M1_FLOW + M1_LABEL + M1_SCORE + r"""
jF, jI = make_judge(F_DIR, DEVICE), make_judge(I_DIR, DEVICE)
assert jF.mode == 'retrieval' and jI.mode == 'full', (jF.mode, jI.mode)
THR_C = COMBO['threshold']
THR_F = read_config(F_DIR)['thresholds']['8']['svc10_orig']
FN = judge_par if DEVICE == 'cuda' else judge_seq
print(f'임계값: 조합 {THR_C:.4f} | F1 base 단독 {THR_F:.4f} (보고용)')

def prepare_split(split):
    cp = os.path.join(M1_DIR, f'cases_{split}.json')
    cases = load_json(cp)                      # 이미 만든 요청이 있으면 그대로 (검색 결과 = LLM이 본 근거가 바뀌지 않게)
    if cases is None:
        store = SQLiteStore(f'/content/m1_{split}.sqlite3', jF.embedder, CHUNK_W, OVERLAP)
        coll = f'm1-{split}'
        if not store.exists(coll):
            print(f'컬렉션 {coll}: QA 근거 {build_collection(store, coll, SIDS[split], sources)}개 등록')
        cases = make_cases(split, SIDS[split], sources, store, coll)
        save_json(cases, cp)
    qa = [c for c in cases.values() if c['flow'] == 'collection' and c['gen'] == 'haiku']
    print(f"{split}: 생성 요청 {len(cases)}개 | QA 검색에 원래 근거가 들어간 비율 {np.mean([c['gold_in'] for c in qa]):.3f}")
    return cases

CASES = {'devL': prepare_split('devL')}
""")

md("## 4. Claude API — 형식 확인 3건, 배치 제출·결과 받기 (드라이브에서 이어서)")
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
A = anthropic.Anthropic(api_key=get_key('ANTHROPIC_API_KEY'), max_retries=3)
PRICE = {'claude-haiku-5-5': (0.05, 0.25), 'claude-sonnet-5-5': (1.0, 5.0), 'claude-opus-5-5': (2.0, 10.0)}   # 배치 $/백만 토큰 (2026-11)

def text_of(m): return ''.join(b.text for b in m.content if getattr(b, 'type', '') == 'text')
def rec(m): return {'text': text_of(m), 'stop': m.stop_reason, 'model': m.model, 'in': m.usage.input_tokens, 'out': m.usage.output_tokens}
def cost(res): return sum(r['in'] / 1e6 * PRICE.get(r['model'], (2, 10))[0] + r['out'] / 1e6 * PRICE.get(r['model'], (2, 10))[1] for r in res.values())

def run_batch(reqs, path, chunk=400):
    # reqs: cid -> 요청. 결과: cid -> rec. 다시 실행하면 제출한 묶음·받은 결과는 건너뛴다
    res = load_json(path) or {}
    bpath = path.replace('.json', '_batches.json')
    B = load_json(bpath) or {'chunks': []}
    sent = {c for ch in B['chunks'] for c in ch['cids']}
    todo = [c for c in reqs if c not in sent and c not in res]
    for k in range(0, len(todo), chunk):
        part = todo[k:k + chunk]
        b = A.messages.batches.create(requests=[{'custom_id': c, 'params': reqs[c]} for c in part])
        B['chunks'].append({'id': b.id, 'cids': part, 'done': False}); save_json(B, bpath)
        print(f'  제출 {b.id} ({len(part)}건)', flush=True)
    for ch in B['chunks']:
        if ch.get('done'): continue
        while True:
            b = A.messages.batches.retrieve(ch['id']); rc = b.request_counts
            print(f'  {time.strftime("%H:%M:%S")} {ch["id"][-8:]} {b.processing_status} | 처리 중 {rc.processing}, 성공 {rc.succeeded}, 오류 {rc.errored}', flush=True)
            if b.processing_status == 'ended': break
            time.sleep(60)
        for r in A.messages.batches.results(ch['id']):
            if r.result.type == 'succeeded': res[r.custom_id] = rec(r.result.message)
        ch['done'] = True; save_json(B, bpath); save_json(res, path)
    miss = [c for c in reqs if c not in res]
    for c in miss[:100]:                                   # 배치 실패분은 바로 요청 1회
        try: res[c] = rec(A.messages.create(**reqs[c]))
        except Exception as e: print(f'  {c}: {type(e).__name__} {str(e)[:150]}'); break
    save_json(res, path)
    print(f'  결과 {len(res)}/{len(reqs)} | 사용량 약 ${cost({c: res[c] for c in reqs if c in res}):.2f}')
    return res

# 형식 확인: 모델마다 1건 바로 생성 + 그 응답 정답 1건 (수 센트)
for g in GENS:
    cid = next(c for c, x in CASES['devL'].items() if x['gen'] == g and x['flow'] == 'collection')
    m = A.messages.create(**gen_params(CASES['devL'][cid])); r = rec(m)
    print(f"[{g}] stop={r['stop']} 입력 {r['in']} 출력 {r['out']} | {r['text'][:200]!r}")
    if r['text'].strip():
        lm = A.messages.create(**label_params(CASES['devL'][cid], r['text'])); labs, ok = parse(text_of(lm), len(sentences(r['text'])))
        print(f"   정답 파싱 {'성공' if ok else '실패'} | {[v['label'] for v in labs.values()]}")
n = len(CASES['devL'])
print(f'dev-L 예상 비용: 생성 약 ${n / 2 * (1500 * 0.05 + 1500 * 0.25 + 1500 * 1.0 + 1500 * 5.0) / 1e6:.1f} + 정답 약 ${n * (2200 * 2.0 + 700 * 10.0) / 1e6:.1f} (응답당 토큰 추정)')
""")

md("## 5. 생성 → 정답 → 판정 (dev-L)")
code(r"""
def run_split(split):
    cases = CASES[split]
    gens = run_batch({c: gen_params(x) for c, x in cases.items()}, os.path.join(M1_DIR, f'gen_{split}.json'))
    valid = [c for c in cases if gen_status(gens.get(c)) == 'valid']
    print(f'{split} 생성: 유효 {len(valid)} | 거절 {sum(gen_status(gens.get(c)) == "refusal" for c in cases)} | '
          f'빈 응답 {sum(gen_status(gens.get(c)) == "empty" for c in cases)} | max_tokens {sum((gens.get(c) or {}).get("stop") == "max_tokens" for c in cases)}')
    raw = run_batch({c: label_params(cases[c], gens[c]['text']) for c in valid}, os.path.join(M1_DIR, f'label_{split}.json'))
    labels, bad = {}, []
    for c in valid:
        if c not in raw: continue
        labs, ok = parse(raw[c]['text'], len(sentences(gens[c]['text'])))
        (labels.__setitem__(c, labs) if ok else bad.append(c))
    for c in bad[:50]:                                     # 파싱 실패는 바로 1회 재요청
        try:
            labs, ok = parse(text_of(A.messages.create(**label_params(cases[c], gens[c]['text']))), len(sentences(gens[c]['text'])))
            if ok: labels[c] = labs
        except Exception as e: print(f'  {c}: {type(e).__name__}'); break
    print(f'{split} 정답: {len(labels)}/{len(valid)} 확보')
    sp = os.path.join(M1_DIR, f'scores_{split}.json')
    scores = load_json(sp) or {}
    refs = {}
    t0 = time.time()
    for k, c in enumerate([c for c in valid if c not in scores]):
        r = cases[c]['ref_judge']
        if r not in refs: refs = {r: (jF.prepare(r), jI.prepare(r))}          # 근거별로 연속 (같은 근거 두 모델)
        scores[c] = [list(p) for p in score_response(jF, jI, refs[r], gens[c]['text'], FN)]
        if (k + 1) % 25 == 0:
            save_json(scores, sp); print(f'  판정 {len(scores)}/{len(valid)} ({time.time() - t0:.0f}s)', flush=True)
    save_json(scores, sp)
    return gens, labels, scores

RUN = {'devL': run_split('devL')}
""")

md("## 6. 사전 기준 판정 — M1-0, M1-1")
code(M1_METRICS + r"""
DEV = response_rows(CASES['devL'], *RUN['devL'], THR_C, THR_F)
REP = {'devL': report(DEV)}
r = REP['devL']
valid_n = int(DEV.status.isin(['ok', 'nolabel']).sum()); ok_n = int((DEV.status == 'ok').sum())
label_ratio = ok_n / max(valid_n, 1)
M1_0 = bool(label_ratio >= 0.90 and r['clean_combo']['n'] >= 200)
M1_1 = bool(M1_0 and r['clean_combo']['rate'] <= 0.15)
print('응답 상태', r['status'], '| max_tokens로 끊긴 응답', r['max_tokens_stops'])
print(f"QA 검색에 원래 근거가 들어간 비율 {r['qa_gold_in']:.3f} | 환각 응답 비율(Opus H)", {g: round(v, 3) for g, v in r['hal_rate_by_gen'].items()})
print(f"\n멀쩡한 응답 표시율 (조합, 임계값 {THR_C:.3f}): {fmt(r['clean_combo'])}")
for g, d in r['by_gen'].items(): print(f'  {g:7} {fmt(d)}')
for t, d in r['by_task'].items(): print(f'  {t:9} {fmt(d)}')
for gt, d in r['by_gen_task'].items(): print(f'  {gt:18} {fmt(d)}')
print(f"보고: F1 base 단독(임계값 {THR_F:.3f}) {fmt(r['clean_f1'])} | 엄격 멀쩡(모두 S) {fmt(r['strict_clean_combo'])}")
print(f"보고: 환각 응답 탐지 아무 지점 {fmt(r['hal_any'])} / 첫 H 문장 이후 {fmt(r['hal_strict'])} | 표시된 응답 중 H 응답 {fmt(r['flag_precision_H'])}")
print(f"\n=== M1-0 유효성: 정답 확보 {label_ratio:.3f} (≥ 0.90), 멀쩡한 응답 {r['clean_combo']['n']} (≥ 200) → {'충족' if M1_0 else '미달 → 판정 보류'} ===")
print(f"=== M1-1: 멀쩡한 응답 표시율 {r['clean_combo']['rate']:.3f} (≤ 0.15) → {'충족' if M1_1 else ('미달' if M1_0 else '판정 보류')} ===")
""")

md("## 7. M1-2 — M1-1 미달일 때만: cal-L로 임계값 재보정 → dev-L 확인")
code(r"""
M1_2, THR_NEW = None, None
if M1_0 and not M1_1:
    CASES['calL'] = prepare_split('calL')
    RUN['calL'] = run_split('calL')
    CAL = response_rows(CASES['calL'], *RUN['calL'], THR_C, THR_F)
    cal_clean = CAL[(CAL.status == 'ok') & CAL.clean]
    THR_NEW = conformal(cal_clean.max_c)
    DEV2 = response_rows(CASES['devL'], *RUN['devL'], THR_NEW, THR_F)
    REP['calL'], REP['devL_recal'] = report(CAL), report(DEV2)
    r2 = REP['devL_recal']
    M1_2 = bool(r2['clean_combo']['rate'] <= 0.15)
    print(f"cal-L 멀쩡한 응답 {len(cal_clean)}개 | 기존 임계값에서 cal-L 표시율 {fmt(REP['calL']['clean_combo'])}")
    print(f'재보정 임계값 {THR_NEW:.4f} (기존 {THR_C:.4f})')
    print(f"dev-L 재보정: 멀쩡한 응답 표시율 {fmt(r2['clean_combo'])} | 환각 응답 탐지 {fmt(r2['hal_any'])} / 첫 H 이후 {fmt(r2['hal_strict'])}")
    print(f"=== M1-2: {'재보정 임계값 후보' if M1_2 else '임계값 재보정으로 해결 안 됨'} ===")
else:
    print('M1-2 실행 안 함 (M1-1 충족 또는 판정 보류)')
""")

md("## 8. 저장")
code(r"""
usage = {}
for f in os.listdir(M1_DIR):
    if f.startswith(('gen_', 'label_')) and f.endswith('.json') and '_batches' not in f:
        usage[f] = round(cost(load_json(os.path.join(M1_DIR, f))), 2)
summ = {'gpu': GPU, 'thr_combo': THR_C, 'thr_f1': THR_F, 'combo': COMBO, 'gens': GENS, 'labeler': LABELER,
        'top_k': TOP_K, 'chunk_words': CHUNK_W, 'overlap': OVERLAP, 'n_sources': {k: len(v) for k, v in SIDS.items()},
        'label_ratio': label_ratio, 'M1_0': M1_0, 'M1_1': M1_1, 'M1_2': M1_2, 'thr_recal': THR_NEW,
        'report': REP, 'cost_usd': usage}
LOCAL = '/content/m1_local'; os.makedirs(LOCAL, exist_ok=True)
json.dump(summ, open(os.path.join(LOCAL, 'm1_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
DEV.to_csv(os.path.join(LOCAL, 'm1_devL.csv'), index=False)
if 'calL' in RUN: CAL.to_csv(os.path.join(LOCAL, 'm1_calL.csv'), index=False)
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(M1_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
print('API 사용량(추정 $)', usage)
zp = shutil.make_archive('/content/m1_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
def _src(c): return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
# API 셀(키·배치 제출·비용)은 K1 합성 노트북도 그대로 쓴다
M1_API = next(_src(c) for c in cells if _src(c).startswith('import anthropic')).split('# 형식 확인')[0]

def fresh_cells():
    # K1 평가 세트: 같은 근거·두 모델·같은 흐름으로 다시 생성(누구도 보지 않은 응답). 드라이브 m1_fresh/, M1-2 없음
    reps = [("os.path.join(OUT, 'm1')", "os.path.join(OUT, 'm1_fresh')"),
            ("if M1_0 and not M1_1:", "if False:                    # m1_fresh: 재보정(M1-2) 없음"),
            ("/content/m1_results", "/content/m1_fresh_results"), ("/content/m1_local", "/content/m1_fresh_local")]
    out = [{"cell_type": "markdown", "metadata": {}, "source":
            "# M1 새 생성 — K1 평가 세트 (`m1_fresh/`)\n\n"
            "아래는 M1 노트북과 **같은 코드**를 드라이브 `m1_fresh/`에 다시 돌린다. 같은 dev-L 근거·두 모델·같은 흐름으로 **응답을 새로 생성**해 "
            "누구도 보지 않은 평가 세트를 만든다(SSOT §6 K1 주 평가). 재보정(M1-2)은 하지 않는다. "
            "출력되는 M1-0·M1-1 판정은 **참고용**이며 M1 결과를 바꾸지 않는다. 기존 조합의 점수는 이 실행에서 저장되고, K1 학습 노트북이 새 판정기 점수를 더한다.\n\n"
            "> 런타임 GPU(아무거나), `ANTHROPIC_API_KEY` 필요, 비용 약 $5(M1 실측 $4.95). 끝나면 `m1_fresh_results.zip`을 내려받는다(올릴 필요 없음 — K1 학습 노트북이 드라이브에서 읽음)."}]
    for c in cells:
        c = dict(c); src = _src(c)
        for a, b in reps: src = src.replace(a, b)
        c['source'] = src; out.append(c)
    joined = ''.join(_src(c) for c in out)
    assert "'m1_fresh')" in joined and 'if False:' in joined and "os.path.join(OUT, 'm1')" not in joined
    return out

if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'm1_modern_llm_check.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
    out = os.path.join(NB_DIR, 'm1_fresh_eval.ipynb')
    json.dump(dict(nb, cells=fresh_cells()), open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(fresh_cells()), 'cells')
