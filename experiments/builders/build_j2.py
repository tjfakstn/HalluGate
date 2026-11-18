"""J2 노트북 생성: 탐지 최대화 강한 검증기 — 오경보 20% 안에서 환각을 가장 많이 잡는 검증기를 고른다.
J1(11/18 미달) 후속. J1 후보는 "H 문장 하나라도 → 표시"라는 0/1 출력이라 오경보를 조절할 수 없었다.
J2는 모든 후보가 연속 점수를 내고(여러 번 판정 투표, 사실 단위 확률, MiniCheck 확률, 앙상블),
dev에서 오경보 20% 이하가 되는 임계값을 정한 뒤 test에서 판정한다. 사전 기준은 SSOT §6 J2.
평가 데이터 = LLM-AggreFact(사람 정답, CC BY-ND 4.0, 평가 전용). RAGTruth 부분은 파일을 읽을 때 걸러 아예 불러오지 않는다
(RAGTruth test 분할 보호). FaithBench는 J1에서 J-test를 본 뒤라 쓰지 않는다. 데이터는 저장소에 넣지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
_spec = importlib.util.spec_from_file_location('build_m1', os.path.join(HERE, 'build_m1.py'))
M1 = importlib.util.module_from_spec(_spec); _argv = sys.argv; sys.argv = ['import']; _spec.loader.exec_module(M1); sys.argv = _argv

J2_API = M1.M1_API                       # get_key, A, PRICE, text_of, rec, cost, run_batch
assert 'def run_batch(' in J2_API and "'claude-opus-5-5'" in J2_API
J2_PARSE = M1.LABEL_PARSE                # R4와 같은 문장 S/H/A 파싱
assert 'return labs, len(labs) == n' in J2_PARSE

# ---- 아래 J2_* 문자열은 smoke_j2.py가 가짜 데이터·가짜 API 결과로 그대로 실행해 확인한다 ----
J2_DATA = r"""
import pandas as pd, pyarrow.parquet as pq
EXCLUDE = 'RAGTruth'                     # RAGTruth test 분할 보호: 이 이름의 행은 읽지 않는다
N_DEV, N_TEST, SEED = 60, 100, 2036      # 데이터셋마다 dev 60·test 100 (사전 고정)

def load_split(path):
    # parquet 술어 필터로 RAGTruth 행을 읽기 단계에서 뺀다 (메모리에 올리지 않음)
    d = pq.read_table(path, filters=[('dataset', '!=', EXCLUDE)]).to_pandas()
    assert not (d.dataset == EXCLUDE).any() and len(d)
    return d

def sample(d, n, prefix, seed=SEED):
    out = pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in d.groupby('dataset', sort=True)]).reset_index(drop=True)
    out['id'] = [f'{prefix}{k}' for k in range(len(out))]
    out['hal'] = out.label.astype(int) == 0          # 0 = 근거가 뒷받침하지 않음 = 환각
    return out[['id', 'dataset', 'doc', 'claim', 'label', 'hal']]

def make_parts(dev_raw, test_raw):
    D, T = sample(dev_raw, N_DEV, 'd'), sample(test_raw, N_TEST, 't')
    D['part'], T['part'] = 'dev', 'test'
    assert not set(zip(D.doc, D.claim)) & set(zip(T.doc, T.claim)), 'dev·test에 같은 (문서, 주장)이 있음'
    return pd.concat([D, T]).reset_index(drop=True)
"""

J2_PROMPTS = r"""
import pysbd
SEG = pysbd.Segmenter(language='en', clean=False)
def sents(text):
    ss = [s.strip() for s in SEG.segment(text) if s.strip()]
    return ss or [text.strip()]

SYS_SENT = '''You check whether a RESPONSE is fully supported by a REFERENCE document.
You receive the REFERENCE and the RESPONSE split into numbered sentences. Judge EVERY sentence against the REFERENCE only (not world knowledge).

A sentence contains a hallucination if any part of it is:
- Evident conflict: clearly contradicts the reference (wrong fact, number, name, date, negation).
- Subtle conflict: deviates from the reference in a way that changes meaning (altered term, scope, time, attribution, certainty).
- Evident baseless info: states information that is not in the reference and cannot be directly inferred from it, even if it may be true in the real world.
- Subtle baseless info: adds unsupported inference, opinion, or embellishment that the reference does not support.

Not hallucinations: faithful paraphrase; reasonable summarization or generalization of what the reference says; statements that the reference lacks the information.

Labels: "S" = fully supported. "H" = contains a hallucination. "A" = ambiguous, careful annotators could reasonably disagree.
For every H or A sentence also give "type" (evident_conflict, subtle_conflict, evident_baseless, subtle_baseless), "phrase" (shortest hallucinated span copied exactly) and "reason" (at most 20 words).

Return ONLY a JSON object, no other text:
{"sentences": [{"i": 1, "label": "S"}, {"i": 2, "label": "H", "type": "evident_baseless", "phrase": "...", "reason": "..."}]}
with exactly one entry per numbered sentence, in order.'''

SYS_ATOM = '''You check whether a RESPONSE is fully supported by a REFERENCE document, fact by fact.
1. Break the RESPONSE into atomic facts (one checkable statement each). Cover every fact, including numbers, names, dates, qualifiers, causal links and opinions.
2. For each fact, decide against the REFERENCE only (not world knowledge):
   "supported" = stated in or directly inferable from the reference;
   "contradicted" = the reference says something incompatible;
   "not_found" = the reference does not establish it (even if it may be true in the real world).
3. Give p_unsupported: your probability (0-100) that AT LEAST ONE fact is contradicted or not_found. Be calibrated: use the full range.

Return ONLY a JSON object, no other text:
{"facts": [{"fact": "...", "verdict": "supported", "evidence": "at most 15 words"}], "p_unsupported": 0}'''

def user_sent(r):
    ss = sents(r.claim)
    lines = '\n'.join(f'[{i + 1}] {s}' for i, s in enumerate(ss))
    return f"REFERENCE:\n<<<\n{r.doc}\n>>>\n\nRESPONSE ({len(ss)} numbered sentences):\n{lines}"

def user_atom(r):
    return f"REFERENCE:\n<<<\n{r.doc}\n>>>\n\nRESPONSE:\n<<<\n{r.claim}\n>>>"

VOTES = {'haiku': ('claude-haiku-5-5', 5), 'sonnet': ('claude-sonnet-5-5', 3)}   # 문장 판정 투표 횟수 (사전 고정)
ATOM_MODEL = 'claude-opus-5-5'

def requests_all(P):
    R = {}
    for m, (model, k) in VOTES.items():
        R[m] = {f'{m}-{r.id}-{v}': {'model': model, 'max_tokens': 2048, 'system': SYS_SENT,
                                    'messages': [{'role': 'user', 'content': user_sent(r)}]}
                for r in P.itertuples() for v in range(k)}
    R['opus_atom'] = {f'opus_atom-{r.id}': {'model': ATOM_MODEL, 'max_tokens': 3000, 'system': SYS_ATOM,
                                             'messages': [{'role': 'user', 'content': user_atom(r)}]} for r in P.itertuples()}
    return R

def est_cost(reqs, out_tokens):
    # 요청 글자 수 / 3.5 ≈ 입력 토큰 (추정). 배치 단가
    tot = 0.0
    for p in reqs.values():
        n_in = (len(p['system']) + sum(len(m['content']) for m in p['messages'])) / 3.5
        pin, pout = PRICE[p['model']]
        tot += n_in / 1e6 * pin + out_tokens / 1e6 * pout
    return tot
OUT_TOK = {'haiku': 250, 'sonnet': 250, 'opus_atom': 700}

def parse_atom(text):
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        p = min(max(float(d['p_unsupported']), 0.0), 100.0) / 100
        bad = any(str(f.get('verdict', '')).lower() in ('contradicted', 'not_found') for f in d.get('facts', []))
        return p, bad, True
    except Exception:
        return 0.0, False, False
""" + J2_PARSE + r"""
def claude_scores(P, RES):
    # 투표 점수 = H(또는 H·A) 문장이 있다고 본 판정의 비율. 파싱 실패 판정 = 표시 안 함(탐지에 불리한 쪽)
    for m, (_, k) in VOTES.items():
        sH, sHA, ok = [], [], []
        for r in P.itertuples():
            n = len(sents(r.claim)); h = ha = o = 0
            for v in range(k):
                x = RES[m].get(f'{m}-{r.id}-{v}')
                labs, good = parse(x['text'], n) if x else ({}, False)
                ls = [e['label'] for e in labs.values()] if good else []
                h += 'H' in ls; ha += ('H' in ls or 'A' in ls); o += good
            sH.append(h / k); sHA.append(ha / k); ok.append(o / k)
        P[f's_{m}_H'], P[f's_{m}_HA'], P[f'ok_{m}'] = sH, sHA, ok
    pa = [parse_atom(RES['opus_atom'][f'opus_atom-{i}']['text']) if f'opus_atom-{i}' in RES['opus_atom'] else (0.0, False, False) for i in P.id]
    P['s_opus_atom'] = [p for p, _, _ in pa]; P['opus_atom_anybad'] = [b for _, b, _ in pa]; P['ok_opus_atom'] = [float(o) for _, _, o in pa]
    return P
"""

J2_METRICS = r"""
MAX_FLAG = 0.20                          # 오경보 상한 (사용자 결정, 사전 고정)
ORDER = ['haiku_H', 'haiku_HA', 'minicheck', 'sonnet_H', 'sonnet_HA', 'opus_atom', 'ens']   # 동점이면 싼 쪽
ENS_PARTS = ['sonnet_H', 'opus_atom', 'minicheck']                                          # 앙상블 = 유효한 것들의 평균

def thr_at(clean, allv, max_flag=MAX_FLAG):
    # 표시 = 점수 ≥ t. 멀쩡한 응답 표시율 ≤ max_flag인 가장 작은 t (dev에 나온 점수 값 중에서)
    clean = np.asarray(clean, float)
    for t in np.unique(np.concatenate([np.asarray(allv, float), [np.inf]])):
        if (clean >= t).mean() <= max_flag: return float(t)
    return float('inf')

def rates(d, col, t):
    f = d[col] >= t
    return float(f[d.hal].mean()), float(f[~d.hal].mean())

def select(P, cands):
    D = P[P.part == 'dev']; thr, ok = {}, []
    for c in cands:
        t = thr_at(D[~D.hal][f's_{c}'], D[f's_{c}']); thr[c] = t
        det, fl = rates(D, f's_{c}', t)
        print(f'  dev {c:10} 임계값 {t:.3f} | 탐지 {det:.3f} | 멀쩡한 응답 표시 {fl:.3f}')
        ok.append((det, -ORDER.index(c), c))
    return (max(ok)[2] if ok else None), thr

def strat_boot(T, fn, n=2000, seed=2036):
    # 데이터셋 안에서 항목을 다시 뽑는다 (데이터셋 구성 고정)
    rng = np.random.default_rng(seed); G = [g for _, g in T.groupby('dataset')]; vals = []
    for _ in range(n):
        d = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in G])
        v = fn(d)
        if v == v: vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def judge_j2(P, sel, thr, seed=2036):
    T = P[P.part == 'test']
    out = {'n_test': len(T), 'n_test_hal': int(T.hal.sum()), 'n_test_clean': int((~T.hal).sum()), 'selected': sel}
    if sel is None:
        out.update(J2_1=False, J2_2=False, J2=False); return out
    col, t = f's_{sel}', thr[sel]
    det, fl = rates(T, col, t)
    out.update(det=det, flag=fl, thr=t, bacc=(det + 1 - fl) / 2)
    out['det_ci'] = strat_boot(T, lambda d: float((d[d.hal][col] >= t).mean()), seed=seed)
    out['flag_ci'] = strat_boot(T, lambda d: float((d[~d.hal][col] >= t).mean()), seed=seed)
    out['J2_1'] = bool(det >= 0.60 and out['det_ci'][0] >= 0.50)
    out['J2_2'] = bool(fl <= MAX_FLAG)
    out['J2'] = out['J2_1'] and out['J2_2']
    out['by_dataset'] = {ds: dict(zip(['det', 'flag'], rates(g, col, t)), n_hal=int(g.hal.sum()), n_clean=int((~g.hal).sum()))
                         for ds, g in T.groupby('dataset')}
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# J2 — 탐지 최대화 강한 검증기 (LLM-AggreFact 사람 정답)

## 왜 하나

J1(11/18)은 미달이었다. 강한 검증기 후보(Haiku·Sonnet·Opus, 문장 S/H/A 한 번 판정)가 모두 J-cal 오경보 15%를 넘었고, Opus는 FaithBench J-test에서 환각 54%를 잡으면서 멀쩡한 응답 18%를 잘못 표시했다.
J1 후보는 "H 문장 하나라도 → 표시"라는 0/1 출력이라 **오경보와 탐지의 균형을 조절할 수 없었다**.
사용자 결정(11/18): **환각을 최대한 잡는 방향, 오경보는 20%까지 받아들인다.** 그래서 모든 후보가 연속 점수를 내게 하고, 오경보 20% 안에서 탐지가 가장 높은 지점을 고른다.

## 무엇을 하나 (학습 없음)

- **데이터**: LLM-AggreFact(사람 정답 문서–주장 쌍, label 0 = 근거가 뒷받침 안 함 = 환각). **RAGTruth 부분은 parquet 술어 필터로 읽지 않는다.** 나머지 10개 데이터셋에서 `random_state=2036`으로 데이터셋마다 **dev 60**(후보 선택·임계값)·**test 100**(판정)
- **후보(모두 연속 점수)**:
  - `haiku_H` / `haiku_HA`: `claude-haiku-5-5` 문장 S/H/A 판정 5회 투표, 점수 = H(또는 H·A) 문장이 있다고 본 비율
  - `sonnet_H` / `sonnet_HA`: `claude-sonnet-5-5` 같은 지시문 3회 투표
  - `opus_atom`: `claude-opus-5-5`가 사실 단위로 쪼개 근거와 대조하고 "근거 없는 사실이 하나라도 있을 확률"(0–100)을 낸다
  - `minicheck`: `Bespoke-MiniCheck-7B` 지지 확률, 점수 = 1 − 확률. 7B가 안 돌면 `flan-t5-large`(MiniCheck)로 대신하고 어느 것인지 기록. 둘 다 안 되면 후보에서 빠진다
  - `ens`: `sonnet_H`·`opus_atom`·`minicheck` 중 유효한 것의 점수 평균
- 문장 지시문은 R4를 일반 문서용으로 고친 것(과제 설명 없음, "근거에서 바로 추론되지 않는 정보 = 근거 없음")

## 사전 정의 기준 (SSOT §6 J2, 실행 전 고정)

- **J2-0(유효성)**: 10개 데이터셋 로드·RAGTruth 행 0개. Claude 후보는 dev·test 각각 판정 파싱 성공 ≥ 95%(미달 후보와 그것을 포함한 앙상블 구성요소는 뺀다)
- **임계값·선택(dev)**: 후보마다 dev 멀쩡한 응답 표시율 ≤ 20%가 되는 가장 낮은 점수 임계값. 그 임계값에서 dev 탐지율이 가장 높은 후보를 고른다(동점이면 haiku_H → haiku_HA → minicheck → sonnet_H → sonnet_HA → opus_atom → ens)
- **J2-1(주, test)**: 선택 후보의 환각 탐지율 ≥ **0.60**, **그리고** 데이터셋 층화 부트스트랩 95% 하한 ≥ **0.50** (2,000회, 시드 2036)
- **J2-2(test)**: 선택 후보의 멀쩡한 응답 표시율 ≤ **0.20**
- **J2 채택 = J2-1 그리고 J2-2** → 게이트웨이에 응답 끝 2단 검증을 넣는다(SPEC부터). 미달이면 서비스 주장을 줄인다
- **J2-3(지연, 방식 결정)**: 선택 후보 응답 1개 검증 p95(dev 30개, 실시간, 투표는 병렬) ≤ 10초면 hold 가능, 넘으면 응답 뒤 경고(비동기)
- 보고만: 모든 후보의 dev·test 탐지·표시율, 데이터셋별, balanced accuracy, `opus_atom` 사실 판정 기준 0/1, 비용. **API 비용 상한 $50**(제출 전 추정 > $47이면 멈춤, $3은 지연 측정·형식 확인 몫)

> 런타임: **GPU**(MiniCheck용, L4·A100). Colab 비밀 값 `ANTHROPIC_API_KEY`, **`HF_TOKEN`**(LLM-AggreFact는 Hugging Face에서 이용 조건 동의가 필요한 데이터셋 — 먼저 https://huggingface.co/datasets/lytang/LLM-AggreFact 에서 동의).
> 끊기면 다시 모두 실행(배치·점수는 드라이브 `j2/`에서 이어서). 끝나면 `j2_results.zip`(원문 없음)이 내려받아진다.
""")

md("## 1. 환경 준비")
code(r"""
import os, sys, json, time, random, subprocess, shutil, traceback
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'anthropic', 'pysbd', 'pyarrow', 'huggingface_hub'], check=True)
import numpy as np, torch
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
J2_DIR = os.path.join(OUT, 'j2'); os.makedirs(J2_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
print('device', GPU)
""")

md("## 2. API 키, LLM-AggreFact (RAGTruth 제외), dev·test 표본")
code(J2_API + "\n" + J2_DATA + r"""
from huggingface_hub import hf_hub_download
HF = get_key('HF_TOKEN')
paths = {s: hf_hub_download('lytang/LLM-AggreFact', f'data/{s}-00000-of-00001.parquet', repo_type='dataset', token=HF) for s in ['dev', 'test']}
P = make_parts(load_split(paths['dev']), load_split(paths['test']))
assert P.dataset.nunique() == 10, sorted(P.dataset.unique())
print(P.groupby(['part', 'dataset']).hal.agg(['size', 'sum']).rename(columns={'size': 'n', 'sum': '환각'}))
print(P.groupby('part').hal.agg(['size', 'sum']))
""")

md("## 3. Claude 후보 — 비용 추정, 형식 확인, 배치")
code(J2_PROMPTS + r"""
BUDGET_USD, RESERVE = 50.0, 3.0
REQ = requests_all(P)
EST = {m: est_cost(r, OUT_TOK[m]) for m, r in REQ.items()}
print('요청 수', {m: len(r) for m, r in REQ.items()}, '| 예상 비용 $', {m: round(v, 2) for m, v in EST.items()}, '합계', round(sum(EST.values()), 2))
assert sum(EST.values()) <= BUDGET_USD - RESERVE, '예상 비용이 상한을 넘음 — 멈춤'
_r = P.iloc[0]
for m, reqs in REQ.items():                          # 형식 확인 1건씩 (실시간)
    x = rec(A.messages.create(**next(iter(v for k, v in reqs.items() if k.startswith(f'{m}-{_r.id}')))))
    print(f"  형식 확인 {m}: 입력 {x['in']} 출력 {x['out']} | " +
          (str(parse_atom(x['text'])) if m == 'opus_atom' else str(parse(x['text'], len(sents(_r.claim)))[1])) + ' | ' + x['text'][:160].replace('\n', ' '))
RES = {}
for m in ['haiku', 'sonnet', 'opus_atom']:          # 싼 것부터
    RES[m] = run_batch(REQ[m], os.path.join(J2_DIR, f'v_{m}.json'))
P = claude_scores(P, RES)
for m in ['haiku', 'sonnet', 'opus_atom']:
    print(f"  {m}: 파싱 성공 dev {P[P.part == 'dev'][f'ok_{m}'].mean():.3f}, test {P[P.part == 'test'][f'ok_{m}'].mean():.3f}")
""")

md("## 4. MiniCheck (7B → 안 되면 flan-t5-large → 안 되면 제외)")
code(r"""
MC_MODEL = None
mp = os.path.join(J2_DIR, 'minicheck.json'); MC = load_json(mp) or {}
if MC.get('_model') and all(i in MC for i in P.id):
    MC_MODEL = MC['_model']
else:
    try:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main'], check=True)
    except Exception:
        traceback.print_exc()
    for name in ['Bespoke-MiniCheck-7B', 'flan-t5-large']:
        try:
            from minicheck.minicheck import MiniCheck
            kw = {'enable_prefix_caching': False} if name.startswith('Bespoke') else {}
            mc = MiniCheck(model_name=name, cache_dir='/content/ckpts', **kw)
            todo = [r for r in P.itertuples() if r.id not in MC]
            for k in range(0, len(todo), 100):
                part = todo[k:k + 100]
                _, prob, _, _ = mc.score(docs=[r.doc for r in part], claims=[r.claim for r in part])
                for r, p in zip(part, prob): MC[r.id] = float(p)
                MC['_model'] = name; save_json(MC, mp); print(f'  {name} {min(k + 100, len(todo))}/{len(todo)}', flush=True)
            MC_MODEL = name; break
        except Exception:
            print(f'MiniCheck {name} 실행 안 됨:'); traceback.print_exc()
            MC = {}                               # 실패한 모델의 부분 결과는 버림
if MC_MODEL:
    P['s_minicheck'] = [1 - MC[i] for i in P.id]; print('MiniCheck 모델:', MC_MODEL)
""")

md("## 5. 후보 정리 → dev에서 임계값·선택 → test 판정")
code(J2_METRICS + r"""
valid = {}
for m in ['haiku', 'sonnet', 'opus_atom']:
    valid[m] = bool(P[P.part == 'dev'][f'ok_{m}'].mean() >= 0.95 and P[P.part == 'test'][f'ok_{m}'].mean() >= 0.95)
CANDS = [c for c in ['haiku_H', 'haiku_HA', 'sonnet_H', 'sonnet_HA'] if valid[c.split('_')[0]]]
if valid['opus_atom']: CANDS.append('opus_atom')
if MC_MODEL: CANDS.append('minicheck')
parts = [c for c in ENS_PARTS if c in CANDS]
if len(parts) >= 2:
    P['s_ens'] = P[[f's_{c}' for c in parts]].mean(axis=1); CANDS.append('ens')
CANDS = sorted(CANDS, key=ORDER.index)
print('J2-0 통과 후보:', CANDS, '| 앙상블 구성:', parts)
SEL, THR = select(P, CANDS)
R = judge_j2(P, SEL, THR)
T = P[P.part == 'test']
print(f"\ntest: {R['n_test']} (환각 {R['n_test_hal']}, 멀쩡 {R['n_test_clean']})")
REP = {}
for c in CANDS:
    d, f = rates(T, f's_{c}', THR[c]); REP[c] = {'thr': THR[c], 'det': d, 'flag': f}
    print(f'  (보고) test {c:10} 탐지 {d:.3f} | 표시 {f:.3f}')
if 'opus_atom' in CANDS:
    REP['opus_atom_anybad'] = dict(zip(['det', 'flag'], rates(T.assign(s_x=T.opus_atom_anybad.astype(float)), 's_x', 0.5)))
    print('  (보고) opus 사실 판정 0/1:', REP['opus_atom_anybad'])
if SEL is None:
    print('\n=== J2 미달: 유효한 후보 없음 ===')
else:
    print(f"\n=== 선택: {SEL} (dev 임계값 {THR[SEL]:.3f}) ===")
    print(f"=== J2-1(주): 탐지율 {R['det']:.3f} [{R['det_ci'][0]:.3f}, {R['det_ci'][1]:.3f}] (≥ 0.60, 하한 ≥ 0.50) → {'충족' if R['J2_1'] else '미달'} ===")
    print(f"=== J2-2: 멀쩡한 응답 표시율 {R['flag']:.3f} [{R['flag_ci'][0]:.3f}, {R['flag_ci'][1]:.3f}] (≤ 0.20) → {'충족' if R['J2_2'] else '미달'} ===")
    print(f"=== J2 {'채택' if R['J2'] else '미채택'} === (balanced accuracy {R['bacc']:.3f})")
    for ds, v in R['by_dataset'].items(): print(f"  (보고) {ds:18} 탐지 {v['det']:.3f} (환각 {v['n_hal']}) | 표시 {v['flag']:.3f} (멀쩡 {v['n_clean']})")
""")

md("## 6. J2-3 지연 — 선택 후보 응답 1개 검증 시간 (dev 30개, 실시간)")
code(r"""
from concurrent.futures import ThreadPoolExecutor
LAT = None
if SEL:
    need = {'haiku_H': ['haiku'], 'haiku_HA': ['haiku'], 'sonnet_H': ['sonnet'], 'sonnet_HA': ['sonnet'], 'opus_atom': ['opus_atom'],
            'minicheck': [], 'ens': [c.split('_')[0] if c != 'opus_atom' else c for c in parts if c != 'minicheck']}[SEL]
    use_mc = SEL == 'minicheck' or (SEL == 'ens' and 'minicheck' in parts)
    rows = list(P[P.part == 'dev'].itertuples())[:30]; LAT = []
    for r in rows:
        calls = [v for m in need for k, v in REQ[m].items() if k.startswith(f'{m}-{r.id}')]
        t1 = time.perf_counter()
        with ThreadPoolExecutor(max(len(calls), 1)) as ex: list(ex.map(lambda p: A.messages.create(**p), calls))
        ms = (time.perf_counter() - t1) * 1000
        if use_mc and 'mc' in globals():
            t2 = time.perf_counter(); mc.score(docs=[r.doc], claims=[r.claim]); ms = max(ms, (time.perf_counter() - t2) * 1000)
        LAT.append(ms)
    p50, p95 = np.percentile(LAT, 50), np.percentile(LAT, 95)
    R['latency_ms'] = {'n': len(LAT), 'p50': float(p50), 'p95': float(p95)}; R['J2_3'] = bool(p95 <= 10000)
    print(f"=== J2-3: {SEL} p50 {p50:.0f}ms, p95 {p95:.0f}ms (≤ 10000) → {'hold 가능' if R['J2_3'] else '응답 뒤 경고(비동기)'} ===")
""")

md("## 7. 저장")
code(r"""
usage = {f: round(cost(load_json(os.path.join(J2_DIR, f))), 2) for f in os.listdir(J2_DIR) if f.startswith('v_') and f.endswith('.json') and '_batches' not in f}
print('API 사용량(배치 단가, 지연 측정·형식 확인 제외) $', usage, '합계', round(sum(usage.values()), 2))
LOCAL = '/content/j2_local'; os.makedirs(LOCAL, exist_ok=True)
cols = ['id', 'dataset', 'part', 'label', 'hal'] + [c for c in P.columns if c.startswith(('s_', 'ok_'))] + (['opus_atom_anybad'] if 'opus_atom_anybad' in P else [])
P[cols].to_csv(os.path.join(LOCAL, 'j2_rows.csv'), index=False)        # 원문(문서·주장) 없음
json.dump(dict(R, cands=CANDS, thr=THR, report=REP, ens_parts=parts, minicheck_model=MC_MODEL, est_usd=EST, cost_usd=usage, gpu=GPU),
          open(os.path.join(LOCAL, 'j2_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(J2_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/j2_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'j2_max_detection.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
