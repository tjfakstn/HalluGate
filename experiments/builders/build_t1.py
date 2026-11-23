"""T1 노트북 생성: 근거 다이어트(SPEC §2.9)의 LLM 입력 토큰 절감과 답변 품질. 사전 기준은 SSOT §6 T1(11/23 고정).
CPU 런타임(Colab 컴퓨팅 단위 불필요). 게이트웨이 코드(trim·store·context)를 그대로 품는다.
RAGTruth **train** 분할 QA 근거만 쓴다(test 분할은 열지 않음). 생성 = claude-haiku-5-5, 판정 = claude-opus-5-5(배치)."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
_spec = importlib.util.spec_from_file_location('build_m1', os.path.join(HERE, 'build_m1.py'))
M1 = importlib.util.module_from_spec(_spec); _argv = sys.argv; sys.argv = ['import']; _spec.loader.exec_module(M1); sys.argv = _argv

SRC = {n: open(os.path.join(ROOT, 'hallugate', f'{n}.py'), encoding='utf-8').read() for n in ['store', 'context', 'trim']}
T1_API = M1.M1_API
assert 'def run_batch(' in T1_API

# ---- 아래 T1_* 문자열은 smoke_t1.py가 가짜 데이터·가짜 API 결과로 그대로 실행해 확인한다 ----
T1_DATA = r"""
import pandas as pd
N_ITEMS, N_DISTRACT, SEED = 200, 7, 2037          # 사전 고정 (SSOT §6 T1)

def train_qa_sources(responses, sources):
    # RAGTruth train 분할 응답만 있는 QA 근거 (test 분할 근거는 쓰지 않음)
    split_of = {}
    for r in responses:
        split_of.setdefault(r['source_id'], set()).add(r['split'])
    out = [s for s in sources if s['task_type'] == 'QA' and split_of.get(s['source_id']) == {'train'}]
    assert all(split_of[s['source_id']] == {'train'} for s in out)
    return sorted(out, key=lambda s: str(s['source_id']))

def make_items(pool):
    rng = random.Random(SEED)
    picked = rng.sample(pool, min(N_ITEMS, len(pool)))
    items = []
    for i, s in enumerate(picked):
        others = rng.sample([o for o in pool if o['source_id'] != s['source_id']], N_DISTRACT)
        blocks = [s['source_info']['passages']] + [o['source_info']['passages'] for o in others]
        order = list(range(len(blocks))); rng.shuffle(order)
        items.append({'i': i, 'sid': s['source_id'], 'q': s['source_info']['question'],
                      'own': s['source_info']['passages'],
                      'ref_L': '\n\n'.join(blocks[k] for k in order), 'ref_S': s['source_info']['passages']})
    return items
"""

T1_TRIM = r"""
from hallugate.trim import Trimmer
from hallugate.context import rag_messages
from types import SimpleNamespace
TOP_K, TOP_KS_REPORT = 4, [2, 4, 8]
GEN_MODEL, LABELER = 'claude-haiku-5-5', 'claude-opus-5-5'

def ngrams(t, n=8):
    w = t.lower().split()
    return {tuple(w[k:k + n]) for k in range(max(len(w) - n + 1, 0))}

def trim_all(items, trimmer, ntok):
    rows = []
    for it in items:
        own = ngrams(it['own'])
        for cond in ['L', 'S']:
            ref = it[f'ref_{cond}']
            row = {'i': it['i'], 'cond': cond, 'ref_tok_full': ntok(ref)}
            for k in TOP_KS_REPORT:
                kept, total = trimmer.select(ref, it['q'], k)
                new = '\n\n'.join(kept) if total > k else ref
                row[f'ref_tok_k{k}'] = ntok(new)
                row[f'own_recall_k{k}'] = len(ngrams(new) & own) / max(len(own), 1)   # 원래 지문이 얼마나 남았나 (보고)
                if k == TOP_K:
                    it[f'trim_{cond}'] = [new] if total <= k else kept
                    row['chunks_total'] = total
            rows.append(row)
    return pd.DataFrame(rows)

def gen_params(q, chunks):
    msgs = rag_messages([{'role': 'user', 'content': q}], [SimpleNamespace(text=c) for c in chunks])   # 게이트웨이와 같은 RAG 지시문
    return {'model': GEN_MODEL, 'max_tokens': 512, 'system': msgs[0]['content'], 'messages': msgs[1:]}

def gen_requests(items):
    R = {}
    for it in items:
        for cond in ['L', 'S']:
            R[f'g-{it["i"]}-{cond}-full'] = gen_params(it['q'], [it[f'ref_{cond}']])
            R[f'g-{it["i"]}-{cond}-trim'] = gen_params(it['q'], it[f'trim_{cond}'])
    return R

PAIR_SYSTEM = '''You compare two answers to the same question. The PASSAGES are the only source of truth.
Judge which answer better answers the QUESTION using correct information supported by the PASSAGES.
Prefer the answer that answers more completely and correctly. An answer that says it does not know when the PASSAGES do contain the answer is worse.
If both are equally good (or equally bad), say "same".
Return ONLY a JSON object: {"better": "1" | "2" | "same", "reason": "at most 20 words"}'''

def pair_params(it, cond, a1, a2):
    u = f"QUESTION: {it['q']}\n\nPASSAGES:\n<<<\n{it['own']}\n>>>\n\nANSWER 1:\n<<<\n{a1}\n>>>\n\nANSWER 2:\n<<<\n{a2}\n>>>"
    return {'model': LABELER, 'max_tokens': 300, 'system': PAIR_SYSTEM, 'messages': [{'role': 'user', 'content': u}]}

def trim_first(i, cond):
    # 다이어트 답을 1번에 둘지 (쌍마다 고정된 무작위, 판정자에게 가림)
    return random.Random(f'{SEED}-{i}-{cond}').random() < 0.5

def parse_pair(text):
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        b = str(d.get('better', '')).strip().lower()
        return b if b in ('1', '2', 'same') else None
    except Exception:
        return None

def judge_requests(items, G):
    P, F = {}, {}
    for it in items:
        for cond in ['L', 'S']:
            gf, gt = G.get(f'g-{it["i"]}-{cond}-full'), G.get(f'g-{it["i"]}-{cond}-trim')
            if not (gf and gt and gen_status(gf) == 'valid' and gen_status(gt) == 'valid'):
                continue
            a_full, a_trim = gf['text'], gt['text']
            first_trim = trim_first(it['i'], cond)
            P[f'p-{it["i"]}-{cond}'] = pair_params(it, cond, *((a_trim, a_full) if first_trim else (a_full, a_trim)))
            case = {'task': 'QA', 'ref_label': f"Question: {it['q']}\n{it['own']}"}
            for v, a in [('full', a_full), ('trim', a_trim)]:
                p = label_params(case, a); p['model'] = LABELER
                F[f'f-{it["i"]}-{cond}-{v}'] = p
    return P, F
"""

T1_METRICS = r"""
def outcome(i, cond, better):
    if better is None: return None
    if better == 'same': return 'SAME'
    trim_is_1 = trim_first(i, cond)
    return 'BETTER' if (better == '1') == trim_is_1 else 'WORSE'

def hal(res, key, text):
    r = res.get(key)
    if r is None: return None
    labs, ok = parse(r['text'], len(sentences(text)))
    return ('H' in [v['label'] for v in labs.values()]) if ok else None

def build_rows(items, G, PR, FR, T):
    rows = []
    for it in items:
        for cond in ['L', 'S']:
            kp = f'p-{it["i"]}-{cond}'
            if kp not in PR: continue
            gf, gt = G[f'g-{it["i"]}-{cond}-full'], G[f'g-{it["i"]}-{cond}-trim']
            rows.append({'i': it['i'], 'cond': cond, 'in_full': gf['in'], 'in_trim': gt['in'],
                         'out_full': gf['out'], 'out_trim': gt['out'],
                         'outcome': outcome(it['i'], cond, parse_pair(PR[kp]['text'])), 'trim_first': trim_first(it['i'], cond),
                         'hal_full': hal(FR, f'f-{it["i"]}-{cond}-full', gf['text']),
                         'hal_trim': hal(FR, f'f-{it["i"]}-{cond}-trim', gt['text'])})
    D = pd.DataFrame(rows).merge(T, on=['i', 'cond'], how='left')
    D['valid'] = D.outcome.notna() & D.hal_full.notna() & D.hal_trim.notna()
    return D

def paired_boot(d, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); x = (d.hal_trim.astype(float) - d.hal_full.astype(float)).values; vals = []
    for _ in range(n):
        vals.append(x[rng.integers(0, len(x), len(x))].mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def judge_t1(D):
    out = {}
    for cond in ['L', 'S']:
        d = D[(D.cond == cond) & D.valid]
        red = 1 - d.in_trim / d.in_full
        w = int((d.outcome == 'WORSE').sum()); n = len(d)
        p, lo, hi = wilson(w, n)
        dh = float(d.hal_trim.mean() - d.hal_full.mean()) if n else float('nan')
        out[cond] = {'n': n, 'input_reduction_median': float(red.median()) if n else None,
                     'ref_token_reduction_median': float((1 - d.ref_tok_k4 / d.ref_tok_full).median()) if n else None,
                     'worse': p, 'worse_ci': [lo, hi], 'same': float((d.outcome == 'SAME').mean()) if n else None,
                     'better': float((d.outcome == 'BETTER').mean()) if n else None,
                     'hal_full': float(d.hal_full.mean()) if n else None, 'hal_trim': float(d.hal_trim.mean()) if n else None,
                     'hal_diff': dh, 'hal_diff_ci': paired_boot(d) if n else None,
                     'own_recall_k': {k: float(d[f'own_recall_k{k}'].mean()) for k in TOP_KS_REPORT} if n else None,
                     'ref_tok_k': {k: float((1 - d[f'ref_tok_k{k}'] / d.ref_tok_full).median()) for k in TOP_KS_REPORT} if n else None,
                     'worse_by_order': {str(t): float((g.outcome == 'WORSE').mean()) for t, g in d.groupby('trim_first')} if n else None}
    L = out['L']
    out['T1_0'] = bool(out['L']['n'] >= 180 and out['S']['n'] >= 180)
    out['T1_1'] = bool(L['n'] and L['input_reduction_median'] >= 0.40)
    out['T1_2'] = bool(L['n'] and L['worse'] <= 0.10 and L['worse_ci'][1] <= 0.15)
    out['T1_3'] = bool(L['n'] and L['hal_diff'] <= 0.03 and L['hal_diff_ci'][1] < 0.08)
    out['T1'] = out['T1_0'] and out['T1_1'] and out['T1_2'] and out['T1_3']
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# T1 — 근거 다이어트: LLM 입력 토큰 절감과 답변 품질

## 왜 하나

게이트웨이가 사용자 서비스의 LLM 비용도 줄여 주면 붙일 이유가 늘어난다(사용자 요청, 11/23). RAG 비용의 대부분은 **근거 문서(입력 토큰)**다.
근거 다이어트(SPEC §2.9)는 긴 근거에서 질문과 관련된 조각만 LLM에 보내고 같은 조각으로 판정한다. 토큰은 줄지만 필요한 내용이 빠지면 답이 나빠질 수 있으므로, **절감과 품질을 함께** 잰다.

## 무엇을 하나 (학습 없음, CPU)

- **데이터**: RAGTruth **train** 분할 QA 근거 `Random(2037)` 200개(test 분할은 열지 않음). **L(주)** = 그 질문의 원래 지문 + 다른 QA 근거 7개의 지문(순서 섞음, 많은 문서를 통째로 넣는 상황), **S(보고)** = 원래 지문만
- **다이어트**: 게이트웨이 `Trimmer` 그대로(120단어 조각, 20 겹침, `all-MiniLM-L6-v2`, top_k=4)
- **생성**: `claude-haiku-5-5`(배치, max_tokens 512), 게이트웨이 RAG 지시문(§2.6) 그대로 — 전체 근거 vs 다이어트 근거
- **판정**(`claude-opus-5-5`, 배치): ① 쌍 비교(원래 지문 기준, 답 순서 무작위·어느 쪽이 다이어트인지 가림) → 다이어트 답 WORSE/SAME/BETTER, ② 충실성(R4 문장 S/H/A 지시문, 원래 지문 기준, H 하나라도 → 환각 응답)

## 사전 정의 기준 (SSOT §6 T1, 실행 전 고정)

- **T1-0(유효성)**: 조건마다 유효 쌍(생성·판정 모두 성공) ≥ 180
- **T1-1(절감, L)**: LLM 입력 토큰(업스트림 usage) 중앙값 감소율 ≥ 40%
- **T1-2(품질 주, L)**: WORSE 비율 ≤ 10%, Wilson 95% 상한 ≤ 15%
- **T1-3(충실성, L)**: 환각 응답 비율 다이어트 − 전체 ≤ +0.03, 쌍체 부트스트랩(2,000회, 시드 2037) 95% 상한 < +0.08
- **채택 = T1-0~3** → `trim`을 비용 절감 권장 옵션으로 문서화(여전히 선택 기능, 기본 top_k 4). 미달이면 기록하고 기본은 끈 채 둔다
- 보고만: S 조건, top_k 2·4·8의 근거 토큰 절감과 원래 지문 재현, BETTER 비율, 판정 순서 영향. **API 상한 $15**(제출 전 추정이 넘으면 멈춤)

> 런타임: **CPU**로 충분(GPU 불필요). Colab 비밀 값 `ANTHROPIC_API_KEY`. 끊기면 다시 모두 실행(드라이브 `t1/`에서 이어서). 끝나면 `t1_results.zip`(원문 없음).
""")

md("## 1. 환경 준비")
code("SRC = " + repr(SRC) + r"""
import os, sys, json, time, random, subprocess, shutil
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'anthropic', 'sentence-transformers', 'tiktoken', 'pysbd'], check=True)
subprocess.run('git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || true', shell=True)
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/hg/hallugate', exist_ok=True)
open('/content/hg/hallugate/__init__.py', 'w').write('')         # 패키지 __init__은 FastAPI 앱을 불러오므로 비워 둔다
for n, s in SRC.items(): open(f'/content/hg/hallugate/{n}.py', 'w').write(s)
sys.path.insert(0, '/content/hg')
OUT = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
T1_DIR = os.path.join(OUT, 't1'); os.makedirs(T1_DIR, exist_ok=True)
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
""")

md("## 2. RAGTruth train QA 근거 200개, L·S 근거 만들기")
code(T1_DATA + r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = [json.loads(l) for l in open(f'{BASE}/source_info.jsonl')]
POOL = train_qa_sources(responses, sources)
ITEMS = make_items(POOL)
print(f'train QA 근거 {len(POOL)}개 중 {len(ITEMS)}개, L 근거 평균 {np.mean([len(x["ref_L"]) for x in ITEMS]):.0f}자, S {np.mean([len(x["ref_S"]) for x in ITEMS]):.0f}자')
""")

md("## 3. 다이어트 (게이트웨이 Trimmer, top_k 2·4·8)")
code(T1_TRIM + r"""
import tiktoken
from sentence_transformers import SentenceTransformer
ENC = tiktoken.get_encoding('cl100k_base')
ntok = lambda t: len(ENC.encode(t, disallowed_special=()))
TRIMMER = Trimmer(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'), 120, 20)
T = trim_all(ITEMS, TRIMMER, ntok)
for cond, d in T.groupby('cond'):
    print(cond, '근거 토큰 중앙값', int(d.ref_tok_full.median()), '| top_k별 감소율',
          {k: round(float((1 - d[f'ref_tok_k{k}'] / d.ref_tok_full).median()), 3) for k in TOP_KS_REPORT},
          '| 원래 지문 재현', {k: round(float(d[f'own_recall_k{k}'].mean()), 3) for k in TOP_KS_REPORT})
""")

md("## 4. 생성 (Haiku, 배치) — 비용 추정 후 제출")
code("import pysbd\nSEG = pysbd.Segmenter(language='en', clean=False, char_span=True)\n" + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + "\n" + T1_API + r"""
BUDGET_USD = 15.0
REQ_G = gen_requests(ITEMS)
def est(reqs, out_tok):
    tot = 0.0
    for p in reqs.values():
        n_in = (len(p.get('system', '')) + sum(len(m['content']) for m in p['messages'])) / 3.5
        pin, pout = PRICE[p['model']]; tot += n_in / 1e6 * pin + out_tok / 1e6 * pout
    return tot
EST_G = est(REQ_G, 300)
EST_J = len(ITEMS) * 2 * 3 * (2500 * PRICE[LABELER][0] + 400 * PRICE[LABELER][1]) / 1e6      # 판정 추정 (쌍 1 + 충실성 2)
print(f'생성 요청 {len(REQ_G)}, 예상 ${EST_G:.2f} + 판정 약 ${EST_J:.2f} (상한 ${BUDGET_USD})')
assert EST_G + EST_J <= BUDGET_USD, '예상 비용이 상한을 넘음 — 멈춤'
G = run_batch(REQ_G, os.path.join(T1_DIR, 'gen.json'))
print('생성 상태', pd.Series([gen_status(G.get(k)) for k in REQ_G]).value_counts().to_dict())
""")

md("## 5. 판정 (Opus, 배치): 쌍 비교 + 충실성")
code(r"""
REQ_P, REQ_F = judge_requests(ITEMS, G)
PR = run_batch(REQ_P, os.path.join(T1_DIR, 'pair.json'))
FR = run_batch(REQ_F, os.path.join(T1_DIR, 'faith.json'))
print(f'쌍 {len(PR)}/{len(REQ_P)}, 충실성 {len(FR)}/{len(REQ_F)}')
""")

md("## 6. 기준 판정")
code(T1_METRICS + r"""
D = build_rows(ITEMS, G, PR, FR, T)
R = judge_t1(D)
for cond in ['L', 'S']:
    r = R[cond]
    print(f"[{cond}] 유효 {r['n']} | 입력 토큰 감소 중앙값 {r['input_reduction_median']:.3f} | WORSE {r['worse']:.3f} [{r['worse_ci'][0]:.3f}, {r['worse_ci'][1]:.3f}] "
          f"SAME {r['same']:.3f} BETTER {r['better']:.3f} | 환각 응답 전체 {r['hal_full']:.3f} → 다이어트 {r['hal_trim']:.3f} "
          f"({r['hal_diff']:+.3f} [{r['hal_diff_ci'][0]:+.3f}, {r['hal_diff_ci'][1]:+.3f}])")
print(f"\n=== T1-0 유효성 → {'충족' if R['T1_0'] else '미달'} ===")
print(f"=== T1-1 절감(L, ≥ 0.40) → {'충족' if R['T1_1'] else '미달'} ===")
print(f"=== T1-2 품질(L, WORSE ≤ 0.10, 상한 ≤ 0.15) → {'충족' if R['T1_2'] else '미달'} ===")
print(f"=== T1-3 충실성(L, 차이 ≤ +0.03, 상한 < +0.08) → {'충족' if R['T1_3'] else '미달'} ===")
print(f"=== T1 {'채택' if R['T1'] else '미채택'} ===")
""")

md("## 7. 저장")
code(r"""
usage = {f: round(cost(load_json(os.path.join(T1_DIR, f))), 2) for f in ['gen.json', 'pair.json', 'faith.json'] if os.path.exists(os.path.join(T1_DIR, f))}
print('API 사용량(배치 단가) $', usage, '합계', round(sum(usage.values()), 2))
LOCAL = '/content/t1_local'; os.makedirs(LOCAL, exist_ok=True)
D.to_csv(os.path.join(LOCAL, 't1_rows.csv'), index=False)          # 원문 없음 (토큰 수·판정 결과만)
json.dump(dict(R, cost_usd=usage, n_pool=len(POOL)), open(os.path.join(LOCAL, 't1_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL): shutil.copy(os.path.join(LOCAL, f), os.path.join(T1_DIR, f))
zp = shutil.make_archive('/content/t1_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 't1_trim_savings.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
