"""T1b 노트북 생성: 근거 다이어트 품질 재설계 (SSOT §6 T1b, 11/23 고정).
T1(미달)에서 쌍 비교가 생성 무작위성·우열 강제 때문에 대조군도 통과 못 한 결함을 고친다:
답마다 절대 평가(yes/partial/no) + 같은 입력 재생성 대조군(full2), temperature 0, max_tokens 1024.
T1의 질문 200개는 빼고 Random(2038)로 새로 뽑는다. CPU 런타임, RAGTruth train 분할만(test 분할은 열지 않음)."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
_spec = importlib.util.spec_from_file_location('build_t1', os.path.join(HERE, 'build_t1.py'))
T1 = importlib.util.module_from_spec(_spec); _argv = sys.argv; sys.argv = ['import']; _spec.loader.exec_module(T1); sys.argv = _argv
M1 = T1.M1

# ---- 아래 T1B_* 문자열은 smoke_t1b.py가 가짜 데이터·가짜 API 결과로 그대로 실행해 확인한다 ----
T1B_DATA = r"""
N_ITEMS_B, SEED_B = 200, 2038                        # 사전 고정 (SSOT §6 T1b)

def make_items_b(pool, exclude):
    # T1에서 결과를 본 질문(exclude)은 빼고 새로 뽑는다. 근거는 T1의 L과 같은 방식
    rng = random.Random(SEED_B)
    cand = [s for s in pool if s['source_id'] not in exclude]
    picked = rng.sample(cand, min(N_ITEMS_B, len(cand)))
    items = []
    for i, s in enumerate(picked):
        others = rng.sample([o for o in pool if o['source_id'] != s['source_id']], N_DISTRACT)
        blocks = [s['source_info']['passages']] + [o['source_info']['passages'] for o in others]
        order = list(range(len(blocks))); rng.shuffle(order)
        items.append({'i': i, 'sid': s['source_id'], 'q': s['source_info']['question'], 'own': s['source_info']['passages'],
                      'ref_L': '\n\n'.join(blocks[k] for k in order), 'ref_S': s['source_info']['passages']})
    assert not {x['sid'] for x in items} & set(exclude), 'T1 질문이 섞임'
    return items
"""

T1B_GEN = r"""
ARMS = ['full', 'full2', 'trim']                     # full2 = 같은 입력 재생성 대조군

def gen_params_b(q, chunks, use_temp):
    p = gen_params(q, chunks); p['max_tokens'] = 1024
    if use_temp: p['temperature'] = 0.0
    return p

def gen_requests_b(items, use_temp=True):
    R = {}
    for it in items:
        R[f'g-{it["i"]}-full'] = gen_params_b(it['q'], [it['ref_L']], use_temp)
        R[f'g-{it["i"]}-full2'] = gen_params_b(it['q'], [it['ref_L']], use_temp)
        R[f'g-{it["i"]}-trim'] = gen_params_b(it['q'], it['trim_L'], use_temp)
    return R

GRADE_SYSTEM = '''You grade ONE answer to a question. The PASSAGES are the only source of truth.
"yes": the answer correctly answers the QUESTION with information supported by the PASSAGES (minor omissions are fine).
"partial": the answer is on the right track but misses an important part of what the PASSAGES say, or mixes correct and incorrect points.
"no": the answer does not answer the question, is wrong, or says the information is unavailable while the PASSAGES do contain it.
If the PASSAGES truly do not contain the answer, an answer that says so is "yes".
Return ONLY a JSON object: {"grade": "yes" | "partial" | "no", "reason": "at most 20 words"}'''

def grade_params(it, ans):
    u = f"QUESTION: {it['q']}\n\nPASSAGES:\n<<<\n{it['own']}\n>>>\n\nANSWER:\n<<<\n{ans}\n>>>"
    return {'model': LABELER, 'max_tokens': 300, 'system': GRADE_SYSTEM, 'messages': [{'role': 'user', 'content': u}]}

def parse_grade(text):
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        g = str(d.get('grade', '')).strip().lower()
        return g if g in ('yes', 'partial', 'no') else None
    except Exception:
        return None

def judge_requests_b(items, G):
    # 세 팔이 모두 정상 생성된 질문만. 평가 요청에는 어느 팔인지 드러나지 않는다(답 하나씩)
    Q, F = {}, {}
    for it in items:
        gs = {a: G.get(f'g-{it["i"]}-{a}') for a in ARMS}
        if not all(g and gen_status(g) == 'valid' for g in gs.values()):
            continue
        case = {'task': 'QA', 'ref_label': f"Question: {it['q']}\n{it['own']}"}
        for a, g in gs.items():
            Q[f'q-{it["i"]}-{a}'] = grade_params(it, g['text'])
            p = label_params(case, g['text']); p['model'] = LABELER
            F[f'f-{it["i"]}-{a}'] = p
    return Q, F
"""

T1B_METRICS = r"""
def build_rows_b(items, G, QR, FR, T):
    rows = []
    for it in items:
        row = {'i': it['i']}
        ok = True
        for a in ARMS:
            g = G.get(f'g-{it["i"]}-{a}')
            if not g or gen_status(g) != 'valid':
                ok = False; break
            q = QR.get(f'q-{it["i"]}-{a}')
            row[f'in_{a}'], row[f'out_{a}'] = g['in'], g['out']
            row[f'trunc_{a}'] = g.get('stop') == 'max_tokens'
            row[f'grade_{a}'] = parse_grade(q['text']) if q else None
            row[f'hal_{a}'] = hal(FR, f'f-{it["i"]}-{a}', g['text'])
        if ok:
            rows.append(row)
    D = pd.DataFrame(rows).merge(T[T.cond == 'L'].drop(columns=['cond']), on='i', how='left')
    D['valid'] = D[[f'grade_{a}' for a in ARMS] + [f'hal_{a}' for a in ARMS]].notna().all(axis=1)
    return D

def paired_boot_b(x, n=2000, seed=SEED_B):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    vals = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def judge_t1b(D):
    d = D[D.valid]; n = len(d)
    yes = {a: (d[f'grade_{a}'] == 'yes').astype(float) for a in ARMS}
    hal_ = {a: d[f'hal_{a}'].astype(float) for a in ARMS}
    out = {'n': n,
           'yes': {a: float(yes[a].mean()) for a in ARMS},
           'partial': {a: float((d[f'grade_{a}'] == 'partial').mean()) for a in ARMS},
           'no': {a: float((d[f'grade_{a}'] == 'no').mean()) for a in ARMS},
           'hal': {a: float(hal_[a].mean()) for a in ARMS},
           'trunc': {a: float(d[f'trunc_{a}'].mean()) for a in ARMS},
           'input_reduction_median': float((1 - d.in_trim / d.in_full).median()),
           'own_recall_k4': float(d.own_recall_k4.mean())}
    out['control_yes_diff'] = float((yes['full2'] - yes['full']).mean())
    out['control_yes_ci'] = paired_boot_b(yes['full2'] - yes['full'])
    out['control_hal_diff'] = float((hal_['full2'] - hal_['full']).mean())
    out['yes_diff'] = float((yes['trim'] - yes['full']).mean())
    out['yes_diff_ci'] = paired_boot_b(yes['trim'] - yes['full'])
    out['hal_diff'] = float((hal_['trim'] - hal_['full']).mean())
    out['hal_diff_ci'] = paired_boot_b(hal_['trim'] - hal_['full'])
    out['T1b_0'] = bool(n >= 180 and abs(out['control_yes_diff']) <= 0.05)
    out['T1b_1'] = bool(out['input_reduction_median'] >= 0.40)
    out['T1b_2'] = bool(out['yes_diff'] >= -0.05 and out['yes_diff_ci'][0] >= -0.10)
    out['T1b_3'] = bool(out['hal_diff'] <= 0.03 and out['hal_diff_ci'][1] < 0.08)
    out['T1b'] = out['T1b_0'] and out['T1b_1'] and out['T1b_2'] and out['T1b_3']
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# T1b — 근거 다이어트 품질 재설계 (대조군 + 답마다 절대 평가)

## 왜 하나

T1(11/23)에서 근거 다이어트는 LLM 입력 토큰을 **−73%** 줄였지만 품질 기준(쌍 비교 WORSE ≤ 10%)에서 미달했다(37.4%).
그런데 다이어트가 거의 없던 S 조건도 WORSE 36%였다 — 같은 입력을 두 번 생성해도 판정자가 한쪽을 고르는 **바닥값**이었고, 답의 31~41%가 512토큰에서 잘렸다.
기준 설계의 결함이므로, 대조군을 넣고 답마다 따로 평가해 **"다이어트가 품질을 떨어뜨리는가"**를 다시 잰다.

## 무엇을 하나 (학습 없음, CPU)

- **데이터**: RAGTruth train QA 근거 중 **T1의 200개를 뺀** 풀에서 `Random(2038)` 200개. 근거는 T1의 L과 같은 방식(원래 지문 + 다른 근거 7개)
- **팔 셋**(`claude-haiku-5-5`, 배치, **temperature 0, max_tokens 1024**): `full`(전체 근거), `full2`(같은 입력 재생성 = 대조군), `trim`(게이트웨이 Trimmer top_k 4)
- **평가**(`claude-opus-5-5`, 배치, 답마다 따로·어느 팔인지 가림): ① 정답성 yes/partial/no(원래 지문 기준), ② 충실성(R4 문장 S/H/A, H 하나라도 → 환각 응답)

## 사전 정의 기준 (SSOT §6 T1b, 실행 전 고정)

- **T1b-0(유효성)**: 세 팔 모두 생성·평가 성공한 질문 ≥ 180, 그리고 **대조군 안정성** `yes` 비율 |full2 − full| ≤ 0.05 (넘으면 판정 보류)
- **T1b-1(절감)**: LLM 입력 토큰 중앙값 감소(trim vs full) ≥ 40%
- **T1b-2(품질 주, 비열등)**: `yes` 비율 trim − full ≥ −0.05, 그리고 쌍체 부트스트랩(2,000회, 시드 2038) 95% 하한 ≥ −0.10
- **T1b-3(충실성)**: 환각 응답 비율 trim − full ≤ +0.03, 95% 상한 < +0.08
- **채택 = T1b-0~3** → `trim`을 비용 절감 권장 옵션으로 문서화(선택 기능 유지)
- temperature 0이 거부되면 형식 확인에서 빼고 기본값으로 진행, 그 사실을 기록(사전 고정). **API 상한 $12**

> 런타임: **CPU**. Colab 비밀 값 `ANTHROPIC_API_KEY`. 끊기면 다시 모두 실행(드라이브 `t1b/`에서 이어서). 끝나면 `t1b_results.zip`(원문 없음).
""")

md("## 1. 환경 준비")
code("SRC = " + repr(T1.SRC) + r"""
import os, sys, json, time, random, subprocess, shutil
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'anthropic', 'sentence-transformers', 'tiktoken', 'pysbd'], check=True)
subprocess.run('git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || true', shell=True)
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/hg/hallugate', exist_ok=True)
open('/content/hg/hallugate/__init__.py', 'w').write('')
for n, s in SRC.items(): open(f'/content/hg/hallugate/{n}.py', 'w').write(s)
sys.path.insert(0, '/content/hg')
OUT = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
T1B_DIR = os.path.join(OUT, 't1b'); os.makedirs(T1B_DIR, exist_ok=True)
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
""")

md("## 2. 질문 200개 (T1 질문 제외) — 긴 근거")
code(T1.T1_DATA + T1B_DATA + r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = [json.loads(l) for l in open(f'{BASE}/source_info.jsonl')]
POOL = train_qa_sources(responses, sources)
T1_SIDS = {x['sid'] for x in make_items(POOL)}             # T1에서 결과를 본 질문 (재현)
ITEMS = make_items_b(POOL, T1_SIDS)
print(f'train QA {len(POOL)}개, T1 제외 후 {len(ITEMS)}개, 긴 근거 평균 {np.mean([len(x["ref_L"]) for x in ITEMS]):.0f}자')
""")

md("## 3. 다이어트 (게이트웨이 Trimmer)")
code(T1.T1_TRIM + r"""
import tiktoken
from sentence_transformers import SentenceTransformer
ENC = tiktoken.get_encoding('cl100k_base')
ntok = lambda t: len(ENC.encode(t, disallowed_special=()))
T = trim_all(ITEMS, Trimmer(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'), 120, 20), ntok)
d = T[T.cond == 'L']
print('근거 토큰 감소 중앙값(k4)', round(float((1 - d.ref_tok_k4 / d.ref_tok_full).median()), 3), '| 원래 지문 재현(k4)', round(float(d.own_recall_k4.mean()), 3))
""")

md("## 4. 생성 (Haiku, temperature 0) — 형식 확인 → 비용 추정 → 배치")
code("import pysbd\nSEG = pysbd.Segmenter(language='en', clean=False, char_span=True)\n" + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + "\n" + T1.T1_API + T1B_GEN + r"""
USE_TEMP = True
_it = ITEMS[0]
try:                                                   # temperature 0 형식 확인 (거부되면 빼고 진행, 사전 고정)
    _r = rec(A.messages.create(**gen_params_b(_it['q'], _it['trim_L'], True)))
    print('형식 확인: temperature 0 사용 | 입력', _r['in'], '출력', _r['out'])
except Exception as e:
    if 'temperature' in str(e).lower():
        USE_TEMP = False; print('temperature 거부 → 기본값으로 진행:', str(e)[:200])
    else:
        raise
BUDGET_USD = 12.0
REQ_G = gen_requests_b(ITEMS, USE_TEMP)
def est(reqs, out_tok):
    tot = 0.0
    for p in reqs.values():
        n_in = (len(p.get('system', '')) + sum(len(m['content']) for m in p['messages'])) / 3.5
        pin, pout = PRICE[p['model']]; tot += n_in / 1e6 * pin + out_tok / 1e6 * pout
    return tot
EST_G = est(REQ_G, 400)
EST_J = len(ITEMS) * 3 * 2 * (2200 * PRICE[LABELER][0] + 500 * PRICE[LABELER][1]) / 1e6
print(f'생성 {len(REQ_G)}건 예상 ${EST_G:.2f} + 평가 약 ${EST_J:.2f} (상한 ${BUDGET_USD})')
assert EST_G + EST_J <= BUDGET_USD, '예상 비용이 상한을 넘음 — 멈춤'
G = run_batch(REQ_G, os.path.join(T1B_DIR, 'gen.json'))
print('생성 상태', pd.Series([gen_status(G.get(k)) for k in REQ_G]).value_counts().to_dict(),
      '| 잘림', sum(1 for k in REQ_G if G.get(k, {}).get('stop') == 'max_tokens'))
""")

md("## 5. 평가 (Opus, 답마다 따로): 정답성 + 충실성")
code(r"""
REQ_Q, REQ_F = judge_requests_b(ITEMS, G)
QR = run_batch(REQ_Q, os.path.join(T1B_DIR, 'grade.json'))
FR = run_batch(REQ_F, os.path.join(T1B_DIR, 'faith.json'))
print(f'정답성 {len(QR)}/{len(REQ_Q)}, 충실성 {len(FR)}/{len(REQ_F)}')
""")

md("## 6. 기준 판정")
code(T1.T1_METRICS + T1B_METRICS + r"""
D = build_rows_b(ITEMS, G, QR, FR, T)
R = judge_t1b(D)
print(f"유효 {R['n']} | yes: full {R['yes']['full']:.3f}, full2 {R['yes']['full2']:.3f}, trim {R['yes']['trim']:.3f}")
print(f"대조군 yes 차이 {R['control_yes_diff']:+.3f} [{R['control_yes_ci'][0]:+.3f}, {R['control_yes_ci'][1]:+.3f}] | 환각 차이 {R['control_hal_diff']:+.3f}")
print(f"\n=== T1b-0 유효성·대조군 안정(|차이| ≤ 0.05) → {'충족' if R['T1b_0'] else '미달(판정 보류)'} ===")
print(f"=== T1b-1 절감: 입력 토큰 감소 중앙값 {R['input_reduction_median']:.3f} (≥ 0.40) → {'충족' if R['T1b_1'] else '미달'} ===")
print(f"=== T1b-2 품질(비열등): yes trim − full {R['yes_diff']:+.3f} [{R['yes_diff_ci'][0]:+.3f}, {R['yes_diff_ci'][1]:+.3f}] (≥ −0.05, 하한 ≥ −0.10) → {'충족' if R['T1b_2'] else '미달'} ===")
print(f"=== T1b-3 충실성: 환각 trim − full {R['hal_diff']:+.3f} [{R['hal_diff_ci'][0]:+.3f}, {R['hal_diff_ci'][1]:+.3f}] (≤ +0.03, 상한 < +0.08) → {'충족' if R['T1b_3'] else '미달'} ===")
print(f"=== T1b {'채택' if R['T1b'] else '미채택'} ===")
print('(보고) partial', R['partial'], '| no', R['no'], '| 잘림', R['trunc'], '| 환각', R['hal'])
""")

md("## 7. 저장")
code(r"""
usage = {f: round(cost(load_json(os.path.join(T1B_DIR, f))), 2) for f in ['gen.json', 'grade.json', 'faith.json'] if os.path.exists(os.path.join(T1B_DIR, f))}
print('API 사용량(배치 단가) $', usage, '합계', round(sum(usage.values()), 2))
LOCAL = '/content/t1b_local'; os.makedirs(LOCAL, exist_ok=True)
D.to_csv(os.path.join(LOCAL, 't1b_rows.csv'), index=False)           # 원문 없음
json.dump(dict(R, cost_usd=usage, use_temperature=USE_TEMP, n_pool=len(POOL)), open(os.path.join(LOCAL, 't1b_summary.json'), 'w'),
          ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL): shutil.copy(os.path.join(LOCAL, f), os.path.join(T1B_DIR, f))
zp = shutil.make_archive('/content/t1b_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 't1b_trim_quality.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
