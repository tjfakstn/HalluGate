"""T1c 노트북 생성: 근거 다이어트 top_k 8 (SSOT §6 T1c, 11/23 고정).
T1b와 같은 설계·기준(세 팔 full·full2·trim, 답마다 정답성·충실성 평가)에서 top_k만 4 → 8로 바꾸고,
T1·T1b 질문 400개를 뺀 새 표본(Random(2039))으로 잰다. CPU 런타임, RAGTruth train 분할만."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
_spec = importlib.util.spec_from_file_location('build_t1b', os.path.join(HERE, 'build_t1b.py'))
TB = importlib.util.module_from_spec(_spec); _argv = sys.argv; sys.argv = ['import']; _spec.loader.exec_module(TB); sys.argv = _argv
T1, M1 = TB.T1, TB.M1

# ---- 아래 T1C_* 문자열은 smoke_t1c.py가 가짜 데이터로 그대로 실행해 확인한다 ----
T1C_DATA = r"""
SEED_C, TOP_K_C = 2039, 8                            # 사전 고정 (SSOT §6 T1c)

def make_items_c(pool):
    # T1·T1b에서 결과를 본 질문은 빼고 새로 뽑는다 (make_items_b를 시드만 바꿔 재사용)
    global SEED_B
    t1 = {x['sid'] for x in make_items(pool)}
    t1b = {x['sid'] for x in make_items_b(pool, t1)}     # T1b 재현 (시드 2038)
    seen = t1 | t1b
    assert len(seen) == len(t1) + len(t1b), 'T1·T1b 질문이 겹침'
    old, SEED_B = SEED_B, SEED_C
    try:
        items = make_items_b(pool, seen)
    finally:
        SEED_B = old
    assert not {x['sid'] for x in items} & seen
    return items, seen
"""

T1C_METRICS = r"""
def judge_t1c(D):
    # T1b와 같은 기준 (SSOT §6 T1c). 키 이름만 T1c로
    R = judge_t1b(D)
    out = {k.replace('T1b', 'T1c'): v for k, v in R.items()}
    d = D[D.valid]
    out['own_recall_k8'] = float(d.own_recall_k8.mean())
    out['ref_token_reduction_k8_median'] = float((1 - d.ref_tok_k8 / d.ref_tok_full).median())
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# T1c — 근거 다이어트 top_k 8 (T1b와 같은 설계)

## 왜 하나

T1b(top_k 4)에서 근거 다이어트는 입력 토큰을 **−72.9%** 줄이고 답의 정답성도 비열등(−1.0%p)이었지만,
**근거 밖 내용이 섞이는 비율이 +3.7%p**로 충실성 기준(+3%p)을 넘어 미채택이었다. 원래 지문 일부가 잘린 질문에 환각이 몰린 정황(사후, 14개)이 있어,
**조각을 8개 남기면(절감은 줄지만) 충실성이 기준 안에 드는가**를 잰다.

## 무엇을 하나 (T1b와 같고, 바뀌는 것은 두 가지)

- **top_k 4 → 8**
- **표본**: RAGTruth train QA 근거 중 **T1·T1b의 400개를 뺀** 풀에서 `Random(2039)` 200개 (T1에서 top_k 8을 고를 때 본 수치와 겹치지 않게)
- 나머지는 T1b 그대로: 세 팔(`full`, `full2` 대조군, `trim`), Haiku 생성(max_tokens 1024, temperature 0 시도 → 거부되면 기본값), Opus 답마다 정답성(yes/partial/no)·충실성(R4)

## 사전 정의 기준 (SSOT §6 T1c, 실행 전 고정 — T1b와 같음)

- **T1c-0**: 유효 ≥ 180, 대조군 `yes` |full2 − full| ≤ 0.05 (넘으면 판정 보류)
- **T1c-1(절감)**: 입력 토큰 중앙값 감소 ≥ 40%
- **T1c-2(정답성, 비열등)**: `yes` trim − full ≥ −0.05, 하한 ≥ −0.10
- **T1c-3(충실성)**: 환각 응답 trim − full ≤ +0.03, 상한 < +0.08
- **채택 = T1c-0~3** → `trim`을 비용 절감 권장 옵션으로 문서화, **기본 top_k를 8로**. 미달이면 근거 다이어트는 실험 기능으로 두고 종료. **API 상한 $12**

> 런타임: **CPU**. Colab 비밀 값 `ANTHROPIC_API_KEY`. 끊기면 다시 모두 실행(드라이브 `t1c/`). 끝나면 `t1c_results.zip`(원문 없음).
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
T1C_DIR = os.path.join(OUT, 't1c'); os.makedirs(T1C_DIR, exist_ok=True)
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
""")

md("## 2. 질문 200개 (T1·T1b 질문 제외) — 긴 근거")
code(T1.T1_DATA + TB.T1B_DATA + T1C_DATA + r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = [json.loads(l) for l in open(f'{BASE}/source_info.jsonl')]
POOL = train_qa_sources(responses, sources)
ITEMS, SEEN = make_items_c(POOL)
print(f'train QA {len(POOL)}개, T1·T1b {len(SEEN)}개 제외 후 {len(ITEMS)}개')
""")

md("## 3. 다이어트 (게이트웨이 Trimmer, **top_k 8**)")
code(T1.T1_TRIM + r"""
TOP_K = TOP_K_C                                       # trim_all이 이 값의 조각으로 trim 팔을 만든다
import tiktoken
from sentence_transformers import SentenceTransformer
ENC = tiktoken.get_encoding('cl100k_base')
ntok = lambda t: len(ENC.encode(t, disallowed_special=()))
T = trim_all(ITEMS, Trimmer(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'), 120, 20), ntok)
d = T[T.cond == 'L']
print('근거 토큰 감소 중앙값(k8)', round(float((1 - d.ref_tok_k8 / d.ref_tok_full).median()), 3), '| 원래 지문 재현(k8)', round(float(d.own_recall_k8.mean()), 3))
""")

md("## 4. 생성 (Haiku) — 형식 확인 → 비용 추정 → 배치")
code("import pysbd\nSEG = pysbd.Segmenter(language='en', clean=False, char_span=True)\n" + M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + "\n" + T1.T1_API + TB.T1B_GEN + r"""
USE_TEMP = True
_it = ITEMS[0]
try:
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
EST_G = est(REQ_G, 500)
EST_J = len(ITEMS) * 3 * 2 * (2200 * PRICE[LABELER][0] + 500 * PRICE[LABELER][1]) / 1e6
print(f'생성 {len(REQ_G)}건 예상 ${EST_G:.2f} + 평가 약 ${EST_J:.2f} (상한 ${BUDGET_USD})')
assert EST_G + EST_J <= BUDGET_USD, '예상 비용이 상한을 넘음 — 멈춤'
G = run_batch(REQ_G, os.path.join(T1C_DIR, 'gen.json'))
print('생성 상태', pd.Series([gen_status(G.get(k)) for k in REQ_G]).value_counts().to_dict(),
      '| 잘림', sum(1 for k in REQ_G if G.get(k, {}).get('stop') == 'max_tokens'))
""")

md("## 5. 평가 (Opus, 답마다 따로): 정답성 + 충실성")
code(r"""
REQ_Q, REQ_F = judge_requests_b(ITEMS, G)
QR = run_batch(REQ_Q, os.path.join(T1C_DIR, 'grade.json'))
FR = run_batch(REQ_F, os.path.join(T1C_DIR, 'faith.json'))
print(f'정답성 {len(QR)}/{len(REQ_Q)}, 충실성 {len(FR)}/{len(REQ_F)}')
""")

md("## 6. 기준 판정")
code(T1.T1_METRICS + TB.T1B_METRICS + T1C_METRICS + r"""
D = build_rows_b(ITEMS, G, QR, FR, T)
R = judge_t1c(D)
print(f"유효 {R['n']} | yes: full {R['yes']['full']:.3f}, full2 {R['yes']['full2']:.3f}, trim {R['yes']['trim']:.3f}")
print(f"대조군 yes 차이 {R['control_yes_diff']:+.3f} [{R['control_yes_ci'][0]:+.3f}, {R['control_yes_ci'][1]:+.3f}] | 환각 차이 {R['control_hal_diff']:+.3f}")
print(f"\n=== T1c-0 유효성·대조군 안정(|차이| ≤ 0.05) → {'충족' if R['T1c_0'] else '미달(판정 보류)'} ===")
print(f"=== T1c-1 절감: 입력 토큰 감소 중앙값 {R['input_reduction_median']:.3f} (≥ 0.40) → {'충족' if R['T1c_1'] else '미달'} ===")
print(f"=== T1c-2 정답성(비열등): yes trim − full {R['yes_diff']:+.3f} [{R['yes_diff_ci'][0]:+.3f}, {R['yes_diff_ci'][1]:+.3f}] (≥ −0.05, 하한 ≥ −0.10) → {'충족' if R['T1c_2'] else '미달'} ===")
print(f"=== T1c-3 충실성: 환각 trim − full {R['hal_diff']:+.3f} [{R['hal_diff_ci'][0]:+.3f}, {R['hal_diff_ci'][1]:+.3f}] (≤ +0.03, 상한 < +0.08) → {'충족' if R['T1c_3'] else '미달'} ===")
print(f"=== T1c {'채택' if R['T1c'] else '미채택'} ===")
print('(보고) partial', R['partial'], '| 잘림', R['trunc'], '| 환각', R['hal'], '| 근거 재현 k8', round(R['own_recall_k8'], 3))
""")

md("## 7. 저장")
code(r"""
usage = {f: round(cost(load_json(os.path.join(T1C_DIR, f))), 2) for f in ['gen.json', 'grade.json', 'faith.json'] if os.path.exists(os.path.join(T1C_DIR, f))}
print('API 사용량(배치 단가) $', usage, '합계', round(sum(usage.values()), 2))
LOCAL = '/content/t1c_local'; os.makedirs(LOCAL, exist_ok=True)
D.to_csv(os.path.join(LOCAL, 't1c_rows.csv'), index=False)           # 원문 없음
json.dump(dict(R, cost_usd=usage, use_temperature=USE_TEMP, n_pool=len(POOL), top_k=TOP_K_C), open(os.path.join(LOCAL, 't1c_summary.json'), 'w'),
          ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL): shutil.copy(os.path.join(LOCAL, f), os.path.join(T1C_DIR, f))
zp = shutil.make_archive('/content/t1c_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 't1c_trim_k8.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
