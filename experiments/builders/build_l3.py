"""L3 노트북 생성: I3-4 지연 측정 — F1 base + I2 두 판정기를 함께 돌린 판정 1회(점수 2개 + 조합)의 p95.
게이트웨이 판정기 코드(hallugate/judge.py)를 그대로 품고, 드라이브의 두 판정기 폴더를 불러 잰다. 학습 없음. 사전 기준은 SSOT §6 L3·I3."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
JUDGE_SRC = open(os.path.join(ROOT, 'hallugate', 'judge.py'), encoding='utf-8').read()
I3 = json.load(open(os.path.join(ROOT, 'docs', 'analysis', 'i3', 'i3_summary.json')))
STD = I3['standardize']
COMBO = {'F_mean': STD['Fb']['mean'], 'F_std': STD['Fb']['std'], 'I_mean': STD['I2']['mean'], 'I_std': STD['I2']['std'],
         'threshold': I3['threshold_svc10_8']}

# ---- L3_INPUTS·L3_CORE는 smoke_l3.py가 맥 CPU에서 실제 가중치로 그대로 실행해 확인한다 ----
L3_INPUTS = r"""
# 판정 입력: RAGTruth train 응답의 문장 끝 + 8토큰 지점 (게이트웨이와 같은 cl100k 토큰, 문장 시작부터)
import pysbd, tiktoken
ENC = tiktoken.get_encoding('cl100k_base')
SEG = pysbd.Segmenter(language='en', clean=False, char_span=True)

def build_reference(src):
    si, tt = src['source_info'], src['task_type']
    if tt == 'QA':
        return f"Question: {si.get('question','')}\n{si.get('passages','')}"
    if tt == 'Summary':
        return si if isinstance(si, str) else json.dumps(si, ensure_ascii=False)
    return '\n'.join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}"
                     for k, v in (si.items() if isinstance(si, dict) else []))

def points_of(text, s=8):
    # (문장 시작 글자, 지점 글자 끝) 목록: 문장마다 s토큰째 + 문장 끝
    ids = ENC.encode(text, disallowed_special=())
    _, starts = ENC.decode_with_offsets(ids)
    ends = starts[1:] + [len(text)]
    out = []
    for sp in SEG.segment(text):
        if not sp.sent.strip(): continue
        a = sp.start + (len(sp.sent) - len(sp.sent.lstrip()))
        tok = [e for st, e in zip(starts, ends) if a <= st < sp.end or (st < a < e)]
        for k, e in enumerate(tok, 1):
            if k % s == 0 or k == len(tok):
                out.append((a, min(e, sp.end)))
    return out

def make_inputs(responses, sources, n_resp=300, n_pts=300, seed=2027):
    rs = random.Random(seed)
    pool = [r for r in responses if r['split'] == 'train']
    cands = []
    for r in rs.sample(pool, min(n_resp, len(pool))):
        ref = build_reference(sources[r['source_id']])
        for a, cut in points_of(r['response']):
            text = r['response'][a:cut].strip()
            if text: cands.append({'ref': ref, 'text': text, 'prior': r['response'][:a], 'task': sources[r['source_id']]['task_type']})
    return rs.sample(cands, min(n_pts, len(cands)))
"""

L3_CORE = r"""
import contextlib, threading
from concurrent.futures import ThreadPoolExecutor

def make_judge(d, device):
    # load_judge와 같은 값으로 (판정기 폴더의 judge_config.json)
    cfg = read_config(d)
    return StudentJudge(d, embedder=cfg.get('embedder', 'sentence-transformers/all-MiniLM-L6-v2'), k=cfg.get('k', 8),
                        max_len=cfg.get('max_len', 512), chunk_words=cfg.get('chunk_words', 40), device=device,
                        mode=input_mode(cfg))

def combine(sf, si):
    # I3 조합: 두 점수를 cal-L 멀쩡한 응답 기준으로 표준화해 평균
    return ((sf - COMBO['F_mean']) / COMBO['F_std'] + (si - COMBO['I_mean']) / COMBO['I_std']) / 2

def judge_seq(jF, jI, refF, refI, x):
    # 차례로: 게이트웨이 judge() 두 번 (각각 토크나이즈·검색·순전파·동기화 포함)
    return jF.judge(refF, x['text'], x['prior']).score, jI.judge(refI, x['text'], x['prior']).score

STREAMS = {}
def _score_on_stream(j, ref, text, prior):
    # judge()와 같은 계산을 판정기 전용 CUDA 스트림에서. 동기화는 그 스트림만 (.cpu()가 현재 스트림을 기다림)
    ctx = torch.cuda.stream(STREAMS.setdefault(id(j), torch.cuda.Stream())) if j.device == 'cuda' else contextlib.nullcontext()
    with ctx, torch.no_grad():
        a, b = j._pair(ref, text, prior)
        enc = j.tok([a], [b], truncation='only_first', max_length=j.max_len, return_tensors='pt').to(j.device)
        with torch.autocast(j.device, enabled=(j.device == 'cuda')):
            return float(j.model(**enc).logits.squeeze().float().cpu())

POOL = ThreadPoolExecutor(2)
def judge_par(jF, jI, refF, refI, x):
    fF = POOL.submit(_score_on_stream, jF, refF, x['text'], x['prior'])
    fI = POOL.submit(_score_on_stream, jI, refI, x['text'], x['prior'])
    return fF.result(), fI.result()

def bench(fn, jF, jI, inputs, refs, warm=10):
    for x in inputs[:warm]: fn(jF, jI, *refs[x['ref']], x)
    ms, sc = [], []
    for x in inputs:
        t0 = time.perf_counter()
        sf, si = fn(jF, jI, *refs[x['ref']], x)
        c = combine(sf, si)
        ms.append((time.perf_counter() - t0) * 1000); sc.append((sf, si, c))
    return np.array(ms), np.array(sc)

def bench_one(j, inputs, refs, which, warm=10):
    for x in inputs[:warm]: j.judge(refs[x['ref']][which], x['text'], x['prior'])
    return np.array([j.judge(refs[x['ref']][which], x['text'], x['prior']).judge_ms for x in inputs])
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# L3 — I3-4 지연 측정: F1 base + I2 함께 돌린 판정 1회

## 왜 하나

I3(F1 base + I2 점수 평균)는 요약·전체 탐지율 기준을 모두 넘어 **조건부 채택**됐다. 남은 조건은 **I3-4: 두 판정기를 함께 돌린 판정 1회 p95 ≤ 68ms(A100)**.
판정기 하나가 각각 약 40ms(p95, A100)라 차례로 돌리면 예산(8토큰 × 8.5ms = 68ms)을 넘을 가능성이 크다. GPU에서 두 판정기를 **동시에** 돌리면 들어오는지 잰다.

## 무엇을 하나

- 판정기 코드는 게이트웨이의 `hallugate/judge.py` 그대로, 가중치는 드라이브 `final/judge_base`(F1 base)와 `i2/judge_i2`(I2)
- 입력: RAGTruth train 응답 300개의 문장 끝 + 8토큰 지점 중 300개 (F1 base는 문장 앞부분 + 근거 검색, I2는 근거 전체 + 응답 처음부터 그 지점까지)
- 판정 1회 = 두 점수 + 조합. 근거 준비(요청당 1회)는 제외
- 방식: **차례로**(게이트웨이 `judge()` 두 번) / **동시**(스레드 2개, 판정기마다 CUDA 스트림)

## 사전 정의 기준 (결과 보기 전, SSOT §6 L3에 고정)

- 두 방식 중 p95가 작은 방식으로 판정. **I3-4 충족 = A100에서 p95 ≤ 68ms**
- 동시 방식의 점수는 차례로 방식과 같아야 한다(최대 차이 ≤ 0.05)
- A100이 아니면 판정 보류(보고만). 판정기 단독 p50·p95도 보고

> **런타임: A100**. 약 10분. 결과는 드라이브 `Colab Notebooks/HalluGate/l3/`와 `l3_results.zip`.
""")

md("## 1. 환경 준비")
code("JUDGE_SRC = " + repr(JUDGE_SRC) + r"""
import os, sys, json, time, random, subprocess, shutil
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'sentence-transformers', 'tiktoken', 'pysbd', '-U', 'transformers>=4.48'], check=True)
subprocess.run('git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || true', shell=True)
import numpy as np, torch
from google.colab import drive
drive.mount('/content/drive')
open('hallugate_judge.py', 'w').write(JUDGE_SRC); sys.path.insert(0, '.')
from hallugate_judge import StudentJudge, read_config, input_mode
OUT = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
F_DIR, I_DIR, L3_DIR = os.path.join(OUT, 'final', 'judge_base'), os.path.join(OUT, 'i2', 'judge_i2'), os.path.join(OUT, 'l3')
os.makedirs(L3_DIR, exist_ok=True)
for d in [F_DIR, I_DIR]:
    assert os.path.exists(os.path.join(d, 'judge_config.json')), f'{d}/judge_config.json 없음'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'
COMBO = """ + repr(COMBO) + r"""
print('device', GPU, '| 조합 상수(I3)', COMBO)
""")

md("## 2. 판정 입력 300개")
code(L3_INPUTS + r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
for r in responses: r['task_type'] = sources[r['source_id']]['task_type']
INPUTS = make_inputs(responses, sources)
print(f"입력 {len(INPUTS)}개 | 과제", {t: sum(x['task'] == t for x in INPUTS) for t in ['QA', 'Summary', 'Data2txt']},
      f"| 응답 앞부분 길이(글자) 중앙값 {np.median([len(x['prior']) + len(x['text']) for x in INPUTS]):.0f}")
""")

md("## 3. 측정")
code(L3_CORE + r"""
jF, jI = make_judge(F_DIR, DEVICE), make_judge(I_DIR, DEVICE)
assert jF.mode == 'retrieval' and jI.mode == 'full', (jF.mode, jI.mode)
refs = {}
for x in INPUTS:
    if x['ref'] not in refs: refs[x['ref']] = (jF.prepare(x['ref']), jI.prepare(x['ref']))
prep_ms = [refs[r][0].prepare_ms for r in refs]

LAT = {}
for name, j, w in [('F1 base 단독', jF, 0), ('I2 단독', jI, 1)]:
    ms = bench_one(j, INPUTS, refs, w)
    LAT[name] = {'p50': float(np.percentile(ms, 50)), 'p95': float(np.percentile(ms, 95))}
ms_seq, sc_seq = bench(judge_seq, jF, jI, INPUTS, refs)
ms_par, sc_par = bench(judge_par, jF, jI, INPUTS, refs)
LAT['차례로'] = {'p50': float(np.percentile(ms_seq, 50)), 'p95': float(np.percentile(ms_seq, 95))}
LAT['동시'] = {'p50': float(np.percentile(ms_par, 50)), 'p95': float(np.percentile(ms_par, 95))}
PAR_DIFF = float(np.abs(sc_par[:, :2] - sc_seq[:, :2]).max())
for k, v in LAT.items(): print(f"{k:10} p50 {v['p50']:6.1f} / p95 {v['p95']:6.1f} ms")
print(f'동시 방식 점수 차이 최대 {PAR_DIFF:.4f} | F1 base 근거 준비 중앙값 {np.median(prep_ms):.1f}ms(요청당 1회, 판정에서 제외)')
""")

md("## 4. 사전 기준 판정과 저장")
code(r"""
par_ok = PAR_DIFF <= 0.05
cands = {'차례로': LAT['차례로']['p95']}
if par_ok: cands['동시'] = LAT['동시']['p95']
BEST = min(cands, key=cands.get); P95 = cands[BEST]
IS_A100 = 'A100' in GPU
I3_4 = bool(IS_A100 and P95 <= 68)
print(f"=== 동시 방식 점수 일치: {'충족' if par_ok else '미달 → 동시 방식 제외'} (최대 차이 {PAR_DIFF:.4f}) ===")
print(f"=== I3-4: {BEST} p95 {P95:.1f}ms @ {GPU} → {'충족' if I3_4 else ('판정 보류 (A100 아님)' if not IS_A100 else '미달')} ===")
summ = {'gpu': GPU, 'latency': LAT, 'par_max_diff': PAR_DIFF, 'best': BEST, 'p95': P95, 'I3_4': I3_4, 'is_a100': IS_A100,
        'n_inputs': len(INPUTS), 'prepare_ms_median': float(np.median(prep_ms)), 'combo': COMBO}
LOCAL = '/content/l3_local'; os.makedirs(LOCAL, exist_ok=True)
json.dump(summ, open(os.path.join(LOCAL, 'l3_summary.json'), 'w'), ensure_ascii=False, indent=1)
np.savetxt(os.path.join(LOCAL, 'l3_ms.csv'), np.c_[ms_seq, ms_par], delimiter=',', header='seq_ms,par_ms', comments='')
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(L3_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/l3_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'l3_combo_latency.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
