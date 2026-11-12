"""X2 노트북 생성: FaithBench(사람 정답, 최신 LLM 10종 요약)로 배포 판정기를 외부 평가한다(보고용, 판정기·임계값 결정에 쓰지 않음).
Claude 정답에 기대지 않는 평가라 "Claude가 정답을 매기고 Claude 응답으로 평가해 부풀었다"는 비판을 확인하는 용도.
판정 지점·조합은 L3·M1과 같은 게이트웨이 코드. K2 판정기(k2/judge_base)가 있으면 새 조합도 함께 보고. 사전 해석 규칙은 SSOT §6 X2.
FaithBench 데이터(CC BY-NC-SA 4.0)는 노트북 안에서 받아 쓰고 저장소에 넣지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
_spec = importlib.util.spec_from_file_location('build_m1', os.path.join(HERE, 'build_m1.py'))
M1 = importlib.util.module_from_spec(_spec); _argv = sys.argv; _spec.loader.exec_module(M1); sys.argv = _argv
L3 = M1.L3

def _src(c): return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
def sub(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:70])
    return s.replace(a, b)

ENV = next(_src(c) for c in M1.cells if _src(c).startswith('SRC = '))
ENV = sub(ENV, "'anthropic', ", "")
ENV = sub(ENV, "R0_DIR, M1_DIR = os.path.join(OUT, 'r0'), os.path.join(OUT, 'm1')\nos.makedirs(M1_DIR, exist_ok=True)",
          "X2_DIR, K2_DIR = os.path.join(OUT, 'x2'), os.path.join(OUT, 'k2')\nos.makedirs(X2_DIR, exist_ok=True)")
ENV = sub(ENV, "assert os.path.exists(os.path.join(R0_DIR, 'r0_items.pkl')), 'r0/r0_items.pkl 없음 (cal-L·dev-L 분할)'\n", "")
ENV = sub(ENV, "subprocess.run('git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || true', shell=True)",
          "subprocess.run('git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || true', shell=True)\n"
          "subprocess.run('git clone --depth 1 https://github.com/vectara/FaithBench.git 2>/dev/null || true', shell=True)")

# ---- X2_* 셀은 smoke_x2.py가 확인한다 ----
X2_DATA = r"""
import glob, re
FB = []
for f in sorted(glob.glob('FaithBench/data_for_release/batch_*.json'), key=lambda p: int(re.findall(r'(\d+)', p)[-1])):
    FB += json.load(open(f))['samples']
SEV = {'Benign': 1, 'Questionable': 2}
def sev(l): return 3 if l.startswith('Unwanted') else SEV.get(l, 0)

def fb_label(s):
    # 최악 등급(FaithBench binarize.py의 worst): 3 = Unwanted가 하나라도, 2 = Questionable, 1 = Benign만, 0 = 표시 없음
    ls = {l for a in s['annotations'] for l in a['label']}
    return max([sev(l) for l in ls], default=0)

def fb_onset(s):
    # 첫 Unwanted 구간의 요약 글자 위치 (없으면 None)
    st = [a['summary_start'] for a in s['annotations'] if any(l.startswith('Unwanted') for l in a['label']) and a.get('summary_start') is not None]
    return min(st) if st else None

def ngrams(t, n=8):
    w = re.findall(r'\w+', t.lower())
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}

def overlap_flags(fb_sources, rt_sources, thr=0.5):
    # FaithBench 근거가 RAGTruth train 요약 근거와 겹치는지: 8-gram 포함률 ≥ 0.5
    rt = [ngrams(t) for t in rt_sources]
    out = {}
    for s in fb_sources:
        g = ngrams(s)
        out[s] = max([len(g & r) / max(len(g), 1) for r in rt], default=0.0) >= thr if g else False
    return out
"""

X2_METRICS = r"""
from sklearn.metrics import roc_auc_score

def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all(): return float('nan')
    return float(roc_auc_score(y, s))

def x2_rows(FB, SC, thr, overlap):
    # 요약 하나 = 행. hal = Unwanted 있음, clean = 표시 없음 또는 Benign만, Questionable이 최악이면 주 분석에서 뺌
    rows = []
    sid = {t: k for k, t in enumerate(sorted({x['source'] for x in FB}))}
    for i, s in enumerate(FB):
        pts = SC.get(str(i), [])
        lab = fb_label(s); on = fb_onset(s)
        r = {'i': i, 'src': sid[s['source']], 'summarizer': s['metadata']['summarizer'], 'worst': lab,
             'hal': lab == 3, 'primary': lab != 2 and not overlap.get(s['source'], False), 'overlap': overlap.get(s['source'], False),
             'n_pts': len(pts), 'onset': on,
             'hhem21': 1 - s['metadata']['hhem-2.1'], 'hhem21_en': 1 - s['metadata']['hhem-2.1-english']}
        for b in ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt_4o', 'trueteacher', 'true_nli']:
            v = s['metadata'].get(b); r[f'pred_{b}'] = None if v is None else int(v == 0)      # 1 = 환각이라고 예측
        for name, col in [('c', 4), ('f', 2), ('i', 3), ('n', 5)]:
            vals = [p[col] for p in pts if len(p) > col and p[col] is not None]
            r[f'max_{name}'] = max(vals) if vals else np.nan
        for name, col in [('c', 4), ('n', 5)]:
            t = thr.get(name)
            if t is None or np.isnan(r[f'max_{name}']): continue
            first = next((p[0] for p in pts if p[col] is not None and p[col] >= t), None)
            r[f'flag_{name}'] = first is not None
            h0 = max([p[0] for p in pts if p[0] <= on], default=0) if on is not None else None     # 첫 Unwanted 구간이 든 문장의 시작
            r[f'strict_{name}'] = bool(on is not None and first is not None and first >= h0)
        rows.append(r)
    return pd.DataFrame(rows)

def balanced_acc(y, pred):
    y, p = np.asarray(y, bool), np.asarray(pred, bool)
    return float((p[y].mean() + (~p[~y]).mean()) / 2)

def cluster_boot(df, fn, n=2000, seed=2034):
    # 같은 근거의 요약 10개가 묶여 있으므로 근거 단위로 다시 뽑는다
    srcs = df.src.unique(); rng = np.random.default_rng(seed)
    g = {s: df[df.src == s] for s in srcs}
    vals = []
    for _ in range(n):
        d = pd.concat([g[s] for s in rng.choice(srcs, len(srcs))])
        v = fn(d)
        if v == v: vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def x2_summary(df):
    P = df[df.primary]
    out = {'n_all': len(df), 'n_primary': len(P), 'n_hal': int(P.hal.sum()), 'n_clean': int((~P.hal).sum()),
           'n_overlap': int(df.overlap.sum()), 'n_questionable_worst': int((df.worst == 2).sum())}
    cols = [c for c in ['max_c', 'max_f', 'max_i', 'max_n', 'hhem21', 'hhem21_en'] if c in P and P[c].notna().all()]
    out['auroc'] = {c: auroc(P.hal, P[c]) for c in cols}
    out['auroc_ci'] = {c: cluster_boot(P, lambda d, c=c: auroc(d.hal, d[c])) for c in cols}
    out['auroc_diff_c_vs_hhem21'] = {'delta': out['auroc']['max_c'] - out['auroc']['hhem21'],
                                     'ci': cluster_boot(P, lambda d: auroc(d.hal, d.max_c) - auroc(d.hal, d.hhem21))}
    if 'max_n' in cols:
        out['auroc_diff_n_vs_c'] = {'delta': out['auroc']['max_n'] - out['auroc']['max_c'],
                                    'ci': cluster_boot(P, lambda d: auroc(d.hal, d.max_n) - auroc(d.hal, d.max_c))}
    out['bacc'] = {}
    for c in ['flag_c', 'flag_n'] + [k for k in P.columns if k.startswith('pred_')]:
        if c in P and P[c].notna().all(): out['bacc'][c] = balanced_acc(P.hal, P[c].astype(bool))
    for c in ['flag_c', 'flag_n']:
        if c in P and P[c].notna().all():
            out[f'{c}_hal_rate'] = float(P[P.hal][c].astype(bool).mean()); out[f'{c}_clean_rate'] = float(P[~P.hal][c].astype(bool).mean())
    for c in ['strict_c', 'strict_n']:
        if c in P and P[c].notna().all(): out[f'{c}_rate'] = float(P[P.hal][c].astype(bool).mean())
    out['by_summarizer_auroc_c'] = {m: auroc(d.hal, d.max_c) for m, d in P.groupby('summarizer')}
    out['by_summarizer_n_hal'] = {m: int(d.hal.sum()) for m, d in P.groupby('summarizer')}
    Q = df[~df.overlap]                                  # 보고: Questionable도 환각으로
    out['auroc_q_as_hal'] = {c: auroc(Q.worst >= 2, Q[c]) for c in cols}
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# X2 — FaithBench 외부 평가 (사람 정답, 최신 LLM 요약) · 보고용

## 왜 하나

M1·K1·K2는 **최신 LLM 응답을 Claude(Opus)가 매긴 정답**으로 평가한다. 생성도 Claude, 정답도 Claude라서 "Claude 기준을 배운 것 아니냐"는 의문이 남는다.
[FaithBench](https://github.com/vectara/FaithBench)(Vectara, NAACL 2025)는 **최신 LLM 10종**(GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Flash, Llama-3.1 등)이 쓴 요약 750개에 **사람 전문가가 환각 구간**을 표시한 데이터라,
Claude 정답에 기대지 않고 판정기를 잴 수 있다. 공개 판정기들이 거의 맞히지 못하도록 **어려운 사례를 골라 만든** 데이터라 절대 수치는 낮게 나오는 것이 정상이다.

## 무엇을 하나 (학습 없음, API 비용 없음)

- 근거 = FaithBench 원문, 응답 = 요약. 게이트웨이 `judge.py`로 문장마다 8토큰 + 문장 끝 지점을 판정(M1과 같은 코드)
- 기존 조합(F1 base + I2, 임계값 A 2.992). **K2 판정기(`k2/judge_base`)가 있으면 새 조합도**(임계값 A·B는 `k2/k2_summary.json`)
- 비교 대상: FaithBench가 함께 제공하는 공개 판정기 예측(HHEM-2.1 점수, GPT-4o 등 LLM 판정·TrueTeacher·TRUE의 0/1)
- RAGTruth 학습 요약 근거와 8-gram이 50% 이상 겹치는 FaithBench 근거는 주 분석에서 뺀다(학습 오염 차단)

## 사전 정의 (SSOT §6 X2, 실행 전 고정 — 보고용, 판정기·임계값 결정에 쓰지 않음)

- **정답**: 요약마다 최악 등급(FaithBench `binarize.py`의 worst). **환각 = Unwanted가 하나라도**, **멀쩡 = 표시 없음 또는 Benign만**, Questionable이 최악인 요약은 주 분석에서 뺌(보고에서는 환각으로 셈)
- **주 지표**: 응답 최대 조합 점수의 AUROC, 근거 단위 군집 부트스트랩 95% 구간(2,000회, 시드 2034)
- **해석 규칙**: 기존 조합 AUROC − HHEM-2.1 AUROC의 구간이 0보다 크면 "공개 판정기(HHEM-2.1)보다 앞섬", 0을 포함하면 "비슷", 0보다 작으면 "뒤짐". 기존 조합 AUROC 구간 하한이 0.5보다 크지 않으면 "FaithBench에서 판별력 확인 안 됨"
- 보고만: F1·I2 단독 AUROC, 임계값 A(·B)에서 균형 정확도·환각 탐지율·멀쩡한 응답 표시율, 첫 Unwanted 구간 이후 탐지, 생성 모델별 AUROC, Questionable 포함 AUROC, LLM 판정기들의 균형 정확도, 새 조합 − 기존 조합 AUROC 차이

> 런타임: **GPU**(아무거나). 판정 약 10~20분. 끊기면 다시 모두 실행(점수는 드라이브 `x2/`에서 이어서). 끝나면 `x2_results.zip`(원문 없음)이 내려받아진다.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. FaithBench와 RAGTruth 학습 근거 겹침")
code(L3.L3_INPUTS + X2_DATA + r"""
assert len(FB) == 750, len(FB)
responses = [json.loads(l) for l in open('RAGTruth/dataset/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open('RAGTruth/dataset/source_info.jsonl')}
_train_sids = {r['source_id'] for r in responses if r['split'] == 'train'}          # test 분할 근거는 보지 않는다
RT_SUM = [build_reference(sources[s]) for s in _train_sids if sources[s]['task_type'] == 'Summary']
OVERLAP = overlap_flags(sorted({s['source'] for s in FB}), RT_SUM)
print(f"FaithBench 요약 {len(FB)}개 | 근거 {len(OVERLAP)}개 중 RAGTruth 학습 요약 근거와 겹침 {sum(OVERLAP.values())}개")
import collections
print('최악 등급(0 표시 없음, 1 Benign, 2 Questionable, 3 Unwanted):', dict(sorted(collections.Counter(fb_label(s) for s in FB).items())))
""")
md("## 3. 판정 — 기존 조합(필수), K2 새 조합(있으면)")
code(L3.L3_CORE + M1.M1_SCORE + r"""
jF, jI = make_judge(F_DIR, DEVICE), make_judge(I_DIR, DEVICE)
FN = judge_par if DEVICE == 'cuda' else judge_seq
THR = {'c': COMBO['threshold']}
K2_JUDGE, K2_SUM = os.path.join(K2_DIR, 'judge_base'), os.path.join(K2_DIR, 'k2_summary.json')
HAS_K2 = os.path.exists(os.path.join(K2_JUDGE, 'judge_config.json')) and os.path.exists(K2_SUM)
if HAS_K2:
    K2S = load_json(K2_SUM); CN = K2S['combo_new']
    THR['n'] = K2S['thr']['new']['A']; THR['nB'] = K2S['thr']['new']['B']; THR['cB'] = K2S['thr']['old']['B']
    jK = make_judge(K2_JUDGE, DEVICE)
print('K2 판정기', '있음' if HAS_K2 else '없음(기존 조합만)', '| 임계값', THR)
sp = os.path.join(X2_DIR, 'scores_x2.json')
SC = load_json(sp) or {}
t0 = time.time()
for i, s in enumerate(FB):
    if str(i) in SC and (not HAS_K2 or all(len(p) == 6 for p in SC[str(i)])): continue
    src, summ = s['source'], s['summary']
    pts = [list(p) for p in score_response(jF, jI, (jF.prepare(src), jI.prepare(src)), summ, FN)]
    if HAS_K2:
        rk = jK.prepare(src)
        for p in pts:
            sk = jK.judge(rk, summ[p[0]:p[1]].strip(), summ[:p[0]]).score
            p.append(((sk - CN['F_mean']) / CN['F_std'] + (p[3] - CN['I_mean']) / CN['I_std']) / 2)
    SC[str(i)] = pts
    if (i + 1) % 50 == 0:
        save_json(SC, sp); print(f'  판정 {i + 1}/{len(FB)} ({time.time() - t0:.0f}s)', flush=True)
save_json(SC, sp)
""")
md("## 4. 지표와 해석 규칙")
code(X2_METRICS + r"""
DF = x2_rows(FB, SC, THR, OVERLAP)
if HAS_K2:                                     # 임계값 B 표시도 보고
    for name, col, key in [('cB', 4, 'cB'), ('nB', 5, 'nB')]:
        DF[f'flag_{name}'] = [any(p[col] >= THR[key] for p in SC[str(i)]) for i in DF.i]
X = x2_summary(DF)
if HAS_K2:
    P = DF[DF.primary]
    for c in ['flag_cB', 'flag_nB']:
        X['bacc'][c] = balanced_acc(P.hal, P[c]); X[f'{c}_hal_rate'] = float(P[P.hal][c].mean()); X[f'{c}_clean_rate'] = float(P[~P.hal][c].mean())
print(f"주 분석: 요약 {X['n_primary']}개 (환각 {X['n_hal']}, 멀쩡 {X['n_clean']}) | 제외: Questionable 최악 {X['n_questionable_worst']}, 겹침 근거 요약 {X['n_overlap']}")
for c, v in X['auroc'].items(): print(f"  AUROC {c:10} {v:.3f} [{X['auroc_ci'][c][0]:.3f}, {X['auroc_ci'][c][1]:.3f}]")
print('  균형 정확도', {k: round(v, 3) for k, v in X['bacc'].items()})
print(f"  임계값 A 기존 조합: 환각 탐지 {X.get('flag_c_hal_rate', float('nan')):.3f}, 멀쩡한 응답 표시 {X.get('flag_c_clean_rate', float('nan')):.3f}, 첫 Unwanted 이후 탐지 {X.get('strict_c_rate', float('nan')):.3f}")
print('  생성 모델별 기존 조합 AUROC', {m.split('/')[-1]: round(v, 3) for m, v in X['by_summarizer_auroc_c'].items()})
d = X['auroc_diff_c_vs_hhem21']; lo, hi = d['ci']
verdict = '앞섬' if lo > 0 else ('뒤짐' if hi < 0 else '비슷')
disc = X['auroc_ci']['max_c'][0] > 0.5
print(f"\n=== X2 해석: 기존 조합 − HHEM-2.1 AUROC {d['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] → 공개 판정기(HHEM-2.1)보다 {verdict} ===")
print(f"=== 기존 조합 AUROC 하한 {X['auroc_ci']['max_c'][0]:.3f} → {'판별력 있음(> 0.5)' if disc else 'FaithBench에서 판별력 확인 안 됨'} ===")
if HAS_K2:
    e = X['auroc_diff_n_vs_c']; print(f"(보고) K2 새 조합 − 기존 조합 AUROC {e['delta']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
X['verdict_vs_hhem21'], X['discriminates'], X['has_k2'], X['thr'] = verdict, bool(disc), HAS_K2, THR
""")
md("## 5. 저장")
code(r"""
LOCAL = '/content/x2_local'; os.makedirs(LOCAL, exist_ok=True)
DF.to_csv(os.path.join(LOCAL, 'x2_rows.csv'), index=False)        # 원문 없음
json.dump(dict(X, gpu=GPU), open(os.path.join(LOCAL, 'x2_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(X2_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/x2_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'x2_faithbench_eval.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
