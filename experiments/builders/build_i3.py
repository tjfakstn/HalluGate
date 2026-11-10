"""I3 노트북 생성: 요약 약점 개선 — F1 base + I2 저장 점수 조합 (학습 없음, CPU 런타임).
I2 dev-L 과제별 결과가 F1 base와 반대 방향(Summary·QA는 I2, Data2txt는 F1 base)이라 두 점수를 표준화해 합친다.
데이터·평가 세트·평가 함수는 X1(=F1)과 같고, I2 평가 지점 번호는 I2 노트북과 같은 함수로 다시 만든다. 사전 기준은 SSOT §6 I3."""
import json, os, importlib.util, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    m = importlib.util.module_from_spec(spec); argv = sys.argv; spec.loader.exec_module(m); sys.argv = argv
    return m
X1, I2 = load('build_x1'), load('build_i2')
def sub(s, a, b, count=1):
    assert s.count(a) == count, (s.count(a), a[:70])
    return s.replace(a, b)

R4C = X1.R4C
ENV = X1.find(R4C, "!pip -q install pysbd")
ENV = sub(ENV, "!pip -q install pysbd sentence-transformers tiktoken", "!pip -q install pysbd tiktoken")
ENV = sub(ENV, "/content/drive/MyDrive/selfproject", "/content/drive/MyDrive/Colab Notebooks/HalluGate")
ENV = sub(ENV, "R1_DIR = os.path.join(OUT_DIR, 'r4c')",
          "R1_DIR = I3_DIR = os.path.join(OUT_DIR, 'i3')\nFINAL_DIR = os.path.join(OUT_DIR, 'final')\nI2_DIR = os.path.join(OUT_DIR, 'i2')")
ENV = sub(ENV, "print('device', DEVICE)",
          "for p in [os.path.join(FINAL_DIR, 'scores_base.npy'), os.path.join(I2_DIR, 'scores_i2.npy')]:\n"
          "    assert os.path.exists(p), f'{p} 없음 (F1 재실행·I2 저장 점수)'\n"
          "import shutil\n"
          "print('device', DEVICE, '(GPU 필요 없음)')")
# I2 평가 지점 번호를 만드는 함수만 (build_eval_points, remap) — I2 노트북과 같은 순서
I2_EVAL = I2.I2_POINTS.split('def train_prefixes(items):')[0] + 'def build_eval_points' + I2.I2_POINTS.split('def build_eval_points')[1].split('def encode(')[0]

I3_CORE = r"""
def to_E(items, name, MAP, n):
    # 기존 질의 번호 점수(F1 base)를 I2 평가 지점 번호로 옮긴다
    a = np.full(n, np.nan, dtype=np.float32)
    for x in items:
        for t, q in x['pts2']:
            if (x['id'], t) in MAP: a[MAP[(x['id'], t)]] = SC[name][q]
    return a

def clean_stats(items, scorer):
    # 표준화 기준: 멀쩡한 응답(cal-L 원래 정답)의 판정 지점 점수 평균·표준편차
    v = np.concatenate([[s for _, s in series(x, scorer)] for x in items if x['onset'] is None])
    return float(np.mean(v)), float(np.std(v))

def compare_e(cal, a, b, kind='8', task=None, n_boot=2000, seed=2027):
    # a(후보) − b(기준). 임계값은 cal 전체로 등각 서비스 10%(서비스와 같은 단일 임계값), task가 있으면 그 과제 환각 응답만 비교
    ta, tb = conformal_flag(cal, f'{a}|{kind}'), conformal_flag(cal, f'{b}|{kind}')
    sub = [x for x in cal if task is None or x['task_type'] == task]
    da, la = detect_vec(sub, f'{a}|{kind}', ta); db, lb = detect_vec(sub, f'{b}|{kind}', tb)
    rng = np.random.default_rng(seed); n = len(da)
    bt = [da[i].mean() - db[i].mean() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return {'task': task or 'all', 'recall_new': float(da.mean()), 'recall_base': float(db.mean()), 'delta': float(da.mean() - db.mean()),
            'ci_lo': float(np.percentile(bt, 2.5)), 'ci_hi': float(np.percentile(bt, 97.5)),
            'delay_new': float(np.nanmedian(la)) if np.isfinite(la).any() else np.nan,
            'delay_base': float(np.nanmedian(lb)) if np.isfinite(lb).any() else np.nan,
            'n_hall': n, 'thr_new': ta, 'thr_base': tb}

def report(cal, dev, name, kind='8'):
    # dev-L 보고: cal 임계값으로 과제별 탐지율·지연과 멀쩡한 응답 표시율
    thr = conformal_flag(cal, f'{name}|{kind}'); out = []
    for task in [None] + TASKS:
        sub = [x for x in dev if task is None or x['task_type'] == task]
        det, dl = detect_vec(sub, f'{name}|{kind}', thr)
        flag = np.mean([any(s >= thr for _, s in series(x, f'{name}|{kind}')) for x in sub if x['onset'] is None])
        out.append({'model': name, 'task': task or 'all', 'recall': float(det.mean()),
                    'delay_med': float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan, 'clean_flagged': float(flag)})
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# I3 — 요약 약점 개선: F1 base + I2 점수 조합 (학습 없음)

## 왜 하나

배포 판정기 F1 base는 **요약 과제에서 가장 약하다**(dev-L 8토큰 서비스 탐지율 0.156, X1 응답 AUROC 0.752 — HHEM보다도 낮음).
근거 전체를 읽는 I2는 요약을 0.156 → 0.266, QA를 0.408 → 0.492로 올렸지만 Data2txt가 0.662 → 0.580으로 내려가 전체로는 같았다(SSOT §5 I2, dev-L 보고값).
**두 판정기가 잘하는 과제가 다르므로 점수를 합치면** 요약은 I2 덕에 오르고 Data2txt는 F1 base 덕에 지킬 수 있다. 저장된 점수만 쓰므로 학습이 필요 없다.

## 무엇을 하나

1. F1 base 저장 점수(`final/scores_base.npy`)와 I2 저장 점수(`i2/scores_i2.npy`)를 같은 판정 지점 번호로 맞춘다(I2 노트북과 같은 함수).
2. 두 점수를 **cal-L 멀쩡한 응답의 8토큰 지점 점수** 평균·표준편차로 표준화한다(logit 크기가 판정기마다 달라서).
3. 조합 두 가지: **(가) 평균**, **(나) 최댓값**. cal-L 8토큰 원래 정답 전체 탐지율이 높은 쪽을 고르고, 차이가 0.01 미만이면 평균.
4. 임계값은 cal-L 등각 서비스 10% (과제 구분 없이 하나 — 서비스에서는 과제를 모른다).

## 사전 정의 기준 (결과 보기 전, SSOT §6 I3에 고정)

- **I3-1 (주, 약점)**: cal-L 8토큰 원래 정답 **Summary** 탐지율 − F1 base ≥ +0.05, 그리고 응답 단위 쌍체 부트스트랩 95% 하한 > 0
- **I3-2 (전체)**: cal-L 8토큰 원래 정답 전체 탐지율 차이 ≥ 0.00
- **I3-3 (안전)**: C1 전체 차이 ≥ −0.01, Data2txt 차이 ≥ −0.03, 원래 정답 지연 중앙값 증가 ≤ 2토큰
- **I3-4 (지연)**: 두 판정기를 함께 돌린 판정 1회 p95 ≤ 68ms(A100) — 이 노트북에서는 재지 않는다. 측정 전까지 "조건부 채택"
- 채택 = I3-1·2·3 충족 + I3-4 측정 통과. dev-L은 보고만.

> **런타임: CPU로 충분**(약 10~20분, 대부분 데이터 준비). 드라이브에 `r0/`, `r4/`, `r4b/labels_train.json`, `final/scores_base.npy`, `i2/scores_i2.npy`. 결과는 드라이브 `Colab Notebooks/HalluGate/i3/`와 `i3_results.zip`.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터와 평가 세트 (F1·X1과 같음)")
code(X1.DATA)
code(X1.PREP)
code(X1.LOAD)
code(X1.CORR)
md("## 3. 평가 함수 (F1·I2와 같음)")
code(X1.SERIES)
code(X1.EVAL_DEFS)
code(X1.CONF)

md("""## 4. 두 판정기 점수를 같은 지점 번호로

I2 노트북과 같은 함수로 평가 지점 번호를 다시 만들고, I2 저장 점수 길이와 맞는지, F1 base·I2의 기존 cal-L 8토큰 결과가 재현되는지 확인한다.""")
code(I2_EVAL + I3_CORE + r"""
SC['F_base'] = np.load(os.path.join(FINAL_DIR, 'scores_base.npy'))
E_KEYS, E_TEXTS, E_NEED, E_MAP = build_eval_points(CAL + DEV, RESP_TEXT)
SC['I2'] = np.load(os.path.join(I2_DIR, 'scores_i2.npy'))
assert len(SC['I2']) == len(E_KEYS), f'I2 점수 길이 {len(SC["I2"])} ≠ 평가 지점 {len(E_KEYS)} — 지점 번호가 I2 노트북과 다름'
SC['Fb'] = to_E(CAL + DEV, 'F_base', E_MAP, len(E_KEYS))
LS = {lab: (remap(c, E_MAP), remap(d, E_MAP)) for lab, (c, d) in LAB_SETS.items()}
_e8 = [e for x in LS['orig'][0] + LS['orig'][1] for t, e in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % 8 == 0]
assert np.isfinite(SC['Fb'][_e8]).all() and np.isfinite(SC['I2'][_e8]).all(), '8토큰 지점에 점수가 없는 곳이 있음'

CALo, CALc = LS['orig'][0], LS['C1'][0]
_r = compare_e(CALo, 'I2', 'Fb')
print(f"재현 확인 cal-L 8토큰 원래 정답: F1 base {_r['recall_base']:.4f} (임계값 {_r['thr_base']:.4f}) | I2 {_r['recall_new']:.4f} (임계값 {_r['thr_new']:.4f})")
assert abs(_r['thr_base'] - 3.968751) < 1e-3 and abs(_r['recall_base'] - 0.5068) < 2e-3, 'F1 base 재현 실패'
assert abs(_r['thr_new'] - 4.562501) < 1e-3 and abs(_r['recall_new'] - 0.5068) < 2e-3, 'I2 재현 실패 (I2 노트북 10절과 다름)'
print(f'평가 지점 {len(E_KEYS):,}개, 8토큰 지점 {len(set(_e8)):,}개')
""")

md("""## 5. 표준화와 조합, 선택 (cal-L만)""")
code(r"""
STATS = {m: clean_stats(CALo, f'{m}|8') for m in ['Fb', 'I2']}
Z = {m: (SC[m] - mu) / sd for m, (mu, sd) in STATS.items()}
SC['MEAN'] = (Z['Fb'] + Z['I2']) / 2
SC['MAX'] = np.maximum(Z['Fb'], Z['I2'])
print('표준화 기준 (평균, 표준편차):', {m: tuple(round(v, 4) for v in s) for m, s in STATS.items()})

SEL_ROWS = [{'combo': c, **compare_e(CALo, c, 'Fb')} for c in ['MEAN', 'MAX']]
sel_tab = pd.DataFrame(SEL_ROWS)
print(sel_tab[['combo', 'recall_new', 'recall_base', 'delta', 'ci_lo', 'ci_hi', 'delay_new', 'delay_base']].round(3).to_string(index=False))
r_mean, r_max = SEL_ROWS[0]['recall_new'], SEL_ROWS[1]['recall_new']
CHOSEN = 'MAX' if r_max - r_mean >= 0.01 else 'MEAN'
print(f'=== 선택: {CHOSEN} (평균 {r_mean:.3f}, 최댓값 {r_max:.3f}) ===')
""")

md("""## 6. 사전 기준 판정 (cal-L)""")
code(r"""
ROWS = []
for lab, cal in [('orig', CALo), ('C1', CALc)]:
    for task in [None] + TASKS:
        for m in [CHOSEN, 'I2']:
            ROWS.append({'labels': lab, 'model': m, **compare_e(cal, m, 'Fb', task=task)})
cal_tab = pd.DataFrame(ROWS)
pd.set_option('display.width', 250)
print(cal_tab[['labels', 'model', 'task', 'recall_new', 'recall_base', 'delta', 'ci_lo', 'ci_hi', 'delay_new', 'delay_base', 'n_hall']].round(3).to_string(index=False))
def row(lab, task, m=None):
    return next(r for r in ROWS if r['labels'] == lab and r['task'] == (task or 'all') and r['model'] == (m or CHOSEN))
s, a, c, d = row('orig', 'Summary'), row('orig', None), row('C1', None), row('orig', 'Data2txt')
I3_1 = s['delta'] >= 0.05 and s['ci_lo'] > 0
I3_2 = a['delta'] >= 0.0
I3_3 = c['delta'] >= -0.01 and d['delta'] >= -0.03 and (a['delay_new'] - a['delay_base']) <= 2
print(f"=== I3-1 (주) Summary {s['recall_base']:.3f} → {s['recall_new']:.3f} (Δ {s['delta']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}], 환각 {s['n_hall']}) → {'충족' if I3_1 else '미달'} ===")
print(f"=== I3-2 (전체) {a['recall_base']:.3f} → {a['recall_new']:.3f} (Δ {a['delta']:+.3f} [{a['ci_lo']:+.3f}, {a['ci_hi']:+.3f}]) → {'충족' if I3_2 else '미달'} ===")
print(f"=== I3-3 (안전) C1 Δ {c['delta']:+.3f}, Data2txt Δ {d['delta']:+.3f}, 지연 {a['delay_base']:.0f} → {a['delay_new']:.0f} → {'충족' if I3_3 else '미달'} ===")
ADOPT_COND = bool(I3_1 and I3_2 and I3_3)
print(f"=== I3: {'조건부 채택 (I3-4 지연 측정 필요)' if ADOPT_COND else '미채택'} ===")
""")

md("""## 7. dev-L 확인 (보고만)""")
code(r"""
DEV_ROWS = []
for lab in ['orig', 'C1']:
    cal, dev = LS[lab]
    for m in ['Fb', 'I2', 'MEAN', 'MAX']:
        DEV_ROWS += [{'labels': lab, **r} for r in report(cal, dev, m)]
dev_tab = pd.DataFrame(DEV_ROWS)
print(dev_tab.round(3).to_string(index=False))
""")

md("## 8. 저장과 zip 다운로드")
code(r"""
LOCAL = '/content/i3_local'; os.makedirs(LOCAL, exist_ok=True)
sel_tab.to_csv(os.path.join(LOCAL, 'i3_select.csv'), index=False)
cal_tab.to_csv(os.path.join(LOCAL, 'i3_calL.csv'), index=False)
dev_tab.to_csv(os.path.join(LOCAL, 'i3_devL.csv'), index=False)
summ = {'chosen': CHOSEN, 'I3_1': bool(I3_1), 'I3_2': bool(I3_2), 'I3_3': bool(I3_3), 'adopt_conditional': ADOPT_COND,
        'standardize': {m: {'mean': mu, 'std': sd} for m, (mu, sd) in STATS.items()},
        'threshold_svc10_8': conformal_flag(CALo, f'{CHOSEN}|8'), 'summary': s, 'all': a, 'C1': c, 'Data2txt': d}
json.dump(summ, open(os.path.join(LOCAL, 'i3_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(I3_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/i3_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
print(json.dumps({k: summ[k] for k in ['chosen', 'I3_1', 'I3_2', 'I3_3', 'adopt_conditional']}, ensure_ascii=False))
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'i3_combine_judges.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
