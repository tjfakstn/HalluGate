"""E4-lite 노트북 생성: E2를 시드 3개 × 2에폭으로 다시 학습해 결과의 안정성을 확인한다.
데이터·근거 검색·학습 함수·평가 코드는 E2 노트북(그리고 그 안의 E1 코드)에서 그대로 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
e2 = json.load(open(os.path.join(NB_DIR, 'e2_prefix_trained_scorer.ipynb')))
C = [''.join(c['source']) for c in e2['cells']]
def find(prefix):
    return next(c for c in C if c.startswith(prefix))

ENV = find("!pip -q install pysbd").replace(
    "SAVE_DIR = os.path.join(OUT_DIR, 'student_prefix')",
    "SAVE_ROOT = os.path.join(OUT_DIR, 'student_prefix_seeds')\nE2_DIR = os.path.join(OUT_DIR, 'student_prefix')")
DATA = find("BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find("# 학습 응답: S2/S2-b의 gold-large")
PREP = find("ENC = tiktoken.get_encoding")
TRAIN_ITEMS = find("train_items = to_items(train)")
PREFIX = find("MIN_PREFIX, NEAR_AFTER, W_NEAR")
RETR = find("from sentence_transformers import SentenceTransformer").split("\nt0 = time.perf_counter()")[0]   # 학습 예시 검색은 시드마다 따로
POINTS = find("def points(item, s):")
TRAIN_DEFS = find("def val_auroc(tok, model):").split("\nt0 = time.perf_counter()")[0]
EVAL_DEFS = find("def calibrate(items, scorer, arl0):")
assert 'def train_prefix' in TRAIN_DEFS and 'def evaluate' in EVAL_DEFS

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# E4-lite — E2 결과는 시드를 바꿔도 유지되는가

## 왜 하나

E2(문장 앞부분 학습)는 사전 기준을 충족했다. 4토큰마다 재판정할 때 재현율 0.494, 탐지 지연 6.5토큰이었다.
하지만 **학습을 한 번(시드 42, 1에폭)만** 돌렸고, 학습 곡선도 끝까지 오르는 중이었다(검증 앞부분 AUROC 0.839 → 0.853).
결과가 우연인지, 그리고 더 학습하면 좋아지는지를 확인해야 중간보고에서 이 결과를 주장할 수 있다.

## 무엇을 하나

- 시드 **3개(42, 7, 2026)** × **2에폭**으로 다시 학습한다. 시드는 앞부분 자르는 위치, 학습 순서, 드롭아웃을 모두 바꾼다.
- 그 밖의 모든 설정(데이터, 정답 정의, 가중치, 근거 검색, 평가 코드)은 E2와 같다. E2 노트북에서 코드를 그대로 가져왔다.
- 체크포인트는 2에폭 동안 6번 검증 분할로 평가해 가장 좋은 것을 고른다.
- 비교용으로 E2 원본 모델(`student_prefix`, 시드 42·1에폭)과 기준선(`student_vark`)도 같은 질의로 다시 판정한다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **안정성(주 기준)**: 시드 3개 **모두** E2 주 기준을 충족한다. E2 주 기준은 ARL₀=100에서 s=4 또는 s=2 중 하나라도 재현율 ≥ 0.474이면서 지연 중앙값 ≤ 9토큰인 것이다.
- **평균**: s=4에서 시드 평균 재현율 ≥ 0.474, 시드 평균 지연 중앙값 ≤ 9토큰
- **보고**: 시드 간 표준편차와 2에폭 효과(두 번째 에폭에서 체크포인트가 선택된 시드 수)는 판정 없이 보고한다.
- 시드 3개 중 하나라도 미달하면 "E2 결과는 시드에 민감"으로 기록하고, 중간보고에서는 시드 평균만 주장한다.

> 런타임 **GPU(T4)**. 시드당 학습 약 20~25분 → 전체 약 1.5시간. 시드마다 모델을 드라이브 `selfproject/student_prefix_seeds/`에 저장하므로,
> 중간에 끊겨도 다시 실행하면 이미 끝난 시드는 건너뛴다.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터 (E2와 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
code(TRAIN_ITEMS)
md("## 3. 앞부분 학습 데이터 함수 (E2와 같음 — 시드마다 다시 만든다)")
code(PREFIX)
md("## 4. 근거 검색과 판정 함수 (E2와 같음)")
code(RETR)
md("## 5. 판정 지점과 검증 정답 (E2와 같음)")
code(POINTS)
md("## 6. 학습 함수 (E2와 같음)")
code(TRAIN_DEFS)

md("## 7. 시드별 학습 — 끝난 시드는 건너뜀")
code(r"""
SEEDS, EPOCHS, N_EVALS = [42, 7, 2026], 2, 6
os.makedirs(SAVE_ROOT, exist_ok=True)
train_log = []
for seed in SEEDS:
    out = os.path.join(SAVE_ROOT, f'seed{seed}_ep{EPOCHS}')
    if os.path.exists(os.path.join(out, 'config.json')):
        print(f'[시드 {seed}] 이미 학습됨 → 건너뜀'); continue
    SEED = seed                                     # train_prefix가 전역 SEED로 난수를 고정한다
    prng = random.Random(seed)
    train_ex = pd.DataFrame([e for x in train_items for e in prefix_examples(x, prng)])
    train_ex['ctx'] = contexts(train_ex.ref_key.tolist(), train_ex.text.tolist())
    print(f'[시드 {seed}] 앞부분 예시 {len(train_ex):,}개, 양성 {train_ex.y.mean():.1%}')
    t0 = time.perf_counter()
    tok_s, model_s, best_va, hist = train_prefix(STUDENT_DIR, epochs=EPOCHS, n_evals=N_EVALS)
    best_step = max(hist, key=lambda h: h[1])[0]
    n_step = hist[-1][0]
    tok_s.save_pretrained(out); model_s.save_pretrained(out)
    pd.DataFrame(hist, columns=['step', 'val_prefix_auroc']).to_csv(os.path.join(out, 'hist.csv'), index=False)
    print(f'[시드 {seed}] {(time.perf_counter()-t0)/60:.1f}분 | 최고 검증 앞부분 AUROC {best_va:.3f} '
          f'(step {best_step}/{n_step}, {"2" if best_step > n_step // 2 else "1"}번째 에폭)')
    del model_s; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
""")

md("## 8. 판정 — 기준선, E2 원본, 시드별 모델을 같은 질의에 적용")
code(r"""
SC = {}
def run(name, path):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    SC[name] = score(tok, m, Q_KEYS, Q_TEXTS)
    del m; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

t0 = time.perf_counter()
run('B', STUDENT_DIR)
if os.path.isdir(E2_DIR): run('E2', E2_DIR)
for seed in SEEDS:
    run(f'S{seed}', os.path.join(SAVE_ROOT, f'seed{seed}_ep{EPOCHS}'))
print(f'판정 {len(queries):,}개 × {len(SC)} 모델, {time.perf_counter()-t0:.0f}초')

def series(x, scorer):
    # 점수기 이름의 첫 단어로 모델을 고른다: B1/B2 → 기준선, E2 → E2 원본, S42 등 → 시드 모델
    key = scorer.split()[0]
    S = SC['B'] if key in ('B1', 'B2') else SC[key]
    if '문장 끝' in scorer:
        return [(t, S[q]) for t, q in x['pts2'] if t in x['ends']]
    s = int(scorer.split('s=')[1].rstrip(')'))
    return [(t, S[q]) for t, q in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % s == 0]
""")
code(EVAL_DEFS)

md("## 9. 결과")
code(r"""
MODELS = ['B'] + (['E2'] if 'E2' in SC else []) + [f'S{s}' for s in SEEDS]
DEV_Q, DEV_Y = point_labels(dev_items)
rows = []
for mname in MODELS:
    names = (['B1 문장 끝', 'B2 앞부분 (s=4)', 'B2 앞부분 (s=2)'] if mname == 'B' else
             [f'{mname} 문장 끝', f'{mname} 앞부분 (s=4)', f'{mname} 앞부분 (s=2)'])
    ys, ss = [], []
    for x in dev_items:
        d = dict(x['pts2'])
        for sn in x['sents']:
            ys.append(sn['y']); ss.append(SC[mname][d[sn['t1']]])
    auc_full, auc_pref = roc_auc_score(ys, ss), roc_auc_score(DEV_Y, SC[mname][DEV_Q])
    for sc in names:
        r100 = evaluate(dev_items, sc, calibrate(val_items, sc, 100))
        rsv = evaluate(dev_items, sc, calibrate_flag(val_items, sc, 0.10))
        rows.append({'model': mname, 'scorer': sc, 'recall_ARL100': r100['recall'],
                     'delay_med_ARL100': r100['delay_tok_median'], 'delay_chars_ARL100': r100['delay_chars_median'],
                     'clean_flag_ARL100': r100['clean_resp_flagged'],
                     'recall_svc10': rsv['recall'], 'delay_med_svc10': rsv['delay_tok_median'],
                     'clean_flag_svc10': rsv['clean_resp_flagged'], 'auc_full': auc_full, 'auc_prefix': auc_pref})
res = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(res.round(3).to_string(index=False))
""")

code(r"""
# 시드 평균 ± 표준편차
seed_rows = res[res.model.str.match(r'S\d+')].copy()
seed_rows['kind'] = seed_rows.scorer.str.split(' ', n=1).str[1]
agg = seed_rows.groupby('kind')[['recall_ARL100', 'delay_med_ARL100', 'recall_svc10', 'delay_med_svc10',
                                  'auc_full', 'auc_prefix']].agg(['mean', 'std']).round(3)
print('=== 시드 3개 평균 ± 표준편차 ===')
print(agg.to_string())
""")

md("## 10. 사전 기준 판정과 저장")
code(r"""
print('=== 안정성: 시드마다 E2 주 기준 (s=4 또는 s=2에서 재현율 ≥ 0.474 그리고 지연 ≤ 9토큰) ===')
per_seed = {}
for seed in SEEDS:
    g = res[res.model == f'S{seed}'].set_index('scorer')
    oks = []
    for s in (4, 2):
        r = g.loc[f'S{seed} 앞부분 (s={s})']
        oks.append(r.recall_ARL100 >= 0.474 and r.delay_med_ARL100 <= 9)
        print(f'  시드 {seed:4} s={s}: 재현율 {r.recall_ARL100:.3f}, 지연 {r.delay_med_ARL100:.1f} → {"충족" if oks[-1] else "미달"}')
    per_seed[seed] = any(oks)
stable = all(per_seed.values())
m = agg.loc['앞부분 (s=4)']
mean_ok = m[('recall_ARL100', 'mean')] >= 0.474 and m[('delay_med_ARL100', 'mean')] <= 9
print(f"\n안정성(시드 3개 모두 충족): {'충족' if stable else '미달'} — {per_seed}")
print(f"평균(s=4): 재현율 {m[('recall_ARL100','mean')]:.3f} ± {m[('recall_ARL100','std')]:.3f}, "
      f"지연 {m[('delay_med_ARL100','mean')]:.1f} ± {m[('delay_med_ARL100','std')]:.1f} → {'충족' if mean_ok else '미달'}")
ep2 = 0
for seed in SEEDS:
    h = pd.read_csv(os.path.join(SAVE_ROOT, f'seed{seed}_ep{EPOCHS}', 'hist.csv'))
    best = h.loc[h.val_prefix_auroc.idxmax()]
    ep2 += best.step > h.step.max() // 2
print(f'2에폭 효과: 두 번째 에폭에서 체크포인트가 선택된 시드 {ep2}/{len(SEEDS)}')
print('\n결론:', 'E2 결과는 시드에 안정적' if stable and mean_ok else 'E2 결과는 시드에 민감 — 중간보고에서는 시드 평균만 주장')

os.makedirs(os.path.join(OUT_DIR, 'e4lite'), exist_ok=True)
res.to_csv(os.path.join(OUT_DIR, 'e4lite', 'e4lite_results.csv'), index=False)
agg.to_csv(os.path.join(OUT_DIR, 'e4lite', 'e4lite_seed_agg.csv'))
hs = []
for seed in SEEDS:
    h = pd.read_csv(os.path.join(SAVE_ROOT, f'seed{seed}_ep{EPOCHS}', 'hist.csv')); h['seed'] = seed; hs.append(h)
pd.concat(hs).to_csv(os.path.join(OUT_DIR, 'e4lite', 'e4lite_train_hist.csv'), index=False)
print('저장:', os.path.join(OUT_DIR, 'e4lite'))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(NB_DIR, 'e4lite_seed_stability.ipynb'), 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
