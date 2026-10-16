"""R4c 노트북 생성: R4b의 보정 학습 정답(Sonnet 재판정)으로 큰 판정기(DeBERTa-v3-base, 0.18B)를 학습한다.
학습 방식은 R1c base와 같고(백본·학습률·체크포인트 선택) 학습 정답만 R4b와 같이 보정한다.
데이터·평가 코드는 R1b, 학습 루프는 R1c, 보정 정답 적용은 R4b 노트북에서 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
def cells_of(name):
    nb = json.load(open(os.path.join(NB_DIR, name)))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
R1B, R1C, R4B = cells_of('r1b_sentence_local_labels.ipynb'), cells_of('r1c_judge_capacity.ipynb'), cells_of('r4b_relabel_retrain.ipynb')
def find(C, prefix):
    return next(c for c in C if c.startswith(prefix))
def sub(s, a, b):
    assert a in s, a[:70]
    return s.replace(a, b, 1)

ENV = find(R1B, "!pip -q install pysbd")
ENV = sub(ENV, "R1_DIR = os.path.join(OUT_DIR, 'r1b')\nR1_OLD = os.path.join(OUT_DIR, 'r1')",
          "R1_DIR = os.path.join(OUT_DIR, 'r4c')\nR1B_DIR, R1C_DIR, R4_DIR, R4B_DIR = (os.path.join(OUT_DIR, d) for d in ['r1b', 'r1c', 'r4', 'r4b'])")
ENV = sub(ENV, "assert os.path.isdir(STUDENT_DIR), f'{STUDENT_DIR} 없음'",
          "for p in [os.path.join(R4B_DIR, 'labels_train.json'), os.path.join(R4_DIR, 'r4_corrected_labels.json')]:\n    assert os.path.exists(p), f'{p} 없음'")
DATA = find(R1B, "BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find(R1B, "# 학습 응답: S2/S2-b의 gold-large")
PREP = find(R1B, "ENC = tiktoken.get_encoding")
LABELS = find(R1B, "def points(item, s):")
LABELS = sub(LABELS, "train_items = to_items(train)", "train_items = TRAIN_ITEMS")
LABELS = sub(LABELS, "    son = [s_onset(sn) for sn in x['sents']]",
             "    son = [s_onset(sn) for sn in x['sents']]\n"
             "    son = [o if o is not None else ADD_ONSET.get((x['id'], i)) for i, o in enumerate(son)]   # R4b 보정 학습 정답")
ADD = find(R4B, "coverage = len(LABELS)")
ADD = ADD[ADD.index("def content_starts(x):"):]                     # R4b-0 판정 부분은 빼고 보정 정답 적용만
RETR = find(R1B, "from sentence_transformers import SentenceTransformer")
CORR = find(R4B, "# R4 보정 정답 C1")
SERIES = find(R1B, "SC = dict(Z0)")
EVAL_DEFS = find(R1B, "def calibrate(items, scorer, arl0):")
AUCS = find(R1B, "def resp_auc(items, m):")
TRAIN = find(R1C, "from transformers import get_linear_schedule_with_warmup")
TRAIN = sub(TRAIN, "        np.save(sp, full)\n",
            "        np.save(sp, full)\n"
            "        if seed not in KEEP_WEIGHTS:                     # 드라이브 용량: 가중치는 시드 42만 남긴다\n"
            "            for f in ['model.safetensors', 'pytorch_model.bin']:\n"
            "                if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))\n")

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R4c — 고친 학습 정답으로 큰 판정기(DeBERTa-v3-base)를 학습하면 문장 끝 판정을 넘는가

## 왜 하나

- R1c: 원래 정답에서는 판정기를 0.08B → 0.18B로 키워도 4토큰 엄격 재현율이 +0.027에 그쳤다.
- R4: 보정 정답에서는 위약을 빼면 판정기가 클수록 순이득이 컸다(+0.02 → +0.07). **정답 잡음이 크기 효과를 가렸을** 가능성.
- R4b: 학습 정답까지 고치자 작은 판정기(MiniLM)의 4토큰 재현율이 0.519 → 0.588로 올라 B 문장 끝(0.621)에 근접했다.

남은 질문: **고친 학습 정답 + 큰 판정기**면 4토큰 앞부분 판정이 문장 끝 판정을 따라잡거나 넘는가.

## 무엇을 하나

- 학습 방식은 **R1c base와 같다**(`cross-encoder/nli-deberta-v3-base`, 학습률 2e-5, 1에폭, 지점별 이진 분류, 배치 64·작은 배치 16). 학습 정답만 **R4b와 같이 보정**(드라이브 `r4b/labels_train.json`, Sonnet 재판정, API 추가 비용 없음).
- 체크포인트 선택은 R4b와 같이 cal-sub의 보정 정답(C1). 시드 42·7·2026.
- 비교: R1c base(같은 백본, 원래 학습 정답), R4b V0b(MiniLM, 같은 보정 학습 정답), B 문장 끝.
- 판정 방식 4토큰 고정(R1c·R4b와 같음). 평가는 dev-L 원래 정답·C1 둘 다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **R4c-1 보정 정답에서의 크기 효과(주)**: C1 4토큰 **base(보정 학습) 시드 평균 − R4b V0b 시드 평균 ≥ +0.03**.
- **R4c-2 목표**: C1 4토큰 base(보정 학습) 시드 평균 ≥ **B 문장 끝 C1 재현율(0.621)**. (보고: 시드별 B 문장 끝 대비 쌍체 부트스트랩)
- **R4c-3 비열등(원래 정답)**: 원래 정답 4토큰 base(보정 학습) − R1c base(원래 학습) 시드 평균 ≥ −0.02.
- 보고: 학습 정답 효과의 재현(C1에서 base 보정 − R1c base), 문장 끝 판정, 지연, 서비스 10%, AUROC, 과제별.
- 시드 3개가 모두 끝나야 판정한다. 일부만 끝나면 보고만 한다.
- 해석 규칙: R4c-1 충족이면 "정답이 깨끗해야 판정기 크기가 효과를 낸다". R4c-2 충족이면 "4토큰 앞부분 판정이 문장 끝 판정과 같거나 높은 재현율을 더 빨리" → 게이트웨이 교체 1순위(판정 1회 약 30ms, A100).

> **런타임: A100** (L4 가능, T4 비권장). R1c base 실측 시드당 80~145분 → **3시드 약 4~7시간**. 끝난 시드는 건너뛰므로 끊겨도 다시 실행하면 이어서 한다.
> **드라이브 용량**: 시드 42 가중치(약 740MB)만 남기고 나머지는 점수만 저장한다. **실행 전에 드라이브 휴지통을 비울 것.**
> 드라이브에 `student_vark`, `r0/`, `r1b/`, `r1c/`(base·mini scores.npy), `r4/r4_corrected_labels.json`, `r4b/labels_train.json`·`V0b_s*/scores.npy`가 있어야 한다.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터 (R1b와 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
md("## 3. R4b 보정 학습 정답 불러오기 (Sonnet 재판정, 추가 비용 없음)")
code(r"""
TRAIN_ITEMS = to_items(train)
ITEMS = {f'tr-{x["id"]}': x for x in TRAIN_ITEMS}
LABELS = {k: {int(i): v for i, v in d.items()} for k, d in json.load(open(os.path.join(R4B_DIR, 'labels_train.json'))).items()}
print(f'학습 재판정 {len(LABELS):,}/{len(ITEMS):,}')
""")
code(ADD)
md("## 4. 학습 판정 지점 — 보정 학습 정답 (R4b와 같음)")
code(LABELS)
md("## 5. 근거 검색과 평가용 보정 정답(R4 C1)")
code(RETR)
code(CORR)
md("## 6. 평가 함수 (R1b와 같음)")
code(SERIES)
code(EVAL_DEFS)
code(AUCS)
md("## 7. 학습 — R1c base와 같은 루프, 학습 정답만 보정 (끝난 시드는 건너뜀)")
code("KEEP_WEIGHTS = {42}\n" + TRAIN)
code(r"""
SEEDS = [42, 7, 2026]
for seed in SEEDS:
    run('base', seed)                     # 드라이브 r4c/base_s*  (SC 키 'base_s*' = 보정 학습 base)
for s in SEEDS:
    for key, path in [(f'R1cbase_s{s}', os.path.join(R1C_DIR, f'base_s{s}', 'scores.npy')),
                      (f'R1cmini_s{s}', os.path.join(R1C_DIR, f'mini_s{s}', 'scores.npy')),
                      (f'R4bV0b_s{s}', os.path.join(R4B_DIR, f'V0b_s{s}', 'scores.npy')),
                      (f'R1bV0b_s{s}', os.path.join(R1B_DIR, f'V0b_s{s}', 'scores.npy'))]:
        if os.path.exists(path): SC[key] = np.load(path)
need = [f'{f}_s{s}' for f in ['R1cbase', 'R4bV0b'] for s in SEEDS]
assert all(k in SC for k in need), f'비교 점수 없음: {[k for k in need if k not in SC]}'
print('비교 점수 준비 완료')
""")

md("## 8. dev-L 평가 (원래 정답·보정 정답)와 사전 기준 판정")
code(r"""
KINDS = ['end', '4']
FAMS = {'B': ['B'], 'E2': ['E2']}
for f in ['base', 'R1cbase', 'R1cmini', 'R4bV0b', 'R1bV0b']:
    FAMS[f] = [f'{f}_s{s}' for s in SEEDS if f'{f}_s{s}' in SC]
ALL_DONE = len(FAMS['base']) == 3
rows = []
for lab, (cal_, dev_) in LAB_SETS.items():
    for f, ms in FAMS.items():
        for m in ms:
            ra, sa = resp_auc(dev_, m), sent_auc(dev_, m)
            for k in KINDS:
                sc = f'{m}|{k}'
                r = evaluate(dev_, sc, calibrate(cal_, sc, 100)); r10 = evaluate(dev_, sc, calibrate_flag(cal_, sc, 0.10))
                rows.append({'labels': lab, 'family': f, 'model': m, 'kind': k, 'recall': r['recall'], 'delay_med': r['delay_tok_median'],
                             'pre_onset': r['pre_onset_alarm_rate'], 'svc_recall': r10['recall'], 'resp_auc': ra, 'sent_auc': sa})
res = pd.DataFrame(rows)
agg = res.groupby(['labels', 'family', 'kind'])[['recall', 'delay_med', 'pre_onset', 'svc_recall', 'resp_auc', 'sent_auc']].mean().reset_index()
agg['n'] = res.groupby(['labels', 'family', 'kind']).size().values
pd.set_option('display.width', 250)
print(agg.round(3).to_string(index=False))
fam = lambda f, k, lab, col='recall': float(agg[(agg.labels == lab) & (agg.family == f) & (agg.kind == k)][col].iloc[0])

def per_resp(items, sc, thr):
    return np.array([float((lambda f: f is not None and f >= x['onset'])(next((t for t, s in series(x, sc) if s >= thr), None)))
                     for x in items if x['onset'] is not None])
def boot_pair(a, b, lab, n=1000, seed=0):
    cal_, dev_ = LAB_SETS[lab]
    da, db = per_resp(dev_, a, calibrate(cal_, a, 100)), per_resp(dev_, b, calibrate(cal_, b, 100))
    rng = np.random.default_rng(seed); N = len(da)
    rd = [da[i].mean() - db[i].mean() for i in (rng.integers(0, N, N) for _ in range(n))]
    lo, hi = np.percentile(rd, [2.5, 97.5])
    return {'A': a, 'B': b, 'labels': lab, 'diff': da.mean() - db.mean(), 'lo': lo, 'hi': hi}

K = '4'
g1 = fam('base', K, 'C1') - fam('R4bV0b', K, 'C1')
g2 = fam('base', K, 'C1'); bend = fam('B', 'end', 'C1')
g3 = fam('base', K, 'orig') - fam('R1cbase', K, 'orig')
R4C_1, R4C_2, R4C_3 = g1 >= 0.03, g2 >= bend, g3 >= -0.02
tag = '' if ALL_DONE else '  (시드 3개 미완료 — 보고만)'
print(f"\n=== R4c-1 보정 정답 크기 효과: base {g2:.3f} − R4b V0b {fam('R4bV0b', K, 'C1'):.3f} = {g1:+.3f} (≥ +0.03) → {'충족' if R4C_1 else '미달'}{tag} ===")
print(f"=== R4c-2 목표: base 4토큰 {g2:.3f} vs B 문장 끝 {bend:.3f} (지연 {fam('base', K, 'C1', 'delay_med'):.1f} vs {fam('B', 'end', 'C1', 'delay_med'):.1f}) → {'충족' if R4C_2 else '미달'}{tag} ===")
BT = [boot_pair(f'base_s{s}|{K}', 'B|end', 'C1') for s in SEEDS if f'base_s{s}' in SC]
for b in BT:
    print(f"   {b['A']} vs B|end (C1): {b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
print(f"=== R4c-3 원래 정답 비열등: base 보정 {fam('base', K, 'orig'):.3f} − R1c base {fam('R1cbase', K, 'orig'):.3f} = {g3:+.3f} (≥ −0.02) → {'충족' if R4C_3 else '미달'}{tag} ===")
print(f"\n(보고) 학습 정답 효과(base): C1 {fam('base', K, 'C1') - fam('R1cbase', K, 'C1'):+.3f} | 참고 MiniLM(R4b): {fam('R4bV0b', K, 'C1') - fam('R1bV0b', K, 'C1'):+.3f}")
print(f"(보고) 서비스 10% 재현율 C1: base {fam('base', K, 'C1', 'svc_recall'):.3f} vs R4b V0b {fam('R4bV0b', K, 'C1', 'svc_recall'):.3f} vs R1c base {fam('R1cbase', K, 'C1', 'svc_recall'):.3f}")
print(f"(보고) 완성 문장 AUROC C1: base {fam('base', K, 'C1', 'sent_auc'):.3f} vs R4b V0b {fam('R4bV0b', K, 'C1', 'sent_auc'):.3f} vs R1c base {fam('R1cbase', K, 'C1', 'sent_auc'):.3f}")
""")

md("## 9. 과제별 (보고)")
code(r"""
tr = []
for lab, (cal_, dev_) in LAB_SETS.items():
    for tt in TASKS:
        its = [x for x in dev_ if x['task_type'] == tt]
        row = {'labels': lab, 'task': tt, 'n_hall': sum(x['onset'] is not None for x in its), 'B|end': evaluate(its, 'B|end', calibrate(cal_, 'B|end', 100))['recall']}
        for f in ['R1cbase', 'R4bV0b', 'base']:
            row[f'{f}|4'] = np.mean([evaluate(its, f'{m}|4', calibrate(cal_, f'{m}|4', 100))['recall'] for m in FAMS[f]])
        tr.append(row)
task_df = pd.DataFrame(tr)
print(task_df.round(3).to_string(index=False))
""")

md("## 10. 저장")
code(r"""
import shutil
summ = {'all_seeds': ALL_DONE, 'R4c_1': bool(R4C_1), 'gain_vs_R4bV0b_C1': g1, 'R4c_2': bool(R4C_2), 'base4_C1': g2, 'Bend_C1': bend,
        'boot_vs_Bend_C1': BT, 'R4c_3': bool(R4C_3), 'gain_vs_R1cbase_orig': g3}
json.dump(summ, open('r4c_summary.json', 'w'), default=float, ensure_ascii=False, indent=1)
hists = []
for s in SEEDS:
    p = os.path.join(R1_DIR, f'base_s{s}', 'hist.csv')
    if os.path.exists(p):
        h = pd.read_csv(p); h['run'] = f'base_s{s}'; hists.append(h)
out = {'r4c_devL.csv': res, 'r4c_family.csv': agg, 'r4c_task.csv': task_df, 'r4c_hist.csv': pd.concat(hists) if hists else pd.DataFrame()}
for f, d in out.items():
    d.to_csv(f, index=False); d.to_csv(os.path.join(R1_DIR, f), index=False)
shutil.copy('r4c_summary.json', os.path.join(R1_DIR, 'r4c_summary.json'))
print('저장 완료: 드라이브 selfproject/r4c/')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r4c_clean_labels_base.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
