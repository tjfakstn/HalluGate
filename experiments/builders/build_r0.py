"""R0 노트북 생성: 근거 문서가 학습과 겹치지 않는 큰 보정·평가 세트(cal-L, dev-L)를 만들고,
기존 dev의 문서 중복(오염) 효과를 재고, 이후 R2(순차 판정)·R3(보정) 실험에 쓸 앞부분 점수 궤적을 저장한다.
데이터·전처리·근거 검색·평가 코드는 E2 노트북(그리고 그 안의 E1 코드)에서 그대로 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
e2 = json.load(open(os.path.join(NB_DIR, 'e2_prefix_trained_scorer.ipynb')))
C = [''.join(c['source']) for c in e2['cells']]
def find(prefix):
    return next(c for c in C if c.startswith(prefix))

ENV = find("!pip -q install pysbd").replace(
    "SAVE_DIR = os.path.join(OUT_DIR, 'student_prefix')",
    "E2_DIR = os.path.join(OUT_DIR, 'student_prefix')\n"
    "SEED_ROOT = os.path.join(OUT_DIR, 'student_prefix_seeds')\n"
    "R0_DIR = os.path.join(OUT_DIR, 'r0')\n"
    "os.makedirs(R0_DIR, exist_ok=True)")
assert 'R0_DIR' in ENV
DATA = find("BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find("# 학습 응답: S2/S2-b의 gold-large")
PREP = find("ENC = tiktoken.get_encoding")
RETR = find("from sentence_transformers import SentenceTransformer").split("\nt0 = time.perf_counter()")[0]
RETR = RETR.split("\n@torch.no_grad()")[0]          # score()는 여러 모델을 한 번에 도는 버전으로 새로 쓴다
PTS = find("def points(item, s):")
POINTS_DEF = PTS.split("\nQUERY_ID")[0]
QID_DEF = "QUERY_ID" + PTS.split("\nQUERY_ID")[1].split("\nfor x in dev_items")[0]
PLAB_DEF = "def point_labels" + PTS.split("def point_labels")[1].split("\nby_all")[0]
EVAL_DEFS = find("def calibrate(items, scorer, arl0):")
for name, c in [('points', POINTS_DEF), ('qid', QID_DEF), ('point_labels', PLAB_DEF)]:
    assert f'def {name}' in c, name
assert 'def evaluate' in EVAL_DEFS and 'def calibrate_flag' in EVAL_DEFS

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R0 — 평가 세트 확장과 근거 문서 중복(오염) 점검

## 왜 하나

다음 모델링(R2 순차 판정 규칙, R3 오경보 보장 보정, R1 발생 시점 학습)을 시작하기 전에 **평가 자체를 고쳐야** 한다. 문제는 두 가지다.

1. **검정력 부족**: 지금 dev의 환각 응답은 154개뿐이다. 엄격 재현율의 95% 구간이 ±0.08이고, 시드 간 편차도 0.015~0.03이다(E4-lite). 0.05 수준의 개선은 측정할 수 없다.
2. **근거 문서 중복**: RAGTruth는 근거 문서 하나마다 6개 모델의 응답이 붙어 있다. 기존 분할은 **응답** 단위로 나눴기 때문에, dev 응답의 **62%가 학습(gold-large)에 쓰인 근거 문서와 같은 문서**를 쓴다. 판정기는 같은 문서에 대한 다른 모델의 응답과 정답을 이미 학습했다. dev 수치가 낙관적일 수 있다.

## 무엇을 하나

1. train 분할에서 **학습·검증·dev 어느 것과도 근거 문서가 겹치지 않는** 응답만 모은다(약 3,200개, 근거 536개, 환각 응답 약 1,500개).
   근거 문서 단위로, 과제별로 반씩 나눠 **cal-L(보정·규칙 학습용)** 과 **dev-L(평가용)** 을 만든다. 새 시드 `random.Random(2027)`.
2. 기존 판정기 5개(student_vark = B, student_prefix = E2, E4-lite 시드 모델 S42·S7·S2026)로 네 세트(dev, 검증, cal-L, dev-L)의 **모든 판정 지점(2토큰 간격 + 문장 끝) 점수**를 계산해 드라이브에 저장한다.
   이 점수 궤적이 R2·R3의 재료다. 이후 실험은 GPU 없이 이 파일로 돈다.
3. 기존 dev 안에서 **학습에 본 근거 문서(seen)** 와 **처음 보는 문서(unseen)** 의 성능을 따로 재서 오염 효과를 직접 측정한다.
4. dev-L에서 모든 판정기를 E1 프로토콜(엄격 기준, ARL₀=100, 서비스 10%) 그대로 다시 평가한다. 임계값은 cal-L로 보정한다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **R0-a 분할 무결성**: cal-L·dev-L의 근거 문서가 학습(gold-large)·검증·dev와 겹치지 않고, cal-L과 dev-L끼리도 겹치지 않는다 (`assert`).
- **R0-b 재현**: 기존 dev에서 B·E2 점수가 E4-lite 기록과 일치한다. 완성 문장 AUROC 0.859 / 0.881 (±0.003), ARL₀=100 엄격 재현율 B1 0.474 / E2 s=4 0.494 (±0.013, 응답 2개). 어긋나면 이후 수치는 해석하지 않는다.
- **R0-c 오염 판정**: 기존 dev에서 B의 완성 문장 AUROC 차이(seen − unseen, **과제별 차이의 평균**)가 **0.02 초과이고 부트스트랩 95% 구간 하한이 0보다 크면** "기존 dev 수치는 문서 중복으로 부풀려짐"으로 기록한다. 그렇지 않으면 "오염 영향은 측정 범위 안에서 작음"으로 기록한다.
- **결정(결과와 무관하게 지금 고정)**: 이후 모든 주장의 주 평가 세트는 **dev-L**, 임계값 보정과 규칙 학습은 **cal-L**로 한다. 기존 dev 수치는 참고로만 쓴다.
- **보고(판정 없음)**: dev-L에서 모든 판정기의 재현율·지연·AUROC, 쌍체 부트스트랩 구간, 기존 dev 대비 구간 폭.

> 런타임 **GPU(T4)**. 판정 질의 약 35만~40만 개 × 모델 5개 → **약 1시간**. 점수는 2만 개 단위 조각으로 드라이브 `selfproject/r0/scores/`에 저장되므로,
> 끊겨도 다시 실행하면 끝난 조각은 건너뛴다. 드라이브에 `student_vark`, `student_prefix`, `student_prefix_seeds/seed*_ep2`가 있어야 한다(시드 모델은 없으면 건너뜀).
""")

md("## 1. 환경 준비")
code(ENV)

md("## 2. 데이터 — 기존 dev·검증·학습 분할 (E1·E2 코드 그대로)")
code(DATA)
code(TRAIN_SPLIT)

md(r"""## 3. 새 분할 — 근거 문서가 겹치지 않는 cal-L / dev-L

근거 문서(source_id) 단위로 나눈다. 같은 문서의 응답 6개는 항상 같은 쪽에 들어간다.""")
code(r"""
S = lambda rs: {r['source_id'] for r in rs}
TRAIN_SRC = S(train)
used_src = TRAIN_SRC | S(val) | S(dev) | S(teacher_resp)
cand_src = sorted(S(train_pool) - used_src)
r0rng = random.Random(2027)
cal_src, devL_src = set(), set()
for tt in TASKS:
    ss = [s for s in cand_src if sources[s]['task_type'] == tt]
    r0rng.shuffle(ss)
    h = len(ss) // 2
    cal_src |= set(ss[:h]); devL_src |= set(ss[h:])
calL = [r for r in train_pool if r['source_id'] in cal_src]
devL = [r for r in train_pool if r['source_id'] in devL_src]

# R0-a 분할 무결성
assert not (S(calL) & used_src) and not (S(devL) & used_src), '새 분할이 기존 분할과 근거 문서를 공유'
assert not (S(calL) & S(devL)), 'cal-L과 dev-L이 근거 문서를 공유'
assert all(r['split'] == 'train' for r in calL + devL), 'test 분할 사용 금지'
ids = lambda rs: {str(r['id']) for r in rs}
assert not (ids(calL) & ids(devL)) and not ((ids(calL) | ids(devL)) & (ids(train) | ids(val) | ids(dev)))
print('R0-a 분할 무결성: 충족')

def desc(name, rs):
    n_h = sum(1 for r in rs if r.get('labels'))
    tt = pd.Series([r['task_type'] for r in rs]).value_counts().to_dict()
    print(f'{name:6} 응답 {len(rs):5} | 근거 {len(S(rs)):4} | 환각 응답 {n_h:5} | 과제 {tt}')
for name, rs in [('dev', dev), ('검증', val), ('cal-L', calL), ('dev-L', devL)]:
    desc(name, rs)
seen = {str(r['id']) for r in dev if r['source_id'] in TRAIN_SRC}
print(f'기존 dev 중 학습 근거 문서를 공유하는 응답: {len(seen)}/{len(dev)} ({len(seen)/len(dev):.0%})')

json.dump({'seed': 2027, 'cal_L_ids': sorted(ids(calL)), 'dev_L_ids': sorted(ids(devL)),
           'cal_L_sources': sorted(cal_src), 'dev_L_sources': sorted(devL_src), 'dev_seen_ids': sorted(seen)},
          open(os.path.join(R0_DIR, 'r0_splits.json'), 'w'))
""")

md("## 4. 토큰 스트림·문장·발생 위치 (E1 코드 그대로)")
code(PREP)
code(r"""
calL_items, devL_items = to_items(calL), to_items(devL)
by_all = {str(r['id']): r for r in dev + val + calL + devL}
ALL = {'dev': dev_items, 'val': val_items, 'calL': calL_items, 'devL': devL_items}
for name, its in ALL.items():
    for x in its:
        x['labels'] = by_all[x['id']].get('labels') or []
        text = x['text']
        x['cs'] = [a + (len(text[a:b]) - len(text[a:b].lstrip())) if text[a:b].strip() else a
                   for a, b in zip(x['ts'], x['te'])]
        x['seen'] = x['id'] in seen
    n_h = sum(x['onset'] is not None for x in its)
    print(f'{name:5} 응답 {len(its):5} (환각 {n_h:5}) | 토큰 {sum(x["n_tok"] for x in its):,}')
""")

md("## 5. 판정 지점 (E1과 같음: 문장 안 2토큰 간격 + 문장 끝)")
code(POINTS_DEF + "\n\n" + QID_DEF + "\n\n" + PLAB_DEF + r"""

for its in ALL.values():
    for x in its:
        x['pts2'] = [(t, qid(x['ref_key'], txt)) for t, txt in points(x, 2) if txt]
Q_KEYS, Q_TEXTS = [q[0] for q in queries], [q[1] for q in queries]
print(f'판정 질의 {len(queries):,}개')
""")

md(r"""## 6. 근거 검색과 판정 — 모델 5개를 한 번에, 조각마다 드라이브에 저장

근거 검색(top-8 조각)은 질의마다 한 번만 하고, 같은 입력을 모든 판정기에 넣는다.
조각 파일에 없는 모델만 새로 계산하므로, 나중에 시드 모델이 생겨도 이어서 채울 수 있다.""")
code(RETR + r"""

import hashlib, math
MODEL_DIRS = {'B': STUDENT_DIR, 'E2': E2_DIR}
MODEL_DIRS.update({f'S{s}': os.path.join(SEED_ROOT, f'seed{s}_ep2') for s in [42, 7, 2026]})
missing = [k for k, v in MODEL_DIRS.items() if not os.path.exists(os.path.join(v, 'config.json'))]
MODEL_DIRS = {k: v for k, v in MODEL_DIRS.items() if k not in missing}
assert 'B' in MODEL_DIRS and 'E2' in MODEL_DIRS, 'student_vark·student_prefix가 드라이브에 있어야 한다'
print('판정기:', list(MODEL_DIRS), '| 없어서 건너뜀:', missing)

TOKS, MODELS = {}, {}
for k, v in MODEL_DIRS.items():
    TOKS[k] = AutoTokenizer.from_pretrained(v)
    MODELS[k] = AutoModelForSequenceClassification.from_pretrained(v).to(DEVICE).eval()
same_tok = all(TOKS[k].get_vocab() == TOKS['B'].get_vocab() for k in TOKS)
print('토크나이저 공유:', same_tok)

SC_DIR = os.path.join(R0_DIR, 'scores'); os.makedirs(SC_DIR, exist_ok=True)
sig = hashlib.md5('\n'.join(f'{k}\t{t}' for k, t in queries).encode()).hexdigest()
meta_p = os.path.join(SC_DIR, 'meta.json')
if os.path.exists(meta_p):
    assert json.load(open(meta_p))['sig'] == sig, '저장된 조각과 질의 목록이 다름 — scores/ 폴더를 비우고 다시 실행'
json.dump({'sig': sig, 'n_queries': len(queries)}, open(meta_p, 'w'))

SHARD, BSZ = 20000, 128
order = sorted(range(len(queries)), key=lambda i: (len(Q_TEXTS[i]), i))
n_sh = math.ceil(len(order) / SHARD)

@torch.no_grad()
def run_batch(idx, names):
    ctx = contexts([Q_KEYS[j] for j in idx], [Q_TEXTS[j] for j in idx])
    txt = [Q_TEXTS[j] for j in idx]
    out, enc = {}, None
    for n in names:
        if enc is None or not same_tok:
            enc = TOKS[n](ctx, txt, truncation='only_first', max_length=MAX_LEN,
                          padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == 'cuda')):          # E2와 같은 정밀도 (재현 확인용)
            out[n] = MODELS[n](**enc).logits.squeeze(-1).float().cpu().numpy()
    return out

t0 = time.perf_counter(); done_q = 0
for sh in range(n_sh):
    p = os.path.join(SC_DIR, f'shard{sh:03d}.npz')
    idx_all = np.array(order[sh * SHARD:(sh + 1) * SHARD])
    have = dict(np.load(p)) if os.path.exists(p) else {}
    todo = [n for n in MODEL_DIRS if n not in have]
    if not todo:
        continue
    res = {n: np.zeros(len(idx_all), dtype=np.float32) for n in todo}
    for i in range(0, len(idx_all), BSZ):
        b = idx_all[i:i + BSZ]
        for n, v in run_batch(b.tolist(), todo).items():
            res[n][i:i + len(b)] = v
    have.update(res); have['idx'] = idx_all
    np.savez(p, **have)
    done_q += len(idx_all)
    el = time.perf_counter() - t0
    print(f'조각 {sh + 1}/{n_sh} 저장 ({", ".join(todo)}) | {el/60:.1f}분 경과, '
          f'남은 예상 {el / done_q * (len(order) - (sh + 1) * SHARD) / 60:.0f}분')

SC = {n: np.full(len(queries), np.nan, dtype=np.float32) for n in MODEL_DIRS}
for sh in range(n_sh):
    z = np.load(os.path.join(SC_DIR, f'shard{sh:03d}.npz'))
    for n in MODEL_DIRS:
        SC[n][z['idx']] = z[n]
assert all(not np.isnan(v).any() for v in SC.values()), '빠진 점수가 있음'
print(f'점수 준비 완료: {len(queries):,}개 × {len(SC)}개 모델')
""")

md(r"""## 7. 평가 함수 (E1 코드 그대로)

점수기 이름은 `모델|판정 시점` 형식이다. 예: `B|end`(문장 끝), `E2|4`(4토큰마다 앞부분 + 문장 끝).""")
code(r"""
def series(x, scorer):
    m, k = scorer.split('|')
    S_ = SC[m]
    if k == 'end':
        return [(t, S_[q]) for t, q in x['pts2'] if t in x['ends']]
    s = int(k)
    return [(t, S_[q]) for t, q in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % s == 0]
""")
code(EVAL_DEFS)
code(r"""
def sent_auc(items, m):
    ys, ss = [], []
    for x in items:
        d = dict(x['pts2'])
        for sn in x['sents']:
            if sn['t1'] in d:
                ys.append(sn['y']); ss.append(SC[m][d[sn['t1']]])
    return roc_auc_score(ys, ss) if len(set(ys)) == 2 else np.nan

def prefix_auc(items, m):
    q, y = point_labels(items)
    return roc_auc_score(y, SC[m][q])
""")

md(r"""## 8. R0-b 재현 — 기존 dev에서 E4-lite 기록과 같은가""")
code(r"""
REF = {'auc_B': 0.8590, 'auc_E2': 0.8807, 'rec_B1': 0.4740, 'rec_E2s4': 0.4935}
got = {'auc_B': sent_auc(dev_items, 'B'), 'auc_E2': sent_auc(dev_items, 'E2'),
       'rec_B1': evaluate(dev_items, 'B|end', calibrate(val_items, 'B|end', 100))['recall'],
       'rec_E2s4': evaluate(dev_items, 'E2|4', calibrate(val_items, 'E2|4', 100))['recall']}
TOL = {'auc_B': 0.003, 'auc_E2': 0.003, 'rec_B1': 0.013, 'rec_E2s4': 0.013}
R0B = all(abs(got[k] - REF[k]) <= TOL[k] for k in REF)
for k in REF:
    print(f'{k:9} 기록 {REF[k]:.4f} | 지금 {got[k]:.4f} | {"일치" if abs(got[k]-REF[k]) <= TOL[k] else "불일치"}')
print('R0-b 재현:', '충족' if R0B else '미달 — 이후 수치를 해석하지 말고 원인부터 확인')
""")

md(r"""## 9. R0-c 오염 — 기존 dev의 seen(학습 근거 문서 공유) vs unseen

과제 구성이 두 집단에서 다를 수 있으므로, **과제별 AUROC 차이의 평균**을 주 지표로 쓴다. 구간은 응답 단위 부트스트랩(1,000회).""")
code(r"""
def strat_diff(items, m, rng=None):
    ds = []
    for tt in TASKS:
        its = [x for x in items if x['task_type'] == tt]
        if rng is not None:
            its = [its[i] for i in rng.integers(0, len(its), len(its))]
        a = sent_auc([x for x in its if x['seen']], m)
        b = sent_auc([x for x in its if not x['seen']], m)
        ds.append(a - b)
    return float(np.nanmean(ds))

brng = np.random.default_rng(0)
rows = []
for m in SC:
    d0 = strat_diff(dev_items, m)
    bs = [strat_diff(dev_items, m, brng) for _ in range(1000)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    rows.append({'model': m, 'auc_seen': sent_auc([x for x in dev_items if x['seen']], m),
                 'auc_unseen': sent_auc([x for x in dev_items if not x['seen']], m),
                 'strat_diff': d0, 'ci_lo': lo, 'ci_hi': hi,
                 'auc_dev_all': sent_auc(dev_items, m), 'auc_devL': sent_auc(devL_items, m)})
contam = pd.DataFrame(rows).set_index('model')
pd.set_option('display.width', 250)
print(contam.round(3).to_string())
b = contam.loc['B']
R0C = b.strat_diff > 0.02 and b.ci_lo > 0
print(f"\nR0-c (B, 과제별 차이 평균 {b.strat_diff:+.3f} [{b.ci_lo:+.3f}, {b.ci_hi:+.3f}]): "
      + ('오염 효과 있음 — 기존 dev 수치는 부풀려짐' if R0C else '오염 효과는 측정 범위 안에서 작음'))

# 과제별: 기존 dev(seen / unseen) vs dev-L
pt = []
for m in ['B', 'E2']:
    for tt in TASKS:
        d_ = [x for x in dev_items if x['task_type'] == tt]
        pt.append({'model': m, 'task': tt, 'dev_seen': sent_auc([x for x in d_ if x['seen']], m),
                   'dev_unseen': sent_auc([x for x in d_ if not x['seen']], m),
                   'devL': sent_auc([x for x in devL_items if x['task_type'] == tt], m)})
contam_task = pd.DataFrame(pt)
print('\n과제별 완성 문장 AUROC'); print(contam_task.round(3).to_string(index=False))

# 엄격 재현율도 seen/unseen으로 (임계값은 기존 검증으로 보정, E2와 같음)
for sc in ['B|end', 'E2|4']:
    thr = calibrate(val_items, sc, 100)
    for flag in [True, False]:
        its = [x for x in dev_items if x['seen'] == flag]
        r = evaluate(its, sc, thr)
        n_h = sum(x['onset'] is not None for x in its)
        print(f"{sc:6} {'seen  ' if flag else 'unseen'} 환각 {n_h:3}개 | 엄격 재현율 {r['recall']:.3f} | 멀쩡한 응답 표시 {r['clean_resp_flagged']:.1%}")
""")

md(r"""## 10. dev-L 평가 — cal-L로 보정, E1 프로토콜 그대로""")
code(r"""
res = []
for m in SC:
    auc_f, auc_p = sent_auc(devL_items, m), prefix_auc(devL_items, m)
    for k in ['end', '4', '2']:
        sc = f'{m}|{k}'
        thr100 = calibrate(calL_items, sc, 100)
        r = evaluate(devL_items, sc, thr100)
        thr10 = calibrate_flag(calL_items, sc, 0.10)
        r10 = evaluate(devL_items, sc, thr10)
        res.append({'model': m, 'kind': k, 'recall_ARL100': r['recall'], 'delay_med_ARL100': r['delay_tok_median'],
                    'delay_chars_ARL100': r['delay_chars_median'], 'realized_ARL0': r['realized_ARL0_dev'],
                    'clean_flag_ARL100': r['clean_resp_flagged'], 'pre_onset_ARL100': r['pre_onset_alarm_rate'],
                    'recall_svc10': r10['recall'], 'delay_med_svc10': r10['delay_tok_median'],
                    'clean_flag_svc10': r10['clean_resp_flagged'], 'auc_full': auc_f, 'auc_prefix': auc_p,
                    'thr100': thr100, 'thr10': thr10})
devL_res = pd.DataFrame(res)
print(devL_res.drop(columns=['thr100', 'thr10']).round(3).to_string(index=False))

seeds = [m for m in SC if m.startswith('S')]
if seeds:
    agg = devL_res[devL_res.model.isin(seeds)].groupby('kind')[
        ['recall_ARL100', 'delay_med_ARL100', 'recall_svc10', 'delay_med_svc10', 'auc_full', 'auc_prefix']].agg(['mean', 'std'])
    print('\n시드 평균 ± 표준편차 (dev-L)'); print(agg.round(3).to_string())
""")

md(r"""## 11. 쌍체 부트스트랩 — dev-L에서 구간이 얼마나 좁아졌나

임계값은 고정하고 환각 응답만 재표집(1,000회). 같은 계산을 기존 dev(검증 보정)에도 해서 구간 폭을 비교한다.""")
code(r"""
def per_resp(items, sc, thr):
    out = []
    for x in items:
        if x['onset'] is None: continue
        first = next((t for t, s in series(x, sc) if s >= thr), None)
        ok = first is not None and first >= x['onset']
        out.append((ok, first - x['onset'] if ok else np.nan))
    return np.array([o[0] for o in out], dtype=float), np.array([o[1] for o in out])

def boot(items, cal_items, a, b, n=1000, seed=0):
    da, la = per_resp(items, a, calibrate(cal_items, a, 100))
    db, lb = per_resp(items, b, calibrate(cal_items, b, 100))
    rng = np.random.default_rng(seed); N = len(da)
    rd, dd, ra = [], [], []
    for _ in range(n):
        i = rng.integers(0, N, N)
        rd.append(da[i].mean() - db[i].mean()); ra.append(da[i].mean())
        dd.append(np.nanmedian(la[i]) - np.nanmedian(lb[i]))
    q = lambda v: np.nanpercentile(v, [2.5, 97.5])
    return {'A': a, 'B': b, 'n_hall': N, 'recall_A': da.mean(), 'recall_B': db.mean(),
            'recall_diff': da.mean() - db.mean(), 'recall_diff_lo': q(rd)[0], 'recall_diff_hi': q(rd)[1],
            'recall_A_lo': q(ra)[0], 'recall_A_hi': q(ra)[1],
            'delay_diff': np.nanmedian(la) - np.nanmedian(lb), 'delay_diff_lo': q(dd)[0], 'delay_diff_hi': q(dd)[1]}

pairs = [('E2|4', 'B|end'), ('E2|4', 'B|4')] + [(f'{s}|4', 'B|end') for s in seeds] + [(f'{s}|4', 'B|4') for s in seeds]
bt = [dict(boot(devL_items, calL_items, a, b), set='dev-L') for a, b in pairs]
bt += [dict(boot(dev_items, val_items, a, b), set='dev (기존)') for a, b in pairs[:2]]
boot_df = pd.DataFrame(bt)
print(boot_df[['set', 'A', 'B', 'n_hall', 'recall_A', 'recall_A_lo', 'recall_A_hi', 'recall_diff', 'recall_diff_lo',
               'recall_diff_hi', 'delay_diff', 'delay_diff_lo', 'delay_diff_hi']].round(3).to_string(index=False))
w = boot_df.assign(width=boot_df.recall_A_hi - boot_df.recall_A_lo).groupby('set').width.mean()
print('\nE2 s=4 재현율 95% 구간 폭:', w.round(3).to_dict())
""")

md(r"""## 12. 판정 요약과 저장

드라이브 `selfproject/r0/`에 결과 표와, R2·R3 실험에 쓸 **판정 지점·점수 궤적**을 저장한다.""")
code(r"""
import pickle, shutil
print('=== R0 사전 기준 ===')
print('R0-a 분할 무결성: 충족 (assert 통과)')
print('R0-b 재현        :', '충족' if R0B else '미달')
print('R0-c 오염        :', '오염 효과 있음' if R0C else '측정 범위 안에서 작음',
      f"(B 과제별 차이 평균 {contam.loc['B'].strat_diff:+.3f}, 95% [{contam.loc['B'].ci_lo:+.3f}, {contam.loc['B'].ci_hi:+.3f}])")
print('결정            : 이후 주 평가 세트 = dev-L, 보정·규칙 학습 = cal-L')

devL_res.to_csv(os.path.join(R0_DIR, 'r0_devL.csv'), index=False)
contam.to_csv(os.path.join(R0_DIR, 'r0_contamination.csv'))
contam_task.to_csv(os.path.join(R0_DIR, 'r0_contamination_task.csv'), index=False)
boot_df.to_csv(os.path.join(R0_DIR, 'r0_bootstrap.csv'), index=False)
pd.DataFrame([{'check': k, 'ref': REF[k], 'got': got[k]} for k in REF]).to_csv(os.path.join(R0_DIR, 'r0_repro.csv'), index=False)

KEEP = ['id', 'task_type', 'gen_model', 'ref_key', 'n_tok', 'onset', 'sents', 'sent_of', 'ends', 'pts2', 'ts', 'te', 'seen']
pickle.dump({'items': {k: [{f: x[f] for f in KEEP} for x in v] for k, v in ALL.items()},
             'q_keys': Q_KEYS, 'q_texts': Q_TEXTS}, open(os.path.join(R0_DIR, 'r0_items.pkl'), 'wb'))
np.savez(os.path.join(R0_DIR, 'r0_scores.npz'), **SC)
for f in ['r0_devL.csv', 'r0_contamination.csv', 'r0_contamination_task.csv', 'r0_bootstrap.csv', 'r0_repro.csv', 'r0_splits.json']:
    shutil.copy(os.path.join(R0_DIR, f), f)
os.system('zip -q r0_results.zip r0_devL.csv r0_contamination.csv r0_contamination_task.csv r0_bootstrap.csv r0_repro.csv')
print('저장 완료: 드라이브 selfproject/r0/ (r0_items.pkl, r0_scores.npz 포함), 업로드용 r0_results.zip')
try:
    from google.colab import files; files.download('r0_results.zip')
except Exception:
    pass
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r0_eval_expansion.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
