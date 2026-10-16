"""R1c 노트북 생성: 판정기 크기(용량)가 엄격 재현율 약 0.50 상한의 원인인지 확인한다.
학습 방식은 R1b의 V0b(문장 단위 정답, 모든 판정 지점, 지점별 이진 분류)로 고정하고 백본만 바꾼다.
데이터·전처리·근거 검색·평가 코드는 R1b 노트북에서 그대로 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
r1b = json.load(open(os.path.join(NB_DIR, 'r1b_sentence_local_labels.ipynb')))
C = [''.join(c['source']) for c in r1b['cells'] if c['cell_type'] == 'code']
def find(prefix):
    return next(c for c in C if c.startswith(prefix))

ENV = find("!pip -q install pysbd").replace(
    "R1_DIR = os.path.join(OUT_DIR, 'r1b')\nR1_OLD = os.path.join(OUT_DIR, 'r1')",
    "R1_DIR = os.path.join(OUT_DIR, 'r1c')\nR1B_DIR = os.path.join(OUT_DIR, 'r1b')")
assert "R1B_DIR" in ENV
DATA = find("BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find("# 학습 응답: S2/S2-b의 gold-large")
PREP = find("ENC = tiktoken.get_encoding")
LABELS = find("def points(item, s):")
assert 'def s_onset' in LABELS
RETR = find("from sentence_transformers import SentenceTransformer")
SERIES = find("SC = dict(Z0)")
EVAL_DEFS = find("def calibrate(items, scorer, arl0):")
AUCS = find("def resp_auc(items, m):")

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# R1c — 판정기 크기가 재현율 상한의 원인인가

## 왜 하나

경보 규칙을 바꾸거나(R2, 52개 규칙) 학습 목표를 바꿔도(R1b, 이진 분류 = 다중 사례) 엄격 재현율은 ARL₀=100에서 약 0.50에서 막혔다.
사례 검토로 이 상한의 일부가 RAGTruth 정답 누락이라는 것을 확인했지만, 이른 경보의 절반 이상(53%)은 판정기의 진짜 오경보였다.
남은 질문은 **판정기가 작아서(MiniLM 6층) 응답 안의 환각 시작 전·후를 못 가르는가**이다.

## 무엇을 하나

학습 방식은 R1b의 **V0b**(문장 단위 정답, 모든 판정 지점, 지점별 이진 분류)로 고정하고 **백본만** 바꾼다. 데이터·근거 검색·판정 지점·평가는 R1b와 같다.

| 이름 | 백본 | 크기 | 시작점 | 시드 |
|---|---|---|---|---|
| **mini** (대조군) | `cross-encoder/nli-MiniLM2-L6-H768` | 약 0.08B | NLI 학습 모델 (student_vark와 같은 출발점, 단 RAGTruth 사전 미세조정 없이 한 번에) | 42·7·2026 |
| **base** | `cross-encoder/nli-deberta-v3-base` | 약 0.2B | NLI 학습 모델 | 42·7·2026 |
| **large** | `lytang/MiniCheck-DeBERTa-v3-Large` | 약 0.4B | 근거 기반 사실 확인(MiniCheck) 학습 모델 — 3주차 기준 판정기 계열 | 42 (비용 때문에 1개) |

- 출력층은 1개 점수로 새로 초기화한다. 학습률 mini·base 2e-5, large 1e-5. 1에폭.
- 체크포인트 선택은 R1b와 같다(cal-sub 300응답, 4토큰 지연 ≤ 9 우선, 그중 재현율 최대).
- 비교용으로 R1b의 V0b·V1b(MiniLM, student_vark에서 시작) 점수를 함께 불러온다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정)

- **판정 방식 선택**: base × {문장 끝, 4, 2, 2+EWMA} 중 cal-L 시드 평균 지연 ≤ 9토큰에서 cal-L 시드 평균 재현율 최대(없으면 재현율 최대). mini·large도 같은 판정 방식으로 비교.
- **R1c 주 기준(용량 효과)**: 선택된 판정 방식에서 dev-L ARL₀=100 **base 시드 평균 재현율 − mini 시드 평균 ≥ +0.03**.
- **R1c 보조(목표 달성)**: base가 R1 주 기준(시드 평균 재현율 ≥ 0.531, 지연 ≤ 9토큰, 시드 3개 중 2개 이상 B 문장 끝 대비 부트스트랩 하한 > 0)을 충족하는가.
- **보고(판정 없음)**: large(시드 1개)와 mini 시드 42의 차이, 완성 문장·응답 AUROC, 서비스 10% 재현율, 과제별, **판정 1회 지연**(현재 GPU, 배치 1, p50/p95). mini 대조군과 R1b V0b의 차이(RAGTruth 사전 미세조정의 효과).
- 해석 규칙: 주 기준이 미달이면 "이 학습 방식에서 0.2B까지 키워도 상한은 그대로 → 상한은 판정기 크기보다 데이터·과제 정의 쪽"으로 기록한다.

> 런타임: **A100 권장**(L4 가능, T4는 DeBERTa가 fp16에서 불안정할 수 있어 비권장). 학습 7회: mini 약 10분 × 3, base 약 25~35분 × 3, large 약 60~90분 × 1.
> 모델·점수는 드라이브 `selfproject/r1c/`에 저장되고 끝난 것은 건너뛴다. `RUN_LARGE = False`로 두면 large를 건너뛴다.
> 드라이브에 `student_vark`(근거 검색·토크나이저와 무관하지만 환경 점검용), R0 결과(`r0/`)가 있어야 한다.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터 (R1b와 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
md("## 3. 학습 판정 지점 — 문장 단위 정답 (R1b와 같음)")
code(LABELS)
md("## 4. 근거 검색 (R1b와 같음)")
code(RETR)
md("## 5. 평가 함수 (R1b와 같음)")
code(SERIES)
code(EVAL_DEFS)
code(AUCS)

md(r"""## 6. 학습 — 백본별 공통 루프 (지점별 이진 분류, 작은 배치로 나눠 기울기 누적)""")
code(r"""
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F, shutil

BACKBONES = {'mini': ('cross-encoder/nli-MiniLM2-L6-H768', 2e-5),
             'base': ('cross-encoder/nli-deberta-v3-base', 2e-5),
             'large': ('lytang/MiniCheck-DeBERTa-v3-Large', 1e-5)}
BATCH_SEQ, MICRO, N_EVALS = 64, 16, 4
BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
if not BF16: print('⚠️ bf16 미지원 GPU — DeBERTa는 fp16에서 불안정할 수 있음 (A100/L4 권장)')

def enc_ids(tok, idxs):
    return tok([ctx_of(P_KEY[g], P_IDX[g]) for g in idxs], [P_TXT[g] for g in idxs], truncation='only_first',
               max_length=MAX_LEN, padding=True, return_tensors='pt').to(DEVICE)

@torch.no_grad()
def score_queries(tok, model, qs, bs=128):
    model.eval(); out = np.zeros(len(qs), dtype=np.float32)
    order = sorted(range(len(qs)), key=lambda i: len(Q_TEXTS[qs[i]]))
    for i in range(0, len(order), bs):
        b = [qs[j] for j in order[i:i + bs]]
        enc = tok([ctx_of(Q_KEYS[q], EQ_IDX[q]) for q in b], [Q_TEXTS[q] for q in b], truncation='only_first',
                  max_length=MAX_LEN, padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
            out[order[i:i + bs]] = model(**enc).logits[:, 0].float().cpu().numpy()
    model.train()
    return out

CALSUB_Q = sorted({q for x in CALSUB for _, q in x['pts2']})
def calsub_metric(tok, model):
    v = score_queries(tok, model, CALSUB_Q)
    SC['_cur'] = np.zeros(len(Q_KEYS), dtype=np.float32); SC['_cur'][CALSUB_Q] = v
    r4 = evaluate(CALSUB, '_cur|4', calibrate(CALSUB, '_cur|4', 100))
    r2 = evaluate(CALSUB, '_cur|2', calibrate(CALSUB, '_cur|2', 100))
    d4 = r4['delay_tok_median']
    return float((r4['recall'] + r2['recall']) / 2), float(d4) if d4 == d4 else 1e9

Y_ALL = np.isin(P_CAT, ['win', 'post']).astype(np.float32)          # V0b: 문장 단위 시작 이후 = 1

def load_init(name):
    path, _ = BACKBONES[name]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, ignore_mismatched_sizes=True).to(DEVICE)
    return tok, model

def train_backbone(name, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok, model = load_init(name)
    lr = BACKBONES[name][1]
    order = list(range(len(P_TXT))); random.Random(seed).shuffle(order)
    n_steps = max(1, len(order) // BATCH_SEQ)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda' and not BF16))
    best, d0 = calsub_metric(tok, model)
    best_key, best_state, hist = (d0 <= 9, best), copy.deepcopy(model.state_dict()), [(0, best, d0)]
    print(f'  [{name} s{seed}] 시작 cal-sub 재현율 {best:.3f}, 지연 {d0} | {n_steps}단계, 학습률 {lr}')
    eval_at = {int(n_steps * (i + 1) / N_EVALS) for i in range(N_EVALS)}
    model.train(); t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        batch = order[(step - 1) * BATCH_SEQ: step * BATCH_SEQ]
        opt.zero_grad(); tot = 0.0
        for k in range(0, len(batch), MICRO):
            mb = batch[k:k + MICRO]
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
                z = model(**enc_ids(tok, mb)).logits[:, 0].float()
            loss = F.binary_cross_entropy_with_logits(z, torch.tensor(Y_ALL[mb], device=DEVICE), reduction='sum') / len(batch)
            scaler.scale(loss).backward(); tot += loss.item()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if step in eval_at:
            va, dv = calsub_metric(tok, model); hist.append((step, va, dv))
            if (dv <= 9, va) > best_key: best_key, best, best_state = (dv <= 9, va), va, copy.deepcopy(model.state_dict())
            print(f'  [{name} s{seed}] 단계 {step}/{n_steps} | 손실 {tot:.3f} | cal-sub 재현율 {va:.3f}, 지연 {dv} | {(time.perf_counter()-t0)/60:.1f}분')
    model.load_state_dict(best_state)
    return tok, model, hist

def run(name, seed):
    out = os.path.join(R1_DIR, f'{name}_s{seed}')
    sp = os.path.join(out, 'scores.npy')
    if not os.path.exists(sp):
        has_w = any(os.path.exists(os.path.join(out, f)) for f in ['model.safetensors', 'pytorch_model.bin'])
        if os.path.exists(os.path.join(out, 'config.json')) and not has_w:
            print(f'  [{name} s{seed}] 저장이 끊긴 폴더 → 지우고 다시 학습'); shutil.rmtree(out)
        if has_w:
            tok = AutoTokenizer.from_pretrained(out); model = AutoModelForSequenceClassification.from_pretrained(out).to(DEVICE)
        else:
            tok, model, hist = train_backbone(name, seed)
            os.makedirs(out, exist_ok=True); tok.save_pretrained(out); model.save_pretrained(out)
            pd.DataFrame(hist, columns=['step', 'calsub_recall', 'calsub_delay4']).to_csv(os.path.join(out, 'hist.csv'), index=False)
        t0 = time.perf_counter()
        v = score_queries(tok, model, EVAL_Q)
        full = np.full(len(Q_KEYS), np.nan, dtype=np.float32); full[EVAL_Q] = v
        np.save(sp, full)
        print(f'  [{name} s{seed}] 평가 질의 채점 {time.perf_counter()-t0:.0f}초')
        del model; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    SC[f'{name}_s{seed}'] = np.load(sp)
    print(f'[{name} s{seed}] 준비 완료')
""")

md(r"""## 7. 백본 확인 — 이름이 맞는지, 크기와 판정 1회 지연 (학습 전, 현재 GPU)""")
code(r"""
LAT = []
qs = EVAL_Q[:300]
for name, (path, lr) in BACKBONES.items():
    try:
        tok, model = load_init(name)
    except Exception as e:
        print(f'{name}: 불러오기 실패 — {type(e).__name__}: {str(e)[:150]}'); continue
    model.eval(); n_par = sum(p.numel() for p in model.parameters()) / 1e6
    ts = []
    with torch.no_grad():
        for i, q in enumerate(qs[:120]):
            enc = tok([ctx_of(Q_KEYS[q], EQ_IDX[q])], [Q_TEXTS[q]], truncation='only_first', max_length=MAX_LEN,
                      return_tensors='pt').to(DEVICE)
            if DEVICE == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
                model(**enc)
            if DEVICE == 'cuda': torch.cuda.synchronize()
            if i >= 20: ts.append((time.perf_counter() - t0) * 1000)
    LAT.append({'backbone': name, 'path': path, 'params_M': round(n_par), 'ms_p50': np.percentile(ts, 50), 'ms_p95': np.percentile(ts, 95),
                'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'})
    del model; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
lat_df = pd.DataFrame(LAT)
print(lat_df.round(1).to_string(index=False))
""")

md("## 8. 학습 — mini·base × 시드 3개, large × 시드 42 (끝난 것은 건너뜀)")
code(r"""
SEEDS = [42, 7, 2026]
RUN_LARGE = True
for seed in SEEDS:
    for name in ['mini', 'base']:
        run(name, seed)
if RUN_LARGE:
    run('large', 42)
# 비교용: R1b(MiniLM, student_vark에서 시작) 점수
for v in ['V0b', 'V1b']:
    for s in SEEDS:
        p = os.path.join(R1B_DIR, f'{v}_s{s}', 'scores.npy')
        if os.path.exists(p): SC[f'R1b{v}_s{s}'] = np.load(p)
print('R1b 비교 점수:', [k for k in SC if k.startswith('R1b')])
""")

md(r"""## 9. 판정 방식 선택 (cal-L만, base 시드 평균)""")
code(r"""
KINDS = ['end', '4', '2', 'ew2']
sel = []
for name in ['mini', 'base']:
    for k in KINDS:
        rs = [evaluate(CAL, f'{name}_s{s}|{k}', calibrate(CAL, f'{name}_s{s}|{k}', 100)) for s in SEEDS]
        sel.append({'backbone': name, 'kind': k, 'cal_recall': np.mean([r['recall'] for r in rs]),
                    'cal_delay': np.mean([r['delay_tok_median'] for r in rs])})
sel_df = pd.DataFrame(sel)
print(sel_df.round(3).to_string(index=False))
elig = sel_df[(sel_df.backbone == 'base') & (sel_df.cal_delay <= 9)].sort_values('cal_recall', ascending=False)
SEL_FALLBACK = len(elig) == 0
if SEL_FALLBACK:
    print('⚠️ cal-L 지연 ≤ 9토큰인 판정 방식 없음 → 지연 조건 없이 재현율 최대')
    elig = sel_df[sel_df.backbone == 'base'].sort_values('cal_recall', ascending=False)
CHOSEN_K = elig.iloc[0].kind
print(f'\n선택된 판정 방식: {CHOSEN_K}')
""")

md(r"""## 10. dev-L 평가와 사전 기준 판정""")
code(r"""
rows = []
models = ['B', 'E2'] + [f'{n}_s{s}' for n in ['mini', 'base', 'large'] for s in SEEDS if f'{n}_s{s}' in SC] \
         + [f'R1b{v}_s{s}' for v in ['V0b', 'V1b'] for s in SEEDS if f'R1b{v}_s{s}' in SC]
for m in models:
    ra, sa = resp_auc(DEV, m), sent_auc(DEV, m)
    for k in KINDS:
        sc = f'{m}|{k}'
        r = evaluate(DEV, sc, calibrate(CAL, sc, 100))
        r10 = evaluate(DEV, sc, calibrate_flag(CAL, sc, 0.10))
        rows.append({'model': m, 'kind': k, 'recall': r['recall'], 'delay_med': r['delay_tok_median'],
                     'pre_onset': r['pre_onset_alarm_rate'], 'realized_ARL0': r['realized_ARL0_dev'],
                     'svc_recall': r10['recall'], 'svc_delay': r10['delay_tok_median'], 'resp_auc': ra, 'sent_auc': sa})
res = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(res[res.kind == CHOSEN_K].round(3).to_string(index=False))

def smean(n, k, col='recall', seeds=SEEDS):
    v = [res[(res.model == f'{n}_s{s}') & (res.kind == k)][col].iloc[0] for s in seeds if f'{n}_s{s}' in SC]
    return float(np.mean(v)) if v else np.nan

base_r, mini_r, base_d = smean('base', CHOSEN_K), smean('mini', CHOSEN_K), smean('base', CHOSEN_K, 'delay_med')
cap_gain = base_r - mini_r
R1C = cap_gain >= 0.03
bts = [boot(f'base_s{s}|{CHOSEN_K}', 'B|end') for s in SEEDS]
n_sig = sum(b['lo'] > 0 for b in bts)
R1C_GOAL = base_r >= 0.531 and base_d <= 9 and n_sig >= 2
print(f"\n=== R1c 사전 기준 (판정 방식 {CHOSEN_K}) ===")
print(f"용량 효과: base {base_r:.3f} − mini {mini_r:.3f} = {cap_gain:+.3f} (≥ +0.03) → {'충족' if R1C else '미달'}")
for b in bts:
    print(f"   {b['A']}: B 문장 끝 대비 {b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
print(f"목표 달성(보조): 재현율 {base_r:.3f} ≥ 0.531, 지연 {base_d:.1f} ≤ 9, 유의 시드 {n_sig}/3 ≥ 2 → {'충족' if R1C_GOAL else '미달'}")
if 'large_s42' in SC:
    lg = res[(res.model == 'large_s42') & (res.kind == CHOSEN_K)].iloc[0]
    mi = res[(res.model == 'mini_s42') & (res.kind == CHOSEN_K)].iloc[0]
    print(f"(보고) large s42 {lg.recall:.3f} / 지연 {lg.delay_med} vs mini s42 {mi.recall:.3f} → {lg.recall - mi.recall:+.3f}")
    LB = boot(f'large_s42|{CHOSEN_K}', 'B|end')
    print(f"(보고) large s42: B 문장 끝 대비 {LB['diff']:+.3f} [{LB['lo']:+.3f}, {LB['hi']:+.3f}]")
r1b_v0 = smean('R1bV0b', CHOSEN_K)
print(f"(보고) RAGTruth 사전 미세조정 효과: R1b V0b {r1b_v0:.3f} − mini {mini_r:.3f} = {r1b_v0 - mini_r:+.3f}")

agg = []
for n in ['mini', 'base', 'large', 'R1bV0b', 'R1bV1b']:
    for k in KINDS:
        seeds = [42] if n == 'large' else SEEDS
        if any(f'{n}_s{s}' in SC for s in seeds):
            agg.append({'model': n, 'kind': k, 'n_seeds': sum(f'{n}_s{s}' in SC for s in seeds),
                        **{c: smean(n, k, c, seeds) for c in ['recall', 'delay_med', 'pre_onset', 'svc_recall', 'svc_delay', 'resp_auc', 'sent_auc']}})
for m, k in [('B', 'end'), ('B', '4'), ('E2', '4')]:
    r = res[(res.model == m) & (res.kind == k)].iloc[0]
    agg.append({'model': m, 'kind': k, 'n_seeds': 1, **{c: r[c] for c in ['recall', 'delay_med', 'pre_onset', 'svc_recall', 'svc_delay', 'resp_auc', 'sent_auc']}})
agg_df = pd.DataFrame(agg)
print('\n시드 평균 비교 (dev-L)'); print(agg_df[agg_df.kind.isin([CHOSEN_K, 'end'])].round(3).to_string(index=False))
""")

md("## 11. 과제별 (보고)")
code(r"""
tr = []
for tt in TASKS:
    its = [x for x in DEV if x['task_type'] == tt]
    row = {'task': tt, 'n_hall': sum(x['onset'] is not None for x in its)}
    for sc in ['B|end', 'B|4'] + [f'{n}_s42|{CHOSEN_K}' for n in ['mini', 'base', 'large'] if f'{n}_s42' in SC]:
        row[sc] = evaluate(its, sc, calibrate(CAL, sc, 100))['recall']
    tr.append(row)
task_df = pd.DataFrame(tr)
print(task_df.round(3).to_string(index=False))
""")

md("## 12. 저장")
code(r"""
summ = {'chosen_kind': CHOSEN_K, 'selection_fallback': bool(SEL_FALLBACK), 'base_seed_mean': base_r, 'mini_seed_mean': mini_r,
        'capacity_gain': cap_gain, 'R1c': bool(R1C), 'base_delay': base_d, 'n_sig_vs_Bend': int(n_sig), 'R1c_goal': bool(R1C_GOAL),
        'boot': bts, 'latency': LAT}
json.dump(summ, open('r1c_summary.json', 'w'), default=float, ensure_ascii=False, indent=1)
hists = []
for n in ['mini', 'base', 'large']:
    for s in SEEDS:
        p = os.path.join(R1_DIR, f'{n}_s{s}', 'hist.csv')
        if os.path.exists(p):
            h = pd.read_csv(p); h['run'] = f'{n}_s{s}'; hists.append(h)
out = {'r1c_devL.csv': res, 'r1c_select_cal.csv': sel_df, 'r1c_seedmean.csv': agg_df, 'r1c_task.csv': task_df,
       'r1c_latency.csv': lat_df, 'r1c_hist.csv': pd.concat(hists) if hists else pd.DataFrame()}
for f, d in out.items():
    d.to_csv(f, index=False); d.to_csv(os.path.join(R1_DIR, f), index=False)
shutil.copy('r1c_summary.json', os.path.join(R1_DIR, 'r1c_summary.json'))
print('저장 완료: 드라이브 selfproject/r1c/')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'r1c_judge_capacity.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells), '->', out)
