"""R1b 노트북 생성: R1의 라벨 설계 결함(응답 단위 정답 ↔ 문장 단위 입력)을 고친 재실험.
build_r1.py의 코드를 그대로 가져와, 정답 정의·학습 단위·체크포인트 선택·실행 단계만 바꾼다."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'build_r1.py'), encoding='utf-8').read()

def sub(old, new, count=1):
    global src
    assert src.count(old) >= 1, old[:80]
    src = src.replace(old, new, count)

# 0) 소개 마크다운 교체
i = src.index('md(r"""# R1 —'); j = src.index('md("## 1. 환경 준비")')
src = src[:i] + 'md(r"""' + r"""# R1b — 문장 단위 정답으로 고친 발생 시점 인식 학습

## 왜 하나

R1(10/13)은 재현율 0.541로 올랐지만 지연이 27토큰이었고, 원인은 **라벨 설계 결함**이었다.
정답을 응답 단위로 정해(응답에서 환각이 시작된 뒤의 모든 지점 = 양성), 환각 문장 **뒤의 멀쩡한 문장들**까지 양성으로 학습했다.
판정기 입력은 지금 문장의 앞부분뿐이라, 판정기는 "환각이 시작되는 지점"이 아니라 "수상한 응답을 뒤에서 늦게" 잡는 법을 배웠다
(완성 문장 AUROC 0.857 → 0.78, Data2txt만 상승). 다만 같은 결함 라벨 위에서도 다중 사례 목표(V1)가 이진 분류(V0)보다 +0.061 나았다.

## 무엇을 바꾸나

1. **정답을 문장 단위로** (E2와 같은 정의): 각 문장 안에서 환각 구간과 처음 겹치는 토큰이 그 문장의 시작점.
   - 환각이 있는 문장: 시작 전 = 음성, 시작 후 10토큰 = 창, 그 뒤 = 이후
   - 환각이 없는 문장(멀쩡한 응답이든, 환각 문장의 앞이든 뒤든) = 음성
2. **다중 사례 단위를 문장마다**: V1b는 환각 문장마다 "창 안에서 적어도 하나"를 학습한다(R1은 응답마다 하나).
3. **체크포인트 선택에 지연 조건**: cal-sub에서 4토큰 간격 지연 중앙값 ≤ 9인 체크포인트를 우선, 그중 재현율 최대(없으면 재현율 최대).
4. 변형은 **V0b(지점별 이진 분류, 대조군)와 V1b(창 안 적어도 하나)** 두 가지만, 시드 42·7·2026 모두 학습(6회).
   판정 방식은 학습이 끝난 뒤 cal-L에서 V1b 시드 평균으로 고른다.

## 사전 정의 기준 (결과 보기 전, SSOT §6에 고정 — R1과 같음)

- **R1b-0 정답 점검(설계 결함 재발 방지)**: V0b의 dev-L 완성 문장 AUROC 시드 평균 ≥ 0.84 (E2·B 수준 0.86 근처). 미달이면 라벨·학습에 다시 문제가 있는 것으로 보고 이후 수치를 해석하지 않는다.
- **선택 규칙**: V1b × {문장 끝, 4, 2, 2+EWMA} 중 cal-L 시드 평균 지연 ≤ 9토큰에서 cal-L 시드 평균 재현율 최대. 없으면 지연 조건 없이 재현율 최대(이 경우 주 기준은 지연에서 미달 가능).
- **R1b 주 기준**: dev-L ARL₀=100에서 시드 평균 재현율 ≥ 0.531, 시드 평균 지연 중앙값 ≤ 9토큰, 시드 3개 중 2개 이상에서 B 문장 끝 대비 쌍체 부트스트랩 하한 > 0.
- **목표 효과**: 같은 판정 방식에서 V1b 시드 평균 − V0b 시드 평균 ≥ +0.03.
- **보고(판정 없음)**: 응답 AUROC, 서비스 10% 재현율·지연, 완성 문장 AUROC, 과제별, R1(결함 라벨) 모델과의 비교.

> 런타임: **GPU 필수, A100 또는 L4 권장**. 학습 6회(각 A100 약 10~15분). 모델은 드라이브 `selfproject/r1b/`에 저장되고 끝난 것은 건너뛴다.
> 드라이브에 `student_vark`, R0 결과(`r0/r0_items.pkl`, `r0/r0_scores.npz`)가 있어야 한다. R1 점수(`r1/*/scores.npy`)가 있으면 비교용으로 함께 불러온다.
""" + '""")\n\n' + src[j:]

# 1) 저장 위치
sub("R1_DIR = os.path.join(OUT_DIR, 'r1')", "R1_DIR = os.path.join(OUT_DIR, 'r1b')\\n\"\n    \"R1_OLD = os.path.join(OUT_DIR, 'r1')")

# 2) 정답 정의: 문장 단위
i = src.index('md(r"""## 3. 학습 판정 지점'); j = src.index('md(r"""## 4. 근거 검색')
src = src[:i] + 'md(r"""' + r"""## 3. 학습 판정 지점 — 정답은 문장 단위 (E2와 같은 정의)

- `neg`: 환각이 없는 문장의 지점 (멀쩡한 응답, 환각 문장의 앞이든 뒤든)
- `pre`: 환각이 있는 문장에서 시작 전 / `win`: 시작 후 10토큰 안 / `post`: 그 이후
- 다중 사례 단위(V1b): 환각이 있는 **문장마다** 창 하나""" + '""")\n' + 'code(POINTS_DEF + r"""' + r"""

WIN_TOK = 10
train_items = to_items(train)
by_tr = {str(r['id']): r for r in train}
TR, P_KEY, P_TXT, P_CAT = [], [], [], []
for x in train_items:
    text = x['text']
    spans = [(l['start'], l['end']) for l in (by_tr[x['id']].get('labels') or [])]
    cs = [a + (len(text[a:b]) - len(text[a:b].lstrip())) if text[a:b].strip() else a for a, b in zip(x['ts'], x['te'])]
    def s_onset(sn):                      # E2와 같다: 이 문장 안에서 환각 구간과 처음 겹치는 토큰
        for t in range(sn['t0'], sn['t1'] + 1):
            if any(min(x['te'][t], b) > max(cs[t], a) for a, b in spans): return t
        return None
    sid = {sn['t0']: i for i, sn in enumerate(x['sents'])}
    son = [s_onset(sn) for sn in x['sents']]
    groups = {}
    for t, txt in points(x, 2):
        if not txt: continue
        si = sid[x['sent_of'][t]]; o = son[si]
        if o is None: cat = 'neg'
        elif t < o: cat = 'pre'
        elif t <= o + WIN_TOK: cat = 'win'
        else: cat = 'post'
        g = len(P_TXT); P_KEY.append(x['ref_key']); P_TXT.append(txt); P_CAT.append(cat)
        if cat in ('win', 'post'): groups.setdefault(si, []).append(g)
    gl = []
    for si, lst in groups.items():
        win = [g for g in lst if P_CAT[g] == 'win']
        if not win:                       # 창 안에 지점이 없으면 시작 이후 첫 지점을 창으로
            P_CAT[lst[0]] = 'win'; win = [lst[0]]
        gl.append(win)
    TR.append({'groups': gl})
P_CAT = np.array(P_CAT)
print(f'학습 응답 {len(TR)} | 판정 지점 {len(P_TXT):,} | 환각 문장(다중 사례 단위) {sum(len(r["groups"]) for r in TR):,} |',
      pd.Series(P_CAT).value_counts().to_dict())
assert sum(len(r['groups']) for r in TR) > 0
""" + '""")\n\n' + src[j:]

# 3) 학습 단위
i = src.index("def make_units(variant, rng):"); j = src.index("@torch.no_grad()\ndef score_ids")
src = src[:i] + r"""def make_units(variant, rng):
    units = []
    for g in range(len(P_TXT)):
        c = P_CAT[g]
        if variant == 'V0b':
            units.append(('s', g, 1.0 if c in ('win', 'post') else 0.0, 1.0))
        elif c == 'post':
            units.append(('s', g, 1.0, POST_W))
        elif c in ('neg', 'pre'):
            units.append(('s', g, 0.0, 1.0))
    if variant == 'V1b':
        for r in TR:
            for win in r['groups']:
                units.append(('g', win))
    rng.shuffle(units)
    return units

""" + src[j:]
i = src.index("def group_loss(variant, ri, tok, model):"); j = src.index("def train_variant(variant, seed):")
src = src[:i] + "def group_loss(variant, win, tok, model):\n    return win, []\n\n" + src[j:]

# 4) 체크포인트 선택: 지연 조건 우선
sub("""    rec = [evaluate(CALSUB, f'_cur|{k}', calibrate(CALSUB, f'_cur|{k}', 100))['recall'] for k in ['4', '2']]
    return float(np.mean(rec))""",
"""    r4 = evaluate(CALSUB, '_cur|4', calibrate(CALSUB, '_cur|4', 100))
    r2 = evaluate(CALSUB, '_cur|2', calibrate(CALSUB, '_cur|2', 100))
    d4 = r4['delay_tok_median']
    return float((r4['recall'] + r2['recall']) / 2), float(d4) if d4 == d4 else 1e9""")
sub("""    best, best_state, hist = calsub_metric(tok, model), copy.deepcopy(model.state_dict()), []
    hist.append((0, best)); print(f'  [{variant} s{seed}] 시작 cal-sub {best:.3f} | 예상 {n_steps}단계')""",
"""    best, d0 = calsub_metric(tok, model)
    best_key, best_state, hist = (d0 <= 9, best), copy.deepcopy(model.state_dict()), [(0, best, d0)]
    print(f'  [{variant} s{seed}] 시작 cal-sub 재현율 {best:.3f}, 지연 {d0} | 예상 {n_steps}단계')""")
sub("""            va = calsub_metric(tok, model); hist.append((step, va))
            if va > best: best, best_state = va, copy.deepcopy(model.state_dict())
            print(f'  [{variant} s{seed}] 단계 {step} | 손실 {loss.item():.3f} | cal-sub {va:.3f} | {(time.perf_counter()-t0)/60:.1f}분')""",
"""            va, dv = calsub_metric(tok, model); hist.append((step, va, dv))
            if (dv <= 9, va) > best_key: best_key, best, best_state = (dv <= 9, va), va, copy.deepcopy(model.state_dict())
            print(f'  [{variant} s{seed}] 단계 {step} | 손실 {loss.item():.3f} | cal-sub 재현율 {va:.3f}, 지연 {dv} | {(time.perf_counter()-t0)/60:.1f}분')""")
sub("pd.DataFrame(hist, columns=['step', 'calsub_recall'])", "pd.DataFrame(hist, columns=['step', 'calsub_recall', 'calsub_delay4'])")

# 5) 실행 단계: 6회 학습 → 판정 방식 선택
i = src.index('md("## 7. 1단계'); j = src.index('md(r"""## 10. dev-L 평가')
src = src[:i] + r'''md("## 7. 학습 — V0b·V1b × 시드 42·7·2026 (끝난 것은 건너뜀)")
code(r"""
SEEDS = [42, 7, 2026]
for seed in SEEDS:
    for v in ['V1b', 'V0b']:
        run(v, seed)
# 비교용: R1(결함 라벨) 점수가 드라이브에 있으면 함께 불러온다
for v in ['V0', 'V1']:
    for s in SEEDS:
        p = os.path.join(R1_OLD, f'{v}_s{s}', 'scores.npy')
        if os.path.exists(p): SC[f'R1{v}_s{s}'] = np.load(p)
print('R1 비교 점수:', [k for k in SC if k.startswith('R1')])
""")

md(r"""## 8. R1b-0 정답 점검과 판정 방식 선택 (cal-L만 사용)""")
code(r"""
KINDS = ['end', '4', '2', 'ew2']
v0_sent = float(np.mean([sent_auc(DEV, f'V0b_s{s}') for s in SEEDS]))
R1B0 = v0_sent >= 0.84
print(f"R1b-0 정답 점검: V0b 완성 문장 AUROC 시드 평균 {v0_sent:.3f} (≥ 0.84) → {'충족' if R1B0 else '미달 — 이후 수치 해석 보류'}")

sel = []
for v in ['V0b', 'V1b']:
    for k in KINDS:
        rs = [evaluate(CAL, f'{v}_s{s}|{k}', calibrate(CAL, f'{v}_s{s}|{k}', 100)) for s in SEEDS]
        sel.append({'variant': v, 'kind': k, 'cal_recall': np.mean([r['recall'] for r in rs]),
                    'cal_delay': np.mean([r['delay_tok_median'] for r in rs]),
                    'cal_pre_onset': np.mean([r['pre_onset_alarm_rate'] for r in rs])})
sel_df = pd.DataFrame(sel)
print(sel_df.round(3).to_string(index=False))
elig = sel_df[(sel_df.variant == 'V1b') & (sel_df.cal_delay <= 9)].sort_values('cal_recall', ascending=False)
SEL_FALLBACK = len(elig) == 0
if SEL_FALLBACK:
    print('⚠️ cal-L 지연 ≤ 9토큰인 V1b 판정 방식 없음 → 지연 조건 없이 재현율 최대 (주 기준은 지연에서 미달 가능)')
    elig = sel_df[sel_df.variant == 'V1b'].sort_values('cal_recall', ascending=False)
CHOSEN_V, CHOSEN_K = 'V1b', elig.iloc[0].kind
print(f'\n선택: {CHOSEN_V}, 판정 방식 {CHOSEN_K}')
""")

''' + src[j:]

# 6) 평가·저장의 변형 이름
sub("[f'{v}_s{s}' for v in ['V0', 'V1', 'V2', 'V3'] for s in SEEDS if f'{v}_s{s}' in SC]",
    "[f'{v}_s{s}' for v in ['V0b', 'V1b', 'R1V0', 'R1V1'] for s in SEEDS if f'{v}_s{s}' in SC]")
sub("SEEDS = [42, 7, 2026]\nrows = []", "rows = []")
sub("gain = rec - seed_mean('V0', CHOSEN_K)", "gain = rec - seed_mean('V0b', CHOSEN_K)")
sub("R1 = c1 and c2 and c3", "R1 = c1 and c2 and c3 and R1B0")
sub("print(f\"\\n=== R1 사전 기준", "print(f\"\\n=== R1b 사전 기준")
sub("print('R1 주 기준:'", "print('R1b 주 기준 (R1b-0 포함):'")
sub("print(f\"목표 효과: V0 대비", "print(f\"목표 효과: V0b 대비")
sub("for v in ['V0', CHOSEN_V]:\n    for k in KINDS:\n        agg.append",
    "for v in ['V0b', CHOSEN_V] + [w for w in ['R1V0', 'R1V1'] if f'{w}_s42' in SC]:\n    for k in KINDS:\n        agg.append")
sub("summ = {'chosen_variant': CHOSEN_V,", "summ = {'R1b_0_v0b_sent_auc': v0_sent, 'R1b_0': bool(R1B0), 'chosen_variant': CHOSEN_V,")
sub("for v in ['V0', 'V1', 'V2', 'V3']:\n    for s in SEEDS:\n        p = os.path.join(R1_DIR",
    "for v in ['V0b', 'V1b']:\n    for s in SEEDS:\n        p = os.path.join(R1_DIR")
sub("out = os.path.join(NB_DIR, 'r1_onset_aware_training.ipynb')", "out = os.path.join(NB_DIR, 'r1b_sentence_local_labels.ipynb')")
src = src.replace("'r1_", "'r1b_").replace("r1_summary.json", "r1b_summary.json").replace("r1_results.zip", "r1b_results.zip")
src = src.replace("selfproject/r1/", "selfproject/r1b/")
src = src.replace('"""R1 노트북 생성', '"""(R1b가 불러씀) R1 노트북 생성', 1)
exec(compile(src, 'build_r1b_generated', 'exec'))
