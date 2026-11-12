"""K2a 노트북 생성: 최신 LLM 실제 응답 데이터 — 학습(train, Sonnet 정답)·보정(cal-M, Opus)·평가(dev-M, Opus).
M1 노트북과 같은 생성 흐름(QA = 컬렉션 RAG, Summary·Data2txt = 원 프롬프트)·정답 지시문·판정 코드를 그대로 쓰고,
cal-M·dev-M은 기존 조합(F1 base + I2)으로 판정 지점 점수까지 저장한다(K2b가 F1 점수만 새 판정기로 다시 낸다).
사전 기준은 SSOT §6 K2. RAGTruth test 분할 근거는 assert로 막는다."""
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
ENV = sub(ENV, "print('device', GPU, '| 조합 상수(I3)', COMBO)",
          "print('device', GPU, '| 조합 상수(I3)', COMBO)\n"
          "# 드라이브 쓰기 점검: 용량이 차면 결과가 조용히 저장되지 않는다(R4c·F1). API 결과를 잃으면 돈을 다시 써야 하므로 시작 전에 막는다\n"
          "_t = os.path.join(K2_DIR, '_write_test.bin')\n"
          "with open(_t, 'wb') as f: f.write(os.urandom(50 * 2**20))\n"
          "assert os.path.getsize(_t) == 50 * 2**20, '드라이브 쓰기 실패 — 휴지통 비우기·용량 확인 후 다시 실행'\n"
          "os.remove(_t); print('드라이브 50MB 쓰기 정상')")
ENV = sub(ENV, "R0_DIR, M1_DIR = os.path.join(OUT, 'r0'), os.path.join(OUT, 'm1')\nos.makedirs(M1_DIR, exist_ok=True)",
          "R0_DIR, K2_DIR = os.path.join(OUT, 'r0'), os.path.join(OUT, 'k2')\nos.makedirs(K2_DIR, exist_ok=True)")

# ---- K2A_* 셀은 smoke_k2.py가 확인한다 ----
K2A_SOURCES = r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
split_of = {}
for r in responses: split_of.setdefault(r['source_id'], set()).add(r['split'])
SIDS = {'calM': sorted({x['ref_key'] for x in R0P['items']['calL']}), 'devM': sorted({x['ref_key'] for x in R0P['items']['devL']})}
HOLD = set(SIDS['calM']) | set(SIDS['devM'])
_pool = sorted(s for s, sp in split_of.items() if sp == {'train'} and s not in HOLD and s in sources)
N_TRAIN_SRC = 2000
SIDS['train'] = sorted(random.Random(2031).sample(_pool, min(N_TRAIN_SRC, len(_pool))))
assert not set(SIDS['calM']) & set(SIDS['devM']), 'cal-M·dev-M 근거가 겹침'
assert not set(SIDS['train']) & HOLD, '학습 근거에 cal-L·dev-L 근거가 섞임'
for sp, ss in SIDS.items():
    assert all(split_of[s] == {'train'} for s in ss), f'{sp}에 test 분할 근거가 섞임'
    print(f'{sp}: 근거 {len(ss)}개 |', {t: sum(sources[s]['task_type'] == t for s in ss) for t in ['QA', 'Summary', 'Data2txt']})
print(f'학습 근거 후보 {len(_pool)}개 중 {len(SIDS["train"])}개 (Random(2031))')
"""

K2A_PREP = r"""
# 묶음별 설정: 생성 개수(근거·모델마다), 정답 모델, 판정 지점 점수 저장 여부
SPLITS = {'train': {'n': {'haiku': 2, 'sonnet': 1}, 'labeler': 'claude-sonnet-5-5', 'score': False},     # 학습 정답은 평가자(Opus)와 다른 모델 (R4b 원칙)
          # 학습 묶음: 파일럿(11/12) 결과 Sonnet 정답의 환각 문장이 응답당 0.10개로 적어, 환각이 많은 Haiku 응답을 근거당 1개 더 (SSOT §7)
          'calM': {'n': 2, 'labeler': 'claude-opus-5-5', 'score': True},
          'devM': {'n': 2, 'labeler': 'claude-opus-5-5', 'score': True}}

def label_params_k2(c, text, model):
    p = label_params(c, text); p['model'] = model
    return p

def expand(base, n):
    # 같은 요청을 n번 (배치 기본 설정이라 응답이 서로 다름). id 끝에 -0, -1. n이 dict면 생성 모델마다 개수
    nn = lambda c: n[c['gen']] if isinstance(n, dict) else n
    return {f'{cid}-{k}': dict(c) for cid, c in base.items() for k in range(nn(c))}

def prepare_split(split):
    cp = os.path.join(K2_DIR, f'cases_{split}.json')
    cases = load_json(cp)                      # 이미 만든 요청이 있으면 그대로 (검색 결과 = LLM이 본 근거가 바뀌지 않게)
    if cases is None:
        store = SQLiteStore(f'/content/k2_{split}.sqlite3', jF.embedder, CHUNK_W, OVERLAP)
        coll = f'k2-{split}'
        if not store.exists(coll):
            print(f'컬렉션 {coll}: QA 근거 {build_collection(store, coll, SIDS[split], sources)}개 등록')
        cases = expand(make_cases(split, SIDS[split], sources, store, coll), SPLITS[split]['n'])
        save_json(cases, cp)
    else:                                      # 설정이 바뀌어 요청이 늘었으면 같은 요청(-0)을 복사해 추가 (검색 결과 그대로)
        base = {cid[:-2]: c for cid, c in cases.items() if cid.endswith('-0')}
        add = {k: v for k, v in expand(base, SPLITS[split]['n']).items() if k not in cases}
        if add:
            cases.update(add); save_json(cases, cp); print(f'{split}: 요청 {len(add)}개 추가')
    qa = [c for c in cases.values() if c['flow'] == 'collection' and c['gen'] == 'haiku']
    print(f"{split}: 생성 요청 {len(cases)}개 | QA 검색에 원래 근거가 들어간 비율 {np.mean([c['gold_in'] for c in qa]) if qa else float('nan'):.3f}")
    return cases
"""

K2A_RUN = r"""
COST_CAP = 50.0          # 이 노트북 전체 API 사용 상한($, 배치 단가·실제 사용 토큰 기준). 넘을 것 같으면 제출 전에 멈춘다

def spent():
    tot = 0.0
    for f in os.listdir(K2_DIR):
        if f.startswith(('gen_', 'label_')) and f.endswith('.json') and '_batches' not in f:
            tot += cost(load_json(os.path.join(K2_DIR, f)) or {})
    return tot

def guard(n_new, per_req):
    # 새로 제출할 요청 n_new개를 응답당 per_req($)로 쳤을 때 상한을 넘으면 제출하지 않고 멈춘다
    s, add = spent(), n_new * per_req
    if s + add > COST_CAP:
        raise SystemExit(f'예산 상한 ${COST_CAP:.0f} 초과 예상: 지금까지 ${s:.2f} + 이번 약 ${add:.2f} — 제출하지 않고 멈춤')

def run_split(split, ids=None, per_gen=0.003, per_label=None):
    cases, cfg = CASES[split], SPLITS[split]
    if ids is not None: cases = {c: cases[c] for c in ids}
    per_label = per_label if per_label is not None else (0.012 if cfg['labeler'] == 'claude-opus-5-5' else 0.006)
    _done = load_json(os.path.join(K2_DIR, f'gen_{split}.json')) or {}
    guard(sum(c not in _done for c in cases), per_gen)
    gens = run_batch({c: gen_params(x) for c, x in cases.items()}, os.path.join(K2_DIR, f'gen_{split}.json'))
    valid = [c for c in cases if gen_status(gens.get(c)) == 'valid']
    print(f'{split} 생성: 유효 {len(valid)} | 거절 {sum(gen_status(gens.get(c)) == "refusal" for c in cases)} | '
          f'빈 응답 {sum(gen_status(gens.get(c)) == "empty" for c in cases)} | max_tokens {sum((gens.get(c) or {}).get("stop") == "max_tokens" for c in cases)}')
    _done = load_json(os.path.join(K2_DIR, f'label_{split}.json')) or {}
    guard(sum(c not in _done for c in valid), per_label)
    raw = run_batch({c: label_params_k2(cases[c], gens[c]['text'], cfg['labeler']) for c in valid}, os.path.join(K2_DIR, f'label_{split}.json'))
    labels, bad = {}, []
    for c in valid:
        if c not in raw: continue
        labs, ok = parse(raw[c]['text'], len(sentences(gens[c]['text'])))
        (labels.__setitem__(c, labs) if ok else bad.append(c))
    for c in bad[:100]:                                    # 파싱 실패는 바로 1회 재요청
        try:
            labs, ok = parse(text_of(A.messages.create(**label_params_k2(cases[c], gens[c]['text'], cfg['labeler']))), len(sentences(gens[c]['text'])))
            if ok: labels[c] = labs
        except Exception as e: print(f'  {c}: {type(e).__name__}'); break
    print(f'{split} 정답({cfg["labeler"]}): {len(labels)}/{len(valid)} 확보')
    scores = {}
    if cfg['score'] and ids is None:
        sp = os.path.join(K2_DIR, f'scores_{split}.json')
        scores = load_json(sp) or {}
        refs, t0 = {}, time.time()
        for k, c in enumerate([c for c in valid if c not in scores]):
            r = cases[c]['ref_judge']
            if r not in refs: refs = {r: (jF.prepare(r), jI.prepare(r))}
            scores[c] = [list(p) for p in score_response(jF, jI, refs[r], gens[c]['text'], FN)]
            if (k + 1) % 25 == 0:
                save_json(scores, sp); print(f'  판정 {len(scores)}/{len(valid)} ({time.time() - t0:.0f}s)', flush=True)
        save_json(scores, sp)
    return gens, labels, scores
"""

K2A_PILOT = r"""
def pilot_projection(cases_all, gens, raw, labels, counts):
    # 파일럿 응답의 실제 사용량으로 생성 모델별 단가·환각 문장 비율을 재고, 묶음별 비용과 학습 H 문장 수를 내다본다
    # counts: {묶음: {생성 모델: 응답 수}}
    gen_of = lambda c: cases_all['train'][c]['gen']
    per = {}
    for g in sorted({gen_of(c) for c in gens}):
        gi = [c for c in gens if gen_of(c) == g]; li = [c for c in raw if c in gens and gen_of(c) == g]; hi = [c for c in labels if gen_of(c) == g]
        per[g] = {'n': len(gi), 'gen': cost({c: gens[c] for c in gi}) / max(len(gi), 1),
                  'label': cost({c: raw[c] for c in li}) / max(len(li), 1),                     # Sonnet 정답 응답당
                  'H': sum(sum(v['label'] == 'H' for v in labels[c].values()) for c in hi) / max(len(hi), 1)}
    opus_ratio = 2.0                                                     # 배치 단가 Opus/Sonnet ($2/$10 vs $1/$5)
    proj = {sp: sum(n * (per[g]['gen'] + per[g]['label'] * (opus_ratio if SPLITS[sp]['labeler'] == 'claude-opus-5-5' else 1.0))
                    for g, n in cnt.items()) for sp, cnt in counts.items()}
    return {'per_gen': per, 'proj_cost': proj, 'proj_total': sum(proj.values()),
            'proj_train_H': sum(n * per[g]['H'] for g, n in counts['train'].items())}
"""

K2A_CHECK = r"""
def k2a_check(CASES, RUN):
    tr_gens, tr_labels, _ = RUN['train']
    valid_tr = [c for c in CASES['train'] if gen_status(tr_gens.get(c)) == 'valid']
    parse_ratio = len(tr_labels) / max(len(valid_tr), 1)
    n_H_train = sum(sum(v['label'] == 'H' for v in labs.values()) for labs in tr_labels.values())
    hal_tr = {}
    for c, labs in tr_labels.items():
        x = CASES['train'][c]; k = (x['gen'], x['task'])
        h = any(v['label'] == 'H' for v in labs.values())
        hal_tr.setdefault(k, [0, 0]); hal_tr[k][0] += h; hal_tr[k][1] += 1
    rows = {sp: response_rows(CASES[sp], *RUN[sp], THR_C, THR_F) for sp in ['calM', 'devM']}
    ok = {sp: d[d.status == 'ok'] for sp, d in rows.items()}
    n_cal_clean = int(ok['calM'].clean.astype(bool).sum())
    n_dev_hal = int((~ok['devM'].clean.astype(bool)).sum())
    k2_0 = bool(parse_ratio >= 0.90 and n_H_train >= 500 and n_cal_clean >= 400 and n_dev_hal >= 150)
    return {'train_valid': len(valid_tr), 'train_parse_ratio': parse_ratio, 'train_H_sent': int(n_H_train),
            'train_hal_resp': int(sum(v[0] for v in hal_tr.values())),
            'train_hal_by_gen_task': {f'{g}|{t}': v for (g, t), v in sorted(hal_tr.items())},
            'calM_clean': n_cal_clean, 'devM_hal': n_dev_hal, 'K2_0': k2_0}, rows
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# K2a — 최신 LLM 실제 응답 데이터 (학습·보정·평가)

## 왜 하나

M1에서 배포 판정기는 최신 Claude 응답의 환각을 **14%**밖에 잡지 못했다. K1에서 직접 모순을 합성해 학습해도 실제 모순은 그대로 놓쳤다(46문장 중 경보 1개).
합성 데이터가 실제 최신 LLM이 틀리는 방식과 다르기 때문으로 보고, 이번에는 **최신 LLM이 실제로 쓴 응답**에 문장 단위 정답을 붙여 경량 판정기를 학습한다(K2b).
또 지금 임계값은 RAGTruth(2023년 무렵 LLM)로 맞춘 것이라 최신 LLM 멀쩡한 응답에는 4~6%만 표시된다 → **최신 LLM 멀쩡한 응답 기준 임계값**도 만든다.

## 무엇을 하나 (M1과 같은 흐름·코드)

생성 = `claude-haiku-5-5`·`claude-sonnet-5-5`(배치, 기본 설정). QA는 컬렉션 RAG(top-4, 120단어 조각), Summary·Data2txt는 RAGTruth 원 프롬프트.

| 묶음 | 근거 | 생성 | 정답 | 판정 지점 점수 |
|---|---|---|---|---|
| **train** (학습) | RAGTruth train 분할 근거 중 cal-L·dev-L 근거를 뺀 것, 최대 2,000개(Random(2031)) | **Haiku 2개 + Sonnet 1개** (11/12 파일럿 뒤 변경) | `claude-sonnet-5-5` (평가자와 다른 모델) | 없음 |
| **calM** (임계값 B 보정) | cal-L 근거 | 모델마다 2개 | `claude-opus-5-5` | 기존 조합 |
| **devM** (주 평가) | dev-L 근거 | 모델마다 2개(M1·m1_fresh와 다른 새 응답) | `claude-opus-5-5` | 기존 조합 |

정답은 R4와 같은 지시문으로 문장마다 S/H/A. **멀쩡한 응답 = H 문장 없음**.

## 사전 정의 기준 (SSOT §6 K2, 실행 전 고정)

- **K2-0(유효성, 여기서 판정)**: train 정답 파싱 ≥ 90%, train H 문장 ≥ 500개, calM 멀쩡한 응답 ≥ 400개, devM 환각 응답 ≥ 150개 — 미달이면 보류
- K2-1(주)·K2-2·K2-3은 K2b에서 판정. 이 노트북에서 출력하는 기존 조합의 devM 수치는 **보고용**이다

## 실행 방법

1. 왼쪽 열쇠 아이콘(보안 비밀)에 `ANTHROPIC_API_KEY`를 넣고 노트북 접근을 켠다
2. 런타임 유형: **GPU**(L4·A100 권장, T4도 가능). calM·devM 판정 지점 점수를 낸다
3. **모두 실행**. 배치 처리는 보통 1~2시간, 점수는 1~2시간. 끊기면 다시 모두 실행 — 제출한 배치·받은 결과·점수는 드라이브 `k2/`에서 이어서 한다
4. 비용: 약 $40~45 (M1 실측 단가 기준 추정). **안전장치**: ① 시작 시 드라이브 쓰기 점검 ② 파일럿 150건으로 실제 단가를 재서 예상 총비용 > $50 또는 학습 환각이 부족하면 멈춤 ③ 묶음마다 제출 전에 누적 사용량 + 이번 묶음 예상이 $50을 넘으면 멈춤 ④ 제출한 배치 번호·받은 결과는 바로 드라이브에 저장돼 다시 실행해도 같은 요청을 두 번 내지 않음
5. 끝나면 `k2a_results.zip`(원문 없음)과 `k2a_raw_backup.zip`(API 원자료 백업, 맥에 보관만)이 내려받아진다 → 올려 주면 K2-0을 확인하고 K2b로 넘어간다
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 근거 문서 — train(학습), calM(cal-L), devM(dev-L)")
code(L3.L3_INPUTS + K2A_SOURCES)
md("## 3. 판정기·생성 요청 준비")
code(L3.L3_CORE + M1.M1_FLOW + M1.M1_LABEL + M1.M1_SCORE + r"""
jF, jI = make_judge(F_DIR, DEVICE), make_judge(I_DIR, DEVICE)
assert jF.mode == 'retrieval' and jI.mode == 'full', (jF.mode, jI.mode)
THR_C = COMBO['threshold']
THR_F = read_config(F_DIR)['thresholds']['8']['svc10_orig']
FN = judge_par if DEVICE == 'cuda' else judge_seq
print(f'기존 조합 임계값(A) {THR_C:.4f} | F1 base 단독 {THR_F:.4f}')
""" + K2A_PREP + r"""
CASES = {sp: prepare_split(sp) for sp in SPLITS}
""")
md("## 4. Claude API — 형식 확인, 배치 제출·결과 받기 (드라이브에서 이어서)")
code(M1.M1_API + r"""
# 형식 확인: 모델마다 1건 바로 생성 + 그 응답에 두 정답 모델 1건씩 (수 센트)
for g in GENS:
    cid = next(c for c, x in CASES['train'].items() if x['gen'] == g and x['flow'] == 'collection')
    r = rec(A.messages.create(**gen_params(CASES['train'][cid])))
    print(f"[{g}] stop={r['stop']} 입력 {r['in']} 출력 {r['out']} | {r['text'][:160]!r}")
    if r['text'].strip():
        for lm_name in ['claude-sonnet-5-5', 'claude-opus-5-5']:
            labs, ok = parse(text_of(A.messages.create(**label_params_k2(CASES['train'][cid], r['text'], lm_name))), len(sentences(r['text'])))
            print(f"   {lm_name} 정답 파싱 {'성공' if ok else '실패'} | {[v['label'] for v in labs.values()]}")
# M1 실측: 생성 $0.94 + Opus 정답 $4.01 / 536응답 → 응답당 생성 $0.00175, Opus 정답 $0.0075, Sonnet 정답 약 절반
est = {sp: len(c) * (0.00175 + (0.0075 if SPLITS[sp]['labeler'] == 'claude-opus-5-5' else 0.00375)) for sp, c in CASES.items()}
print('예상 비용(M1 실측 단가):', {k: round(v, 1) for k, v in est.items()}, f'합계 약 ${sum(est.values()):.0f}')
""")
md("""## 5. 파일럿 — 학습 묶음 300건(처음 150 + Haiku 추가분 150)으로 실제 비용·환각 수를 재고, 기준에 못 미치면 여기서 멈춤

전체를 제출하기 전에 학습 묶음에서 처음 파일럿 150건(Random(2033), 이미 받음)과 새로 추가한 Haiku 응답 150건(Random(2035))만 먼저 받아 **생성 모델별 응답당 실제 비용**과 **환각 문장 비율**을 잰다.
예상 총비용이 상한($50)을 넘거나, 예상 학습 H 문장이 K2-0 기준(500)의 1.2배(600)에 못 미치면 **나머지를 제출하지 않고 멈춘다**(이때까지 약 $1).
파일럿 결과는 버리지 않고 본 실행에 그대로 들어간다.""")
code(K2A_RUN + K2A_PILOT + r"""
# 파일럿 1(처음 150건, 이미 받은 것 그대로) + 파일럿 2(새로 추가한 Haiku 응답 150건)
PILOT1 = sorted(random.Random(2033).sample(sorted(c for c in CASES['train'] if c.endswith('-0')), 150))
PILOT2 = sorted(random.Random(2035).sample(sorted(c for c in CASES['train'] if c.endswith('-1')), 150))
PILOT = PILOT1 + PILOT2
pg, pl, _ = run_split('train', ids=PILOT, per_gen=0.01, per_label=0.01)
_gen, _raw = load_json(os.path.join(K2_DIR, 'gen_train.json')), load_json(os.path.join(K2_DIR, 'label_train.json'))
COUNTS = {sp: {g: sum(x['gen'] == g for x in c.values()) for g in GENS} for sp, c in CASES.items()}
PJ = pilot_projection(CASES, {c: _gen[c] for c in PILOT if c in _gen}, {c: _raw[c] for c in PILOT if c in _raw}, pl, COUNTS)
print('요청 수', COUNTS)
print(json.dumps(PJ, ensure_ascii=False, indent=1, default=float))
print(f"예상 총비용 ${PJ['proj_total']:.1f} (상한 ${COST_CAP:.0f}) | 예상 학습 H 문장 {PJ['proj_train_H']:.0f} (필요 ≥ 600)")
if PJ['proj_total'] > COST_CAP or PJ['proj_train_H'] < 600:
    raise SystemExit('파일럿 기준 미달 — 나머지는 제출하지 않음. 결과를 Claude Code에 알려 주세요')
_mx = lambda k: max(v[k] for v in PJ['per_gen'].values())         # 묶음 예산 확인은 비싼 쪽 단가로 (보수적)
PER = {'gen': _mx('gen') * 1.3, 'sonnet': _mx('label') * 1.3, 'opus': _mx('label') * 2.0 * 1.3}
""")
md("## 6. 생성 → 정답 → (calM·devM) 기존 조합 판정")
code(r"""
RUN = {}
for sp in ['calM', 'devM', 'train']:          # 평가·보정 먼저 (학습 묶음이 가장 큼). 묶음마다 제출 전에 예산 확인(파일럿 단가 × 1.3)
    RUN[sp] = run_split(sp, per_gen=PER['gen'], per_label=PER['opus' if SPLITS[sp]['labeler'] == 'claude-opus-5-5' else 'sonnet'])
""")
md("## 7. K2-0 판정과 저장")
code(M1.M1_METRICS + K2A_CHECK + r"""
CHK, ROWS = k2a_check(CASES, RUN)
print(json.dumps({k: v for k, v in CHK.items() if k != 'train_hal_by_gen_task'}, ensure_ascii=False, indent=1))
print('학습 응답 환각 비율(생성 모델|과제: 환각/정답 확보):', CHK['train_hal_by_gen_task'])
REP = {sp: report(d) for sp, d in ROWS.items()}
for sp in ['calM', 'devM']:
    r = REP[sp]
    print(f"[{sp}] 기존 조합(임계값 A, 보고용) 멀쩡한 응답 표시율 {fmt(r['clean_combo'])} | 환각 응답 탐지 {fmt(r['hal_any'])} | 환각 응답 비율", {g: round(v, 3) for g, v in r['hal_rate_by_gen'].items()})
print(f"\n=== K2-0: train 정답 파싱 {CHK['train_parse_ratio']:.3f} (≥ 0.90), train H 문장 {CHK['train_H_sent']} (≥ 500), "
      f"calM 멀쩡한 응답 {CHK['calM_clean']} (≥ 400), devM 환각 응답 {CHK['devM_hal']} (≥ 150) → {'충족' if CHK['K2_0'] else '미달 → 보류'} ===")
usage = {}
for f in os.listdir(K2_DIR):
    if f.startswith(('gen_', 'label_')) and f.endswith('.json') and '_batches' not in f:
        usage[f] = round(cost(load_json(os.path.join(K2_DIR, f))), 2)
LOCAL = '/content/k2a_local'; os.makedirs(LOCAL, exist_ok=True)
summ = {'gpu': GPU, 'n_sources': {k: len(v) for k, v in SIDS.items()}, 'splits': SPLITS, 'check': CHK, 'report_thrA': REP,
        'thr_combo_A': THR_C, 'cost_usd': usage}
json.dump(summ, open(os.path.join(LOCAL, 'k2a_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for sp, d in ROWS.items(): d.to_csv(os.path.join(LOCAL, f'k2a_{sp}.csv'), index=False)
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(K2_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
print('API 사용량(추정 $)', usage, f'합계 ${sum(usage.values()):.2f}')
zp = shutil.make_archive('/content/k2a_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
# 보험: API 원자료(생성·정답·요청) 백업 — 드라이브가 사라져도 돈을 다시 쓰지 않게 맥에 보관(원문 포함, 저장소에 커밋 금지)
BK = '/content/k2a_raw'; os.makedirs(BK, exist_ok=True)
for f in os.listdir(K2_DIR):
    if f.startswith(('cases_', 'gen_', 'label_', 'scores_')) and f.endswith('.json'): shutil.copy(os.path.join(K2_DIR, f), BK)
files.download(shutil.make_archive('/content/k2a_raw_backup', 'zip', BK))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'k2a_modern_llm_data.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
