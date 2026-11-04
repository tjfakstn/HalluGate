"""I2 노트북 생성: 긴 문맥 판정기 (후보 ①). ModernBERT-large가 근거 전체 + 응답 처음부터 판정 지점까지를 한 번에 본다.
F1 base와 학습 정답(R4b 보정)·판정 지점 정의·평가 세트·평가 함수는 같고, 백본·입력이 다르다.
데이터·평가 코드는 R4c 노트북(=F1)에서, 채택 판정 함수는 I1에서 가져온다. 근거: L2(11/04) 지연 관문 충족, 512 초과 응답 75%."""
import json, os, importlib.util, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
def cells_of(name):
    nb = json.load(open(os.path.join(NB_DIR, name)))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
R4C = cells_of('r4c_clean_labels_base.ipynb')
def find(C, prefix):
    return next(c for c in C if c.startswith(prefix))
def sub(s, a, b, count=1):
    assert s.count(a) == count, (s.count(a), a[:70])
    return s.replace(a, b)

_spec = importlib.util.spec_from_file_location('build_i1', os.path.join(HERE, 'build_i1.py'))
_I1 = importlib.util.module_from_spec(_spec); _argv = sys.argv; _spec.loader.exec_module(_I1); sys.argv = _argv
DECIDE = sub(_I1.DECIDE, "f'I1|{kind}'", "f'I2|{kind}'", count=2)

ENV = find(R4C, "!pip -q install pysbd")
ENV = sub(ENV, "!pip -q install pysbd sentence-transformers tiktoken", '!pip -q install pysbd sentence-transformers tiktoken -U "transformers>=4.48"')
ENV = sub(ENV, "/content/drive/MyDrive/selfproject", "/content/drive/MyDrive/Colab Notebooks/HalluGate")
ENV = sub(ENV, "R1_DIR = os.path.join(OUT_DIR, 'r4c')",
          "R1_DIR = I2_DIR = os.path.join(OUT_DIR, 'i2')\nFINAL_DIR = os.path.join(OUT_DIR, 'final')")
ENV = sub(ENV, "print('device', DEVICE)",
          "assert os.path.exists(os.path.join(FINAL_DIR, 'scores_base.npy')), 'final/scores_base.npy 없음 (F1 재실행 점수, 비교 기준)'\n"
          "import shutil\n"
          "_t = os.path.join(I2_DIR, '_write_test.bin')\n"
          "try:\n"
          "    with open(_t, 'wb') as f: f.write(os.urandom(50 * 2**20))\n"
          "    ok = os.path.getsize(_t) == 50 * 2**20; os.remove(_t)\n"
          "except OSError as e:\n"
          "    ok = False; print('드라이브 쓰기 실패:', e)\n"
          "print('드라이브 50MB 쓰기', '정상' if ok else '실패 — 휴지통 비우기·용량 확인. 가중치는 마지막 셀 zip으로 받는다')\n"
          "print('device', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')")
DATA = find(R4C, "BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find(R4C, "# 학습 응답: S2/S2-b의 gold-large")
PREP = find(R4C, "ENC = tiktoken.get_encoding")
LOADLAB = find(R4C, "TRAIN_ITEMS = to_items(train)")
ADD = find(R4C, "def content_starts(x):")
LABELS = find(R4C, "def points(item, s):")
RETR = find(R4C, "from sentence_transformers import SentenceTransformer")
RETR = sub(RETR, "P_IDX = topk_idx(P_KEY, P_TXT)", "P_IDX = None   # I2: 근거 검색 없음")
RETR = sub(RETR, "EQ_IDX = dict(zip(EVAL_Q, topk_idx([Q_KEYS[q] for q in EVAL_Q], [Q_TEXTS[q] for q in EVAL_Q])))",
           "EQ_IDX = None  # I2: 근거 검색 없음")
CORR = find(R4C, "# R4 보정 정답 C1")
SERIES = find(R4C, "SC = dict(Z0)")
EVAL_DEFS = find(R4C, "def calibrate(items, scorer, arl0):")

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

# ---- I2 전용 셀(I2_POINTS, I2_TRAIN)은 smoke_i2.py가 가짜 데이터·작은 모델로 그대로 실행해 확인한다 ----
I2_POINTS = r"""
# I2 판정 입력 = (근거 전체, 응답 처음부터 판정 지점 토큰까지). 판정 지점·정답은 F1과 같다.
MAX_LEN2 = 2048                    # L2: RAGTruth train 근거+응답 p90 1,182 → 2,048
N_TRAIN = 80000                    # 학습 지점 표본 (F1은 모든 지점 약 25만 개를 1에폭 — 긴 입력이라 일부만)

def need_point(x, t):
    # 평가에 쓰는 지점만 채점: 문장 끝 + 4토큰 간격 (8·16토큰 간격은 그 부분집합)
    return t in x['ends'] or (t - x['sent_of'][t] + 1) % 4 == 0

def train_prefixes(items):
    # LABELS 셀과 같은 순서로 학습 지점마다 응답 앞부분을 만든다 (P_CAT와 1:1)
    out = []
    for x in items:
        for t, txt in points(x, 2):
            if not txt: continue
            out.append((x['ref_key'], x['text'][:x['te'][t]]))
    return out

def build_eval_points(items, resp_text):
    # 평가 지점마다 새 번호. R0 번호는 (근거, 문장 앞부분)으로 묶여 응답 앞부분이 다른 지점을 구분하지 못한다
    keys, texts, need, MAP = [], [], [], {}
    for x in items:
        text = resp_text[x['id']]
        assert len(text) >= x['te'][-1], x['id']
        for t, _ in x['pts2']:
            MAP[(x['id'], t)] = len(keys)
            keys.append(x['ref_key']); texts.append(text[:x['te'][t]]); need.append(need_point(x, t))
    return keys, texts, np.array(need), MAP

def remap(items, MAP):
    out = []
    for x in items:
        y = copy.copy(x); y['pts2'] = [(t, MAP[(x['id'], t)]) for t, _ in x['pts2']]
        out.append(y)
    return out

def encode(tok, pairs):
    return tok([REFS[k] for k, _ in pairs], [p for _, p in pairs], truncation='only_first', max_length=MAX_LEN2,
               padding=True, return_tensors='pt').to(DEVICE)
"""

I2_TRAIN = r"""
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
BATCH, LR, N_EVALS = 16, 1e-5, 4
MICRO = 4 if DEVICE != 'cuda' or torch.cuda.get_device_properties(0).total_memory > 60 * 2**30 else 2

@torch.no_grad()
def score_pairs(tok, model, pairs, bs=32):
    model.eval(); out = np.zeros(len(pairs), dtype=np.float32)
    order = sorted(range(len(pairs)), key=lambda i: len(REFS[pairs[i][0]]) + len(pairs[i][1]))
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
            out[b] = model(**encode(tok, [pairs[j] for j in b])).logits[:, 0].float().cpu().numpy()
    model.train()
    return out

def calsub_metric(tok, model):
    # 체크포인트 선택: cal-sub(보정 정답 C1, F1과 같은 300응답) 8토큰 ARL₀=100 엄격 재현율
    v = score_pairs(tok, model, [(E_KEYS[e], E_TEXTS[e]) for e in CALSUB_E])
    SC['_cur'] = np.full(len(E_KEYS), np.nan, dtype=np.float32); SC['_cur'][CALSUB_E] = v
    return float(evaluate(CALSUB2, '_cur|8', calibrate(CALSUB2, '_cur|8', 100))['recall'])

def train_i2(tok, model, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    idx = random.Random(seed).sample(range(len(TP)), min(N_TRAIN, len(TP)))
    n_steps = max(1, len(idx) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda' and not BF16))
    best = calsub_metric(tok, model); best_state, hist = copy.deepcopy(model.state_dict()), [(0, best)]
    print(f'  시작 cal-sub 재현율 {best:.3f} | {n_steps}단계, 묶음 {BATCH}(조각 {MICRO}), 학습률 {LR}')
    eval_at = {int(n_steps * (i + 1) / N_EVALS) for i in range(N_EVALS)}
    model.train(); t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        batch = idx[(step - 1) * BATCH: step * BATCH]
        opt.zero_grad(); tot = 0.0
        for k in range(0, len(batch), MICRO):
            mb = batch[k:k + MICRO]
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
                z = model(**encode(tok, [TP[g] for g in mb])).logits[:, 0].float()
            loss = F.binary_cross_entropy_with_logits(z, torch.tensor(Y_ALL[mb], device=DEVICE), reduction='sum') / len(batch)
            scaler.scale(loss).backward(); tot += loss.item()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if step in eval_at:
            va = calsub_metric(tok, model); hist.append((step, va))
            if va > best: best, best_state = va, copy.deepcopy(model.state_dict())
            print(f'  단계 {step}/{n_steps} | 손실 {tot:.3f} | cal-sub 재현율 {va:.3f} | {(time.perf_counter() - t0) / 60:.1f}분')
    model.load_state_dict(best_state)
    return model, hist
"""

md(r"""# I2 — 긴 문맥 판정기: 근거 전체 + 응답 앞부분 전체 (ModernBERT-large)

## 왜 하나

F1 base는 근거의 비슷한 조각 8개(512토큰 안)와 **현재 문장 앞부분만** 본다. 그런데 RAGTruth train에서 근거+응답이 512토큰을 넘는 응답이 **75%**(Data2txt 99.7%, Summary 82%)다(L2 사후 확인).
11/04 조사에서 기대 이득이 가장 큰 후보는 **근거 전체와 앞 문장들을 한 번에 보는 긴 문맥 판정기**였고(LettuceDetect와 같은 ModernBERT 백본), L2에서 2,048토큰 판정 1회가 A100 p95 28ms(large)로 8토큰 재판정 예산(68ms) 안에 든다는 것을 확인했다.

## 무엇을 하나

| | F1 base | I2 |
|---|---|---|
| 백본 | DeBERTa-v3-base (NLI 사전학습) | **ModernBERT-large** (`tasksource/ModernBERT-large-nli`, NLI 사전학습. 모델 카드상 RAGTruth·환각 데이터 미포함, Apache-2.0) |
| 근거 | 40단어 조각 top-8 (512토큰 안) | **근거 전체** (입력 2,048토큰, 넘치면 근거 뒤쪽만 자름) |
| 판정 텍스트 | 현재 문장 앞부분 | **응답 처음부터 판정 지점까지** (앞 문장 전부 포함) |
| 판정 지점·학습 정답 | 문장 안 2토큰 간격 + 문장 끝, R4b 보정 정답(문장 단위 "발생 이후인가") | 같음 |
| 학습 | 모든 지점(약 25만) 1에폭, 묶음 64, 학습률 2e-5 | 지점 **8만 개 표본** 1에폭, 묶음 16, 학습률 1e-5 (긴 입력이라 비용 때문) |
| 체크포인트 선택 | cal-sub(C1) 4·2토큰 재현율 | cal-sub(C1, 같은 300응답) **8토큰** 재현율 |

입력이 판정 지점에서 끝나므로 학습과 스트리밍 판정의 조건이 같다(뒤 토큰을 미리 보지 않음).

## 사전 정의 기준 (결과 보기 전, SSOT §6 I2에 고정 — I1 결과를 보기 전에 정함)

비교는 **cal-L에서만**(F1 base = 재실행 `final/scores_base.npy`). 임계값은 각 모델의 cal-L 등각 서비스 10%. I1과 같은 기준이다.
- **I2-0 입력 점검**: 평가 지점 중 근거가 잘리는 비율 ≤ 2%
- **I2-1 (주)**: 8토큰 재판정, 원래 정답 탐지율 차이(I2 − F1 base) **≥ +0.03, 그리고** 응답 단위 쌍체 부트스트랩 95% 하한 > 0
- **I2-2 (안전)**: C1 차이 ≥ −0.01, 원래 정답 탐지 지연 중앙값 증가 ≤ 2토큰
- **I2-3 (지연)**: A100 판정 1회(토크나이즈 포함, 배치 1, 실제 입력) p95 ≤ 68ms
- **채택** = I2-1·2·3 모두 충족. dev-L은 채택 판정 후 보고만.
- **I1과 I2가 모두 채택되면**: cal-L 8토큰 원래 정답 탐지율이 높은 쪽. 차이가 0.01 미만이면 지연이 짧고 게이트웨이 변경이 작은 I1.
- 단서: 시드 1개, 학습 지점 표본 8만 개(F1의 약 3분의 1).

> **런타임: A100** (80GB 권장, 40GB면 조각 크기를 자동으로 줄임). 학습 약 1~1.5시간 + 평가 지점 채점 약 20~30분. 드라이브에 F1과 같은 입력(`r0/`, `r4/`, `r4b/`)과 `final/scores_base.npy`.
> 결과는 드라이브 `Colab Notebooks/HalluGate/i2/`와 `i2_judge.zip`(자동 다운로드).
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터 (F1과 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
md("## 3. 보정 학습 정답 (F1과 같음)")
code(LOADLAB)
code(ADD)
md("## 4. 학습 판정 지점 (F1과 같음)")
code(LABELS)
md("## 5. 평가 세트 (F1과 같음, 근거 검색은 쓰지 않음)")
code(RETR)
code(CORR)
md("## 6. 평가 함수 (F1과 같음)")
code(SERIES)
code(EVAL_DEFS)

md("""## 7. I2 입력: 학습 지점과 평가 지점

평가 지점은 응답 앞부분 전체가 입력이라 R0 질의 번호를 쓸 수 없어, 응답·지점마다 새 번호를 매긴 사본을 만든다(지점·정답은 그대로).""")
code(I2_POINTS + r"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
I2_INIT = 'tasksource/ModernBERT-large-nli'
tok = AutoTokenizer.from_pretrained(I2_INIT)

TP = train_prefixes(train_items)
assert len(TP) == len(P_TXT) == len(P_CAT), (len(TP), len(P_TXT))
_m = np.mean([p.rstrip().endswith(t[-20:].rstrip()) for (_, p), t in zip(TP[:20000], P_TXT[:20000])])
assert _m > 0.98, f'학습 지점 앞부분과 F1 판정 텍스트 끝이 맞지 않음 ({_m:.3f})'
Y_ALL = np.isin(P_CAT, ['win', 'post']).astype(np.float32)        # F1(V0b)과 같은 정답
print(f'학습 지점 {len(TP):,}개 중 {min(N_TRAIN, len(TP)):,}개 표본 | 양성 비율 {Y_ALL.mean():.3f}')

RESP_TEXT = {str(r['id']): r['response'] for r in responses if r['split'] == 'train'}   # cal-L·dev-L은 train 분할에서 나옴
E_KEYS, E_TEXTS, E_NEED, E_MAP = build_eval_points(CAL + DEV, RESP_TEXT)
LAB_SETS_I2 = {lab: (remap(c, E_MAP), remap(d, E_MAP)) for lab, (c, d) in LAB_SETS.items()}
CALSUB2 = remap(CALSUB, E_MAP)
CALSUB_E = sorted({e for x in CALSUB2 for t, e in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % 8 == 0})
NEED_E = np.flatnonzero(E_NEED)
print(f'평가 지점 {len(E_KEYS):,}개 중 채점 {len(NEED_E):,}개 (문장 끝 + 4토큰 간격) | cal-sub 8토큰 지점 {len(CALSUB_E):,}개')

# I2-0 입력 점검: 근거가 잘리는 평가 지점 비율 (표본 5,000)
rs = random.Random(5).sample(list(NEED_E), min(5000, len(NEED_E)))
n_full = [len(tok(REFS[E_KEYS[e]], E_TEXTS[e])['input_ids']) for e in rs]
TRUNC = float(np.mean(np.array(n_full) > MAX_LEN2))
I2_0 = TRUNC <= 0.02
print(f'입력 길이 중앙값 {np.median(n_full):.0f} / p95 {np.percentile(n_full, 95):.0f} | 근거 잘림 {TRUNC:.2%}')
print(f"=== I2-0 입력 점검: {'충족' if I2_0 else '미달 → 입력 길이 재검토, 결과 해석 보류'} ===")
""")

md("## 8. 학습 루프")
code(I2_TRAIN)

md("""## 9. 학습·저장 (시드 42)

로컬에 fp16으로 먼저 저장하고 드라이브에 복사·크기 검증. 점수는 저장한 파일을 다시 불러 매긴다. 채점은 조각마다 저장해 끊겨도 이어서 한다.""")
code(r"""
LOCAL = '/content/i2_local'
D = os.path.join(LOCAL, 'judge_i2')
os.makedirs(LOCAL, exist_ok=True)
WFILES = ['model.safetensors', 'pytorch_model.bin']
def has_weights(d):
    return os.path.isdir(d) and any(os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0 for f in WFILES)
def sizes(d):
    return {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))}
def to_drive():
    dst = os.path.join(I2_DIR, 'judge_i2')
    try:
        shutil.copytree(D, dst, dirs_exist_ok=True)
        ok = sizes(D) == {f: v for f, v in sizes(dst).items() if f in sizes(D)}
    except OSError as e:
        print('드라이브 복사 실패:', e); ok = False
    print(f'드라이브 저장 {"확인" if ok else "실패 — 마지막 셀 zip을 꼭 내려받을 것"}')
    return ok
def fix_cfg(d):
    # 초기 NLI 모델 설정의 problem_type='single_label_classification'이 num_labels=1 저장본에 남으면
    # 새 transformers가 다시 불러올 때 거부한다. 손실은 직접 계산하므로 problem_type을 지운다.
    p = os.path.join(d, 'config.json')
    if os.path.isfile(p):
        c = json.load(open(p))
        if c.pop('problem_type', None) is not None:
            json.dump(c, open(p, 'w'), indent=2); print('config.json problem_type 제거:', d)
def load_model(path, **kw):
    fix_cfg(path)
    return AutoModelForSequenceClassification.from_pretrained(path, attn_implementation='sdpa', reference_compile=False, **kw).to(DEVICE)

DD = os.path.join(I2_DIR, 'judge_i2')
if not has_weights(D) and has_weights(DD):
    shutil.copytree(DD, D, dirs_exist_ok=True); print('드라이브 저장본 사용')
fix_cfg(DD)
if not has_weights(D):
    t0 = time.perf_counter()
    model = load_model(I2_INIT, num_labels=1, ignore_mismatched_sizes=True)
    model, hist = train_i2(tok, model, 42)
    model.config.problem_type = None
    os.makedirs(D, exist_ok=True); tok.save_pretrained(D); model.half().save_pretrained(D)
    pd.DataFrame(hist, columns=['step', 'calsub_recall8']).to_csv(os.path.join(D, 'hist.csv'), index=False)
    print(f'학습 {(time.perf_counter() - t0) / 60:.0f}분')
    del model; gc.collect(); torch.cuda.empty_cache()
    DRIVE_OK = to_drive()
else:
    DRIVE_OK = has_weights(DD)

sp, spd = os.path.join(LOCAL, 'scores_i2.npy'), os.path.join(I2_DIR, 'scores_i2.npy')
if not os.path.exists(sp) and os.path.exists(spd): shutil.copy(spd, sp)
if not os.path.exists(sp):
    model = load_model(D).float()
    full = np.full(len(E_KEYS), np.nan, dtype=np.float32)
    PART = 20000
    t0 = time.perf_counter()
    for i in range(0, len(NEED_E), PART):
        pp = os.path.join(I2_DIR, f'scores_part{i // PART:03d}.npy')
        part = NEED_E[i:i + PART]
        if os.path.exists(pp):
            v = np.load(pp)
        else:
            v = score_pairs(tok, model, [(E_KEYS[e], E_TEXTS[e]) for e in part])
            try: np.save(pp, v)
            except OSError as e: print('조각 저장 실패', e)
        full[part] = v
        print(f'  채점 {min(i + PART, len(NEED_E)):,}/{len(NEED_E):,} | {(time.perf_counter() - t0) / 60:.1f}분')
    np.save(sp, full)
    try: shutil.copy(sp, spd)
    except OSError as e: print('점수 드라이브 저장 실패', e)
    del model; gc.collect(); torch.cuda.empty_cache()
SC['I2'] = np.load(sp)
SC['F_base'] = np.load(os.path.join(FINAL_DIR, 'scores_base.npy'))
""")

md("""## 10. 채택 판정 (cal-L만)

먼저 F1 base 점수로 F1 재실행의 8토큰 서비스 임계값(3.969)이 재현되는지 확인한다.""")
code(DECIDE + r"""
_chk = conformal_flag(CAL, 'F_base|8')
assert abs(_chk - 3.968751) < 1e-3, f'F1 재실행 임계값 재현 실패: {_chk}'
print(f'F1 base 8토큰 cal-L 서비스 임계값 재현: {_chk:.4f}')

CMP = {k: compare(LAB_SETS, LAB_SETS_I2, k) for k in ['end', '4', '8', '16']}
rows = [{'kind': k, 'labels': lab, **v} for k, d in CMP.items() for lab, v in d.items()]
cmp_tab = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(cmp_tab.round(3).to_string(index=False))

# I2-3 지연: 판정 1회 = 토크나이즈 + 순전파, 배치 1, 실제 평가 입력 200개
model = load_model(D).float().eval()
@torch.no_grad()
def judge_once(e):
    enc = encode(tok, [(E_KEYS[e], E_TEXTS[e])])
    with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=(DEVICE == 'cuda')):
        s = float(model(**enc).logits.squeeze().float().cpu())
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return s
LQ = random.Random(2027).sample(list(NEED_E), 200)
for e in LQ[:10]: judge_once(e)
ms = []
for e in LQ:
    t0 = time.perf_counter(); judge_once(e); ms.append((time.perf_counter() - t0) * 1000)
P95 = float(np.percentile(ms, 95))
GPU = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'
print(f'판정 1회 {GPU}: p50 {np.percentile(ms, 50):.1f} / p95 {P95:.1f} ms')
del model; gc.collect(); torch.cuda.empty_cache()

o, c1 = CMP['8']['orig'], CMP['8']['C1']
I2_1 = o['delta'] >= 0.03 and o['ci_lo'] > 0
I2_2 = c1['delta'] >= -0.01 and (o['delay_i1'] - o['delay_base']) <= 2
I2_3 = P95 <= 68 and 'A100' in GPU
ADOPT = bool(I2_0 and I2_1 and I2_2 and I2_3)
print(f"=== I2-1 (주) 8토큰 원래 정답: {o['recall_base']:.3f} → {o['recall_i1']:.3f} (Δ {o['delta']:+.3f} [{o['ci_lo']:+.3f}, {o['ci_hi']:+.3f}]) → {'충족' if I2_1 else '미달'} ===")
print(f"=== I2-2 (안전) C1 Δ {c1['delta']:+.3f}, 지연 {o['delay_base']:.0f} → {o['delay_i1']:.0f}토큰 → {'충족' if I2_2 else '미달'} ===")
print(f"=== I2-3 (지연) {GPU} p95 {P95:.1f}ms → {'충족' if I2_3 else '미달 (A100에서 다시 재야 함)' if P95 <= 68 else '미달'} ===")
print(f"=== I2 채택: {'채택 후보 (I1도 채택이면 SSOT §6 규칙으로 하나 선택)' if ADOPT else '미채택'} ===")
print('(표의 *_i1 열 = I2 값. 판정 함수를 I1 노트북과 공유)')
""")

md("""## 11. dev-L 확인 (보고만, 채택 판정에 쓰지 않음)""")
code(r"""
def evaluate_at(items, scorer, thr):
    r = evaluate(items, scorer, thr)
    return {'recall': r['recall'], 'delay_med': r['delay_tok_median'], 'clean_flagged': r['clean_resp_flagged']}
DEV_ROWS = []
for k in ['end', '4', '8', '16']:
    for lab in ['orig', 'C1']:
        c = CMP[k][lab]
        for model_name, sets, sc, thr in [('F_base', LAB_SETS, f'F_base|{k}', c['thr_base']), ('I2', LAB_SETS_I2, f'I2|{k}', c['thr_i1'])]:
            dev_ = sets[lab][1]
            DEV_ROWS.append({'model': model_name, 'kind': k, 'labels': lab, 'task': 'all', **evaluate_at(dev_, sc, thr)})
            if k == '8' and lab == 'orig':
                for task in sorted({x['task_type'] for x in dev_}):
                    DEV_ROWS.append({'model': model_name, 'kind': k, 'labels': lab, 'task': task,
                                     **evaluate_at([x for x in dev_ if x['task_type'] == task], sc, thr)})
dev_tab = pd.DataFrame(DEV_ROWS)
print(dev_tab.round(3).to_string(index=False))
""")

md("## 12. 저장과 zip 다운로드")
code(r"""
cfg = {'name': 'judge_i2', 'backbone': I2_INIT, 'seed': 42, 'n_train_points': min(N_TRAIN, len(TP)),
       'input': {'text_a': 'full reference (build_reference), truncated from the end if over max_len',
                 'text_b': 'response from its start up to and including the judged token', 'retrieval': None},
       'max_len': MAX_LEN2, 'stride_unit': 'tiktoken cl100k_base tokens, counted from sentence start (sentence end always judged)',
       'default': {'kind': '8', 'threshold': 'svc10_orig'}, 'threshold': CMP['8']['orig']['thr_i1'],
       'thresholds': {k: {f'svc10_{lab}': CMP[k][lab]['thr_i1'] for lab in ['orig', 'C1']} for k in CMP},
       'score': 'logit (higher = hallucinated)', 'weights_dtype': 'float16 (load as float32)',
       'i2': {'I2_0': bool(I2_0), 'I2_1': bool(I2_1), 'I2_2': bool(I2_2), 'I2_3': bool(I2_3), 'adopt': ADOPT}}
json.dump(cfg, open(os.path.join(D, 'judge_config.json'), 'w'), indent=1, ensure_ascii=False)
summ = {'I2_0': bool(I2_0), 'I2_1': bool(I2_1), 'I2_2': bool(I2_2), 'I2_3': bool(I2_3), 'adopt': ADOPT,
        'trunc': TRUNC, 'p95_ms': P95, 'gpu': GPU, 'drive_weights_ok': DRIVE_OK, 'calL_compare': CMP}
json.dump(summ, open(os.path.join(LOCAL, 'i2_summary.json'), 'w'), default=float, ensure_ascii=False, indent=1)
cmp_tab.to_csv(os.path.join(LOCAL, 'i2_calL.csv'), index=False)
dev_tab.to_csv(os.path.join(LOCAL, 'i2_devL.csv'), index=False)
if DRIVE_OK: to_drive()
for f in ['i2_summary.json', 'i2_calL.csv', 'i2_devL.csv']:
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(I2_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/i2_judge', 'zip', LOCAL)
print(f'zip {os.path.getsize(zp) / 2**20:.0f}MB:', zp)
from google.colab import files
files.download(zp)
print(json.dumps({k: summ[k] for k in ['I2_0', 'I2_1', 'I2_2', 'I2_3', 'adopt', 'trunc', 'p95_ms']}, ensure_ascii=False, indent=1, default=float))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'i2_longctx_judge.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
