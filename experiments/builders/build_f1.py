"""F1 노트북 생성: 모델 완성(동결).
R4c base(DeBERTa-v3-base)와 R4b V0b(MiniLM)를 보정 학습 정답으로 시드 42 1개씩 재학습해 가중치를 확실히 남기고,
cal-L로 운영 임계값을 정하고, 게이트웨이 판정기 코드(hallugate/judge.py)로 같은 점수가 나오는지,
CPU·GPU 판정 지연과 int8 양자화를 확인해 배포 구성을 고른다.
데이터·평가 코드는 R4c 노트북, MiniLM 학습 루프는 R4b 노트북에서 가져온다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')
def cells_of(name):
    nb = json.load(open(os.path.join(NB_DIR, name)))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
R4C, R4B = cells_of('r4c_clean_labels_base.ipynb'), cells_of('r4b_relabel_retrain.ipynb')
def find(C, prefix):
    return next(c for c in C if c.startswith(prefix))
def sub(s, a, b):
    assert a in s, a[:70]
    return s.replace(a, b, 1)

JUDGE_SRC = open(os.path.join(ROOT, 'hallugate', 'judge.py'), encoding='utf-8').read()

ENV = find(R4C, "!pip -q install pysbd")
ENV = sub(ENV, "/content/drive/MyDrive/selfproject", "/content/drive/MyDrive/Colab Notebooks/HalluGate")   # 새 구글 계정(10/28)
ENV = sub(ENV, "R1_DIR = os.path.join(OUT_DIR, 'r4c')", "R1_DIR = FINAL_DIR = os.path.join(OUT_DIR, 'final')\nR4C_DIR = os.path.join(OUT_DIR, 'r4c')")
ENV = sub(ENV, "print('device', DEVICE)",
          "assert os.path.isdir(STUDENT_DIR), f'{STUDENT_DIR} 없음 (MiniLM 학습 시작점)'\n"
          "# 드라이브 쓰기 점검: R4c에서는 용량 부족으로 가중치가 조용히 저장되지 않았다\n"
          "import shutil\n"
          "_t = os.path.join(FINAL_DIR, '_write_test.bin')\n"
          "try:\n"
          "    with open(_t, 'wb') as f: f.write(os.urandom(50 * 2**20))\n"
          "    ok = os.path.getsize(_t) == 50 * 2**20; os.remove(_t)\n"
          "except OSError as e:\n"
          "    ok = False; print('드라이브 쓰기 실패:', e)\n"
          "print('드라이브 50MB 쓰기', '정상' if ok else '실패 — 휴지통 비우기·용량 확인. 가중치는 마지막 셀의 zip 다운로드로 받는다')\n"
          "print('device', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')")
DATA = find(R4C, "BASE = 'RAGTruth/dataset'")
TRAIN_SPLIT = find(R4C, "# 학습 응답: S2/S2-b의 gold-large")
PREP = find(R4C, "ENC = tiktoken.get_encoding")
LOADLAB = find(R4C, "TRAIN_ITEMS = to_items(train)")
ADD = find(R4C, "def content_starts(x):")
LABELS = find(R4C, "def points(item, s):")
RETR = find(R4C, "from sentence_transformers import SentenceTransformer")
CORR = find(R4C, "# R4 보정 정답 C1")
SERIES = find(R4C, "SC = dict(Z0)")
EVAL_DEFS = find(R4C, "def calibrate(items, scorer, arl0):")
AUCS = find(R4C, "def resp_auc(items, m):")
TRAIN_BASE = find(R4C, "KEEP_WEIGHTS = {42}")            # R1c 학습 루프 (train_backbone)
TRAIN_MINI = find(R4B, "from transformers import get_linear_schedule_with_warmup")   # R1b/R4b 학습 루프 (train_variant)
assert 'def train_backbone' in TRAIN_BASE and 'def train_variant' in TRAIN_MINI

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# F1 — 모델 완성(동결): 최종 가중치·운영 임계값·게이트웨이 일치·판정 지연

## 왜 하나

모델링은 R4c로 결론이 났다(SSOT 10/15·10/21). 게이트웨이 교체 후보는 **R4c base(DeBERTa-v3-base, 보정 학습 정답, 4토큰)**,
작은 대안은 **R4b V0b(MiniLM)**. 그런데 둘 다 드라이브 용량 때문에 **가중치가 저장되지 않았고**, 운영 임계값·CPU 지연·
게이트웨이 코드와의 점수 일치도 확인한 적이 없다. 이 노트북은 연구가 아니라 **모델을 서비스에 넣을 수 있는 상태로 만드는 마무리**다.

## 무엇을 하나

1. 두 모델을 R4c·R4b와 **같은 방식으로 시드 42 1개씩 재학습**하고, fp16으로 로컬에 먼저 저장한 뒤 드라이브에 복사·크기 검증. 실패해도 마지막 셀에서 zip으로 내려받는다.
2. **재현 확인**: 재학습 모델의 dev-L 성능이 기존 시드 3개와 같은 수준인지.
3. **운영 임계값**: cal-L에서 판정 간격(문장 끝·4·8·16토큰)마다 ARL₀=100 임계값과 등각 보정 서비스 10% 임계값(멀쩡한 응답 표시 ≤ 10%)을 정해 `judge_config.json`에 넣는다.
4. **게이트웨이 일치**: 저장한 가중치를 게이트웨이 `hallugate/judge.py`(그대로 복사)로 불러 같은 점수가 나오는지. 저장소 테스트용 고정 예제 50개도 내보낸다.
5. **판정 지연**: GPU, CPU(스레드 수별) fp32·int8. 사전 규칙으로 배포 구성(모델·재판정 간격)을 고른다.

## 사전 정의 기준 (결과 보기 전, SSOT §6 F1에 고정)

- **F1-1 재현**: 재학습 모델의 dev-L 4토큰 ARL₀=100 엄격 재현율이 원래 정답·C1 모두 **같은 방식 기존 시드 3개 최솟값 − 0.01 이상** (base: R4c, mini: R4b V0b). 미달 모델은 교체 후보에서 뺀다.
- **F1-2 일치**: 게이트웨이 코드로 cal-L 질의 300개 채점 → 노트북 점수와 **Pearson ≥ 0.995, 기본 임계값 판정 일치 ≥ 99%** (GPU·CPU 각각).
- **F1-3 int8**: CPU 동적 양자화 점수가 fp32 대비 **Pearson ≥ 0.99, 판정 일치 ≥ 98%** 면 int8 사용 허용.
- **배포 구성 규칙**: 토큰당 예산 8.5ms(L1 Haiku 평균 속도 환산 4토큰 ≈ 34ms). 각 (모델, 장치·정밀도)에서 **판정 p95 ≤ s × 8.5ms인 최소 재판정 간격 s ∈ {4, 8, 16}**. 우선순위 base(s ≤ 8) → mini(s ≤ 8) → base 문장 끝만.
- **기본 임계값** = cal-L **원래 정답** 등각 보정 서비스 10%(Claude 재판정에 의존하지 않고 보수적). ARL₀=100·C1 임계값도 함께 저장. **임계값·모델 선택에 dev-L을 쓰지 않는다**(재현 확인·보고만).

> **런타임: A100** (L4 가능). base 학습 약 80~95분 + MiniLM 약 10분 + 채점·측정 약 30분 → **약 2~2.5시간**. 끝난 단계는 건너뛴다.
> **실행 전에 드라이브 휴지통을 비울 것.** 드라이브에 `student_vark`, `r0/`, `r4/r4_corrected_labels.json`, `r4b/labels_train.json`·`V0b_s*/scores.npy`, `r4c/base_s*/scores.npy`가 있어야 한다.
> 결과는 드라이브 `Colab Notebooks/HalluGate/final/`과 `f1_final_judges.zip`(자동 다운로드).
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터 (R4c와 같음)")
code(DATA)
code(TRAIN_SPLIT)
code(PREP)
md("## 3. 보정 학습 정답 (R4b Sonnet 재판정, 추가 비용 없음)")
code(LOADLAB)
code(ADD)
md("## 4. 학습 판정 지점")
code(LABELS)
md("## 5. 근거 검색과 평가용 보정 정답(R4 C1)")
code(RETR)
code(CORR)
md("## 6. 평가 함수 (R1b와 같음)")
code(SERIES)
code(EVAL_DEFS)
code(AUCS)
md("## 7. 학습 루프 — base는 R1c(=R4c), MiniLM은 R1b(=R4b) 그대로")
code(TRAIN_BASE)
code(TRAIN_MINI)

md("""## 8. 최종 모델 학습·저장 (시드 42)

로컬(`/content/final_local`)에 fp16으로 먼저 저장하고 드라이브에 복사한 뒤 파일 크기가 같은지 확인한다.
평가 점수는 **저장한 파일을 다시 불러서** 매긴다(임계값이 실제로 배포할 가중치의 점수와 맞도록).""")
code(r"""
LOCAL = '/content/final_local'
os.makedirs(LOCAL, exist_ok=True)
WFILES = ['model.safetensors', 'pytorch_model.bin']
NAMES = ['base', 'mini']
BACKBONE_OF = {'base': 'cross-encoder/nli-deberta-v3-base (R1c 방식, 사전학습 NLI에서 시작)',
               'mini': 'cross-encoder/nli-MiniLM2-L6-H768 → student_vark에서 시작 (R1b V0b 방식)'}
DRIVE_OK = {}

def has_weights(d):
    return os.path.isdir(d) and any(os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0 for f in WFILES)

def sizes(d):
    return {f: os.path.getsize(os.path.join(d, f)) for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))}

def to_drive(name):
    src, dst = os.path.join(LOCAL, f'judge_{name}'), os.path.join(FINAL_DIR, f'judge_{name}')
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        ok = sizes(src) == {f: v for f, v in sizes(dst).items() if f in sizes(src)}
    except OSError as e:
        print(f'  [{name}] 드라이브 복사 실패: {e}'); ok = False
    DRIVE_OK[name] = ok
    print(f'  [{name}] 드라이브 저장 {"확인" if ok else "실패 — 마지막 셀 zip을 꼭 내려받을 것"} ({sum(sizes(src).values()) / 2**20:.0f}MB)')

def load_judge(name):
    d = os.path.join(LOCAL, f'judge_{name}')
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).float().to(DEVICE)   # fp16 저장 → fp32로 올림 (게이트웨이와 같게)
    return tok, model

def finalize(name):
    d, dd = os.path.join(LOCAL, f'judge_{name}'), os.path.join(FINAL_DIR, f'judge_{name}')
    if not has_weights(d) and has_weights(dd):
        shutil.copytree(dd, d, dirs_exist_ok=True); print(f'  [{name}] 드라이브 저장본 사용')
    if not has_weights(d):
        t0 = time.perf_counter()
        if name == 'base':
            tok, model, hist = train_backbone('base', 42)
        else:
            tok, model, _, hist = train_variant('V0b', 42)
        os.makedirs(d, exist_ok=True)
        tok.save_pretrained(d); model.half().save_pretrained(d)
        pd.DataFrame(hist, columns=['step', 'calsub_recall', 'calsub_delay4']).to_csv(os.path.join(d, 'hist.csv'), index=False)
        print(f'  [{name}] 학습 {(time.perf_counter() - t0) / 60:.0f}분, 로컬 저장')
        del model; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
        to_drive(name)
    sp, spd = os.path.join(LOCAL, f'scores_{name}.npy'), os.path.join(FINAL_DIR, f'scores_{name}.npy')
    if not os.path.exists(sp) and os.path.exists(spd): shutil.copy(spd, sp)
    if not os.path.exists(sp):
        tok, model = load_judge(name)
        t0 = time.perf_counter()
        v = score_queries(tok, model, EVAL_Q)
        full = np.full(len(Q_KEYS), np.nan, dtype=np.float32); full[EVAL_Q] = v
        np.save(sp, full)
        try: shutil.copy(sp, spd)
        except OSError as e: print('  점수 드라이브 저장 실패', e)
        print(f'  [{name}] 평가 질의 {len(EVAL_Q):,}개 채점 {time.perf_counter() - t0:.0f}초')
        del model; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    SC[f'F_{name}'] = np.load(sp)
    DRIVE_OK.setdefault(name, has_weights(dd))
    print(f'[{name}] 준비 완료')

for name in ['mini', 'base']:          # 짧은 MiniLM 먼저 (저장 경로 점검)
    finalize(name)
""")

md("## 9. F1-1 재현 확인 (dev-L, 기존 시드 3개와 비교)")
code(r"""
SEEDS = [42, 7, 2026]
REF_PATHS = {'base': [os.path.join(R4C_DIR, f'base_s{s}', 'scores.npy') for s in SEEDS],
             'mini': [os.path.join(R4B_DIR, f'V0b_s{s}', 'scores.npy') for s in SEEDS]}
for name, paths in REF_PATHS.items():
    for s, p in zip(SEEDS, paths):
        assert os.path.exists(p), f'{p} 없음'
        SC[f'ref_{name}_s{s}'] = np.load(p)

F1_1, rep = {}, []
for name in NAMES:
    ok = True
    for lab, (cal_, dev_) in LAB_SETS.items():
        sc = f'F_{name}|4'
        r = evaluate(dev_, sc, calibrate(cal_, sc, 100))
        refs = [evaluate(dev_, f'ref_{name}_s{s}|4', calibrate(cal_, f'ref_{name}_s{s}|4', 100))['recall'] for s in SEEDS]
        floor = min(refs) - 0.01; ok &= r['recall'] >= floor
        rep.append({'model': name, 'labels': lab, 'recall': r['recall'], 'delay_med': r['delay_tok_median'],
                    'ref_seeds': np.round(refs, 3).tolist(), 'floor': floor, 'pass': r['recall'] >= floor})
    F1_1[name] = bool(ok)
rep = pd.DataFrame(rep)
print(rep.round(3).to_string(index=False))
for name in NAMES:
    print(f"=== F1-1 재현 [{name}]: {'충족' if F1_1[name] else '미달 → 교체 후보에서 제외'} ===")
""")

md("""## 10. 운영 임계값 (cal-L에서만 정함)과 dev-L 확인

- `arl100_*`: 멀쩡한 응답 100토큰당 경보 1회(엄격 평가 조건)
- `svc10_*`: **등각 보정** — 멀쩡한 응답 중 표시 비율 ≤ 10% (응답별 최대 점수의 ⌈(n+1)(1−α)⌉번째 값)
- `*_orig`: 원래 RAGTruth 정답의 멀쩡한 응답으로 보정(기본), `*_C1`: Opus 보정 정답 기준""")
code(r"""
KINDS_F = ['end', '4', '8', '16']
ALPHA = 0.10
def conformal_flag(items, scorer, alpha=ALPHA):
    mx = np.sort([max([s for _, s in series(x, scorer)] or [-1e9]) for x in items if x['onset'] is None])
    n = len(mx); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(mx[k - 1]) + 1e-6 if k <= n else float('inf')

THR = {}
for name in NAMES:
    THR[name] = {}
    for k in KINDS_F:
        sc = f'F_{name}|{k}'
        THR[name][k] = {}
        for lab, (cal_, _) in LAB_SETS.items():
            THR[name][k][f'arl100_{lab}'] = calibrate(cal_, sc, 100)
            THR[name][k][f'svc10_{lab}'] = conformal_flag(cal_, sc)
DEFAULT = {'kind': '4', 'threshold': 'svc10_orig'}

rows = []
for name in NAMES:
    for k in KINDS_F:
        sc = f'F_{name}|{k}'
        for tk in ['arl100_orig', 'svc10_orig', 'svc10_C1']:
            for evlab, (_, dev_) in LAB_SETS.items():
                r = evaluate(dev_, sc, THR[name][k][tk])
                rows.append({'model': name, 'kind': k, 'thr': tk, 'eval_labels': evlab, 'threshold': THR[name][k][tk],
                             'recall': r['recall'], 'delay_med': r['delay_tok_median'], 'pre_onset': r['pre_onset_alarm_rate'],
                             'clean_flagged': r['clean_resp_flagged'], 'calls_per_resp': r['calls_per_resp']})
dev_tab = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(dev_tab.round(3).to_string(index=False))
# 기준선 참고 (같은 방식, dev-L)
for lab, (cal_, dev_) in LAB_SETS.items():
    r = evaluate(dev_, 'B|end', calibrate(cal_, 'B|end', 100))
    print(f"(참고) B 문장 끝 ARL₀=100 {lab}: 재현율 {r['recall']:.3f}, 지연 {r['delay_tok_median']:.0f}")
""")

md("""## 11. F1-2 게이트웨이 코드와 점수 일치

저장소의 `hallugate/judge.py`를 그대로 써서(노트북 생성 시점의 코드) 저장한 가중치를 불러 채점한다.
저장소 테스트용 고정 예제 50개(근거, 판정 텍스트, CPU fp32 점수)를 `parity_fixture.json`으로 내보낸다.""")
code("JUDGE_SRC = " + repr(JUDGE_SRC) + r"""
open('hallugate_judge.py', 'w').write(JUDGE_SRC)
import importlib, sys
sys.path.insert(0, '.')
import hallugate_judge; importlib.reload(hallugate_judge)
from hallugate_judge import StudentJudge

CAL_Q = sorted({q for x in CAL for _, q in x['pts2']})
Q300 = random.Random(2027).sample(CAL_Q, 300)

def gw_scores(j, qs):
    cache, out = {}, []
    for q in qs:
        key = Q_KEYS[q]
        if key not in cache: cache[key] = j.prepare(REFS[key])
        out.append(j.judge(cache[key], Q_TEXTS[q]).score)
    return np.array(out, dtype=np.float64)

def agree(a, b, thr):
    return float(np.mean((a >= thr) == (b >= thr)))

F1_2, PAR, GW_CPU = {}, [], {}
for name in NAMES:
    d = os.path.join(LOCAL, f'judge_{name}')
    nb = SC[f'F_{name}'][Q300].astype(np.float64)
    thr = THR[name][DEFAULT['kind']][DEFAULT['threshold']]
    ok = True
    for dev in (['cuda'] if DEVICE == 'cuda' else []) + ['cpu']:
        torch.set_num_threads(os.cpu_count())
        j = StudentJudge(d, k=K, max_len=MAX_LEN, chunk_words=CHUNK_WORDS, device=dev)
        g = gw_scores(j, Q300)
        if dev == 'cpu': GW_CPU[name] = g
        pr, ag, mx = float(np.corrcoef(nb, g)[0, 1]), agree(nb, g, thr), float(np.max(np.abs(nb - g)))
        nan = int(np.isnan(g).sum())
        okd = nan == 0 and pr >= 0.995 and ag >= 0.99; ok &= okd
        PAR.append({'model': name, 'device': dev, 'pearson': pr, 'agree': ag, 'max_abs_diff': mx, 'nan': nan, 'pass': okd})
        del j; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    F1_2[name] = bool(ok)
    fx = [{'reference': REFS[Q_KEYS[q]], 'text': Q_TEXTS[q], 'score_cpu_fp32': float(s)} for q, s in zip(Q300[:50], GW_CPU[name][:50])]
    json.dump({'model': f'judge_{name}', 'k': K, 'max_len': MAX_LEN, 'chunk_words': CHUNK_WORDS,
               'embedder': 'sentence-transformers/all-MiniLM-L6-v2', 'items': fx},
              open(os.path.join(d, 'parity_fixture.json'), 'w'), ensure_ascii=False)
PAR = pd.DataFrame(PAR)
print(PAR.round(4).to_string(index=False))
for name in NAMES:
    print(f"=== F1-2 게이트웨이 일치 [{name}]: {'충족' if F1_2[name] else '미달 → 게이트웨이 판정기 코드 수정 필요'} ===")
""")

md("""## 12. F1-3 int8과 판정 지연, 배포 구성

판정 1회 = 판정 텍스트 임베딩 + 근거 top-8 검색 + 교차 인코더(게이트웨이 `judge()`와 같음). 근거 준비(`prepare`)는 요청당 1회라 따로 적는다.
CPU는 이 런타임의 vCPU로 잰다 — 실제 배포 서버와 다를 수 있으므로 **스레드 수별로** 적는다.""")
code(r"""
TEXTS = [(REFS[Q_KEYS[q]], Q_TEXTS[q]) for q in Q300]
def bench(j, n):
    refs = {}
    for r, _ in TEXTS[:5]:                             # 예열
        if r not in refs: refs[r] = j.prepare(r)
        j.judge(refs[r], 'warm up sentence.')
    prep, ms = [], []
    for r, t in TEXTS[:n]:
        if r not in refs:
            refs[r] = j.prepare(r); prep.append(refs[r].prepare_ms)
        ms.append(j.judge(refs[r], t).judge_ms)
    return np.percentile(ms, 50), np.percentile(ms, 95), (np.median(prep) if prep else np.nan)

def quantize(j):
    j.model = torch.ao.quantization.quantize_dynamic(j.model, {torch.nn.Linear}, dtype=torch.qint8)
    return j

F1_3, I8 = {}, []
NT = os.cpu_count()
torch.set_num_threads(NT)
for name in NAMES:
    j = quantize(StudentJudge(os.path.join(LOCAL, f'judge_{name}'), k=K, max_len=MAX_LEN, chunk_words=CHUNK_WORDS, device='cpu'))
    g8 = gw_scores(j, Q300); g = GW_CPU[name]
    thr = THR[name][DEFAULT['kind']][DEFAULT['threshold']]
    pr, ag = float(np.corrcoef(g, g8)[0, 1]), agree(g, g8, thr)
    F1_3[name] = bool(pr >= 0.99 and ag >= 0.98)
    I8.append({'model': name, 'pearson': pr, 'agree': ag, 'pass': F1_3[name]})
    print(f"=== F1-3 int8 [{name}]: Pearson {pr:.4f}, 판정 일치 {ag:.3f} → {'충족 (int8 허용)' if F1_3[name] else '미달 (fp32만)'} ===")
    del j; gc.collect()

LAT = []
N_GPU, N_CPU = 200, 60
for name in NAMES:
    d = os.path.join(LOCAL, f'judge_{name}')
    if DEVICE == 'cuda':
        j = StudentJudge(d, k=K, max_len=MAX_LEN, chunk_words=CHUNK_WORDS, device='cuda')
        p50, p95, pm = bench(j, N_GPU)
        LAT.append({'model': name, 'config': f'gpu:{torch.cuda.get_device_name(0)}', 'p50': p50, 'p95': p95, 'prepare_ms': pm})
        del j; gc.collect(); torch.cuda.empty_cache()
    for nt in sorted({1, 2, 4, NT}):
        torch.set_num_threads(nt)
        for prec in ['fp32', 'int8']:
            if prec == 'int8' and not F1_3[name]: continue
            j = StudentJudge(d, k=K, max_len=MAX_LEN, chunk_words=CHUNK_WORDS, device='cpu')
            if prec == 'int8': quantize(j)
            p50, p95, pm = bench(j, N_CPU)
            LAT.append({'model': name, 'config': f'cpu:{nt}t:{prec}', 'p50': p50, 'p95': p95, 'prepare_ms': pm})
            print(f'  [{name}] cpu {nt}스레드 {prec}: p50 {p50:.0f} / p95 {p95:.0f} ms')
            del j; gc.collect()
torch.set_num_threads(NT)

BUDGET = 8.5          # ms/토큰 (L1 Haiku 평균 속도 환산 4토큰 ≈ 34ms)
def min_stride(p95):
    return next((s for s in [4, 8, 16] if p95 <= s * BUDGET), np.nan)
LAT = pd.DataFrame(LAT)
LAT['min_stride'] = [min_stride(v) for v in LAT['p95']]
print(LAT.round(1).to_string(index=False))

def recall_at(name, s):
    k = str(s) if s else 'end'
    row = dev_tab[(dev_tab.model == name) & (dev_tab.kind == k) & (dev_tab.thr == 'svc10_orig') & (dev_tab.eval_labels == 'orig')].iloc[0]
    return k, row['recall'], row['delay_med']

def choose(cfg_prefix):
    # 사전 규칙: base(s ≤ 8) → mini(s ≤ 8) → base 문장 끝만. F1-1·F1-2 미달 모델은 제외
    for name in NAMES:
        if not (F1_1[name] and F1_2[name]): continue
        c = LAT[(LAT.model == name) & LAT.config.str.startswith(cfg_prefix)]
        c = c[c.min_stride.notna() & (c.min_stride <= 8)]
        if len(c):
            best = c.sort_values(['min_stride', 'p95']).iloc[0]
            return {'model': name, 'config': best.config, 'stride_tokens': int(best.min_stride), 'p95_ms': float(best.p95)}
    return {'model': 'base', 'config': cfg_prefix, 'stride_tokens': None, 'p95_ms': None}   # 문장 끝만

DEPLOY = {}
if DEVICE == 'cuda': DEPLOY['gpu'] = choose('gpu')
DEPLOY['cpu'] = choose(f'cpu:{NT}t')
for where, dcfg in DEPLOY.items():
    k, rc, dl = recall_at(dcfg['model'], dcfg['stride_tokens'])
    dcfg.update({'devL_svc10_recall_orig': float(rc), 'devL_delay_med': float(dl)})
    print(f"=== 배포 구성 [{where}]: {dcfg['model']} / {dcfg['config']} / 재판정 간격 {dcfg['stride_tokens'] or '문장 끝만'}토큰 "
          f"(p95 {dcfg['p95_ms'] or float('nan'):.0f}ms) → dev-L 서비스 10% 재현율 {rc:.3f}, 지연 {dl:.0f}토큰 ===")
""")

md("## 13. 저장 (판정기 폴더 + 요약) 과 zip 다운로드")
code(r"""
for name in NAMES:
    d = os.path.join(LOCAL, f'judge_{name}')
    cfg = {'name': f'judge_{name}', 'backbone': BACKBONE_OF[name], 'seed': 42,
           'trained_on': 'RAGTruth train gold-large 3,000 responses, sentence-local labels + R4b Sonnet re-judged labels (C1-style)',
           'embedder': 'sentence-transformers/all-MiniLM-L6-v2', 'k': K, 'max_len': MAX_LEN, 'chunk_words': CHUNK_WORDS,
           'stride_unit': 'tiktoken cl100k_base tokens, counted from sentence start (sentence end always judged)',
           'default': DEFAULT, 'threshold': THR[name][DEFAULT['kind']][DEFAULT['threshold']], 'thresholds': THR[name],
           'score': 'logit (higher = hallucinated)', 'weights_dtype': 'float16 (load as float32)',
           'f1': {'reproduce': F1_1[name], 'gateway_parity': F1_2[name], 'int8_ok': F1_3[name]}}
    json.dump(cfg, open(os.path.join(d, 'judge_config.json'), 'w'), indent=1, ensure_ascii=False)

summ = {'F1_1': F1_1, 'F1_2': F1_2, 'F1_3': F1_3, 'deploy': DEPLOY, 'default': DEFAULT, 'drive_weights_ok': DRIVE_OK,
        'reproduce': rep.to_dict('records'), 'parity': PAR.to_dict('records'), 'int8': I8, 'thresholds': THR}
json.dump(summ, open(os.path.join(LOCAL, 'f1_summary.json'), 'w'), default=float, ensure_ascii=False, indent=1)
dev_tab.to_csv(os.path.join(LOCAL, 'f1_devL.csv'), index=False)
LAT.to_csv(os.path.join(LOCAL, 'f1_latency.csv'), index=False)
for name in NAMES:
    to_drive(name)                                   # judge_config·고정 예제 포함 다시 복사
for f in ['f1_summary.json', 'f1_devL.csv', 'f1_latency.csv']:
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(FINAL_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)

zp = shutil.make_archive('/content/f1_final_judges', 'zip', LOCAL)
print(f'zip {os.path.getsize(zp) / 2**20:.0f}MB:', zp)
try:
    shutil.copy(zp, os.path.join(FINAL_DIR, 'f1_final_judges.zip'))
except OSError as e:
    print('zip 드라이브 저장 실패', e)
from google.colab import files
files.download(zp)
print(json.dumps({k: summ[k] for k in ['F1_1', 'F1_2', 'F1_3', 'deploy', 'drive_weights_ok']}, ensure_ascii=False, indent=1, default=float))
""")

md("""## 14. (선택) T4 등 다른 GPU에서 판정 지연만 재기

배포 후보 GPU(T4 등)에서 런타임을 새로 열고 **이 셀 하나만** 실행한다. 드라이브 `final/judge_*`와 고정 예제를 쓴다.""")
code("JUDGE_SRC = " + repr(JUDGE_SRC) + r"""
import os, json, numpy as np, torch, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'sentence-transformers'], check=True)
from google.colab import drive
drive.mount('/content/drive')
open('hallugate_judge.py', 'w').write(JUDGE_SRC); sys.path.insert(0, '.')
from hallugate_judge import StudentJudge
FD = '/content/drive/MyDrive/Colab Notebooks/HalluGate/final'
for name in ['base', 'mini']:
    d = os.path.join(FD, f'judge_{name}')
    fx = json.load(open(os.path.join(d, 'parity_fixture.json')))
    j = StudentJudge(d, k=fx['k'], max_len=fx['max_len'], chunk_words=fx['chunk_words'], device='cuda')
    refs, ms, diff = {}, [], []
    for it in fx['items']:
        if it['reference'] not in refs: refs[it['reference']] = j.prepare(it['reference'])
    for it in fx['items'][:5]:
        j.judge(refs[it['reference']], 'warm up.')
    for _ in range(4):
        for it in fx['items']:
            v = j.judge(refs[it['reference']], it['text'])
            ms.append(v.judge_ms); diff.append(abs(v.score - it['score_cpu_fp32']))
    print(f"{name} @ {torch.cuda.get_device_name(0)}: p50 {np.percentile(ms, 50):.1f} / p95 {np.percentile(ms, 95):.1f} ms | "
          f"CPU fp32 점수와 최대 차이 {max(diff):.3f}")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'f1_final_judge.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
