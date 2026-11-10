"""X1 노트북 생성: 외부 공개 판정기(LettuceDetect·HHEM-2.1·HDM-2) 예비 비교 (dev-L, 보고용).
데이터·평가 세트·정답·평가 함수는 R4C(=F1) 노트북과 같고, F1 base는 저장된 점수(final/scores_base.npy)를 쓴다.
사전 기준은 SSOT §6 X1. 본 비교는 S5 test(이 노트북은 test를 열지 않는다)."""
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
CONF = _I1.DECIDE.split('def compare(')[0]           # conformal_flag, detect_vec (I1·I2와 같은 서비스 10% 보정)

ENV = find(R4C, "!pip -q install pysbd")
ENV = sub(ENV, "!pip -q install pysbd sentence-transformers tiktoken",
          '!pip -q install pysbd tiktoken peft bitsandbytes nltk -U "transformers>=4.48"\n!pip -q install --no-deps hdm2')
ENV = sub(ENV, "/content/drive/MyDrive/selfproject", "/content/drive/MyDrive/Colab Notebooks/HalluGate")
ENV = sub(ENV, "R1_DIR = os.path.join(OUT_DIR, 'r4c')",
          "R1_DIR = X1_DIR = os.path.join(OUT_DIR, 'x1')\nFINAL_DIR = os.path.join(OUT_DIR, 'final')")
ENV = sub(ENV, "print('device', DEVICE)",
          "assert os.path.exists(os.path.join(FINAL_DIR, 'scores_base.npy')), 'final/scores_base.npy 없음 (F1 재실행 점수, 비교 기준)'\n"
          "import shutil\n"
          "torch.backends.cuda.matmul.allow_tf32 = True\n"
          "# HDM-2는 접근 승인이 필요한 모델: HF에서 약관 동의 후 Colab 보안 비밀(열쇠 아이콘)에 HF_TOKEN을 넣는다\n"
          "try:\n"
          "    from google.colab import userdata\n"
          "    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
          "    print('HF_TOKEN 설정됨')\n"
          "except Exception as e:\n"
          "    print('HF_TOKEN 없음 — HDM-2는 제외로 기록된다:', type(e).__name__)\n"
          "print('device', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')")
DATA = find(R4C, "BASE = 'RAGTruth/dataset'")
PREP = find(R4C, "ENC = tiktoken.get_encoding")
CORR = find(R4C, "# R4 보정 정답 C1")
CORR = sub(CORR, "# 체크포인트 선택은 보정 정답 기준 cal-sub (같은 300응답)\n_byid = {x['id']: x for x in CAL_C}\nCALSUB = [_byid[x['id']] for x in CALSUB]", "")
SERIES = find(R4C, "SC = dict(Z0)")
EVAL_DEFS = find(R4C, "def calibrate(items, scorer, arl0):")

LOAD = r"""
# R0 평가 세트(cal-L, dev-L)와 F1 base 저장 점수. 응답 본문·RAGTruth 원 프롬프트는 train 분할에서 찾는다(test는 열지 않음)
import pickle
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
Z0 = dict(np.load(os.path.join(R0_DIR, 'r0_scores.npz')))
CAL, DEV = R0P['items']['calL'], R0P['items']['devL']
for k in {x['ref_key'] for x in CAL + DEV}:
    REFS.setdefault(k, build_reference(sources[k]))
_tr = {str(r['id']): r for r in responses if r['split'] == 'train'}
assert all(x['id'] in _tr for x in CAL + DEV), 'cal-L·dev-L 응답이 train 분할에 없음'
RESP_TEXT = {x['id']: _tr[x['id']]['response'] for x in CAL + DEV}
PROMPT = {x['id']: sources[_tr[x['id']]['source_id']]['prompt'] for x in CAL + DEV}
print(f"cal-L {len(CAL)} 응답 (환각 {sum(x['onset'] is not None for x in CAL)}) | dev-L {len(DEV)} 응답 (환각 {sum(x['onset'] is not None for x in DEV)})")
"""

# ---- X1 전용 셀(X1_COMMON, X1_METRICS, 각 도구의 실행 함수)은 smoke_x1.py가 가짜 데이터로 그대로 실행해 확인한다 ----
X1_COMMON = r"""
S_B = 8                      # 트랙 B 재판정 간격 (배포 설정과 같음)
NO_SCORE = -1.0              # 아직 점수를 낼 토큰이 없는 지점·문장 (확률 도구는 0~1이라 경보가 울리지 않음)

def need8(x, t):
    return t in x['ends'] or (t - x['sent_of'][t] + 1) % S_B == 0

def build_points(items, resp_text):
    # 트랙 B 판정 지점(문장 끝 + 8토큰)마다 새 번호. 같은 응답의 사본(remap8)으로 기존 평가 함수를 그대로 쓴다
    P, MAP = [], {}
    for x in items:
        text = resp_text[x['id']]
        assert len(text) >= x['te'][-1], x['id']
        sidx = {sn['t0']: i for i, sn in enumerate(x['sents'])}
        for t, _ in x['pts2']:
            if not need8(x, t): continue
            MAP[(x['id'], t)] = len(P)
            P.append({'id': x['id'], 't': t, 'si': sidx[x['sent_of'][t]], 'cut': x['te'][t]})
    return P, MAP

def remap8(items, MAP):
    out = []
    for x in items:
        y = copy.copy(x); y['pts2'] = [(t, MAP[(x['id'], t)]) for t, _ in x['pts2'] if (x['id'], t) in MAP]
        out.append(y)
    return out

def to_points(items, name, MAP, n):
    # 기존 질의 번호 점수(F1 base)를 트랙 B 지점 번호로 옮긴다
    a = np.full(n, np.nan, dtype=np.float32)
    for x in items:
        for t, q in x['pts2']:
            if (x['id'], t) in MAP: a[MAP[(x['id'], t)]] = SC[name][q]
    return a

def content_start(text, s, e):
    seg = text[s:e]; k = len(seg) - len(seg.lstrip())
    return s + k if s + k < e else s

def tok_sentence(x, text, spans):
    # 도구 토큰마다 문장 번호 (토큰 내용 시작 위치 기준 — cl100k 토큰의 문장 배정과 같은 규칙). 문장 밖이면 -1
    out = []
    for s, e in spans:
        c = content_start(text, s, e)
        out.append(next((i for i, sn in enumerate(x['sents']) if sn['a'] <= c < sn['b']), -1))
    return out

def sent_scores(x, text, spans, probs):
    # 트랙 A: 문장 점수 = 그 문장 토큰 확률 최댓값
    out = [NO_SCORE] * len(x['sents'])
    for i, p in zip(tok_sentence(x, text, spans), probs):
        if i >= 0: out[i] = max(out[i], float(p))
    return out

def point_scores(x, text, spans, probs, pts):
    # 트랙 B: 지점 점수 = 현재 문장 토큰 중 지점(cut)까지 끝난 토큰의 확률 최댓값.
    # 인과 모델 1회 통과(HDM-2)와, 지점까지 잘라 넣은 입력(LettuceDetect) 모두 이 함수로 계산한다
    cs = [content_start(text, s, e) for s, e in spans]
    out = []
    for q in pts:
        sn = x['sents'][q['si']]
        out.append(max([float(p) for (s, e), c, p in zip(spans, cs, probs) if e <= q['cut'] and sn['a'] <= c < sn['b']] or [NO_SCORE]))
    return out

# 묶음 크기(토큰 수 상한): A100(40GB 이상) 기준값, 그보다 작은 GPU(T4 16GB)는 4분의 1
BIG_GPU = DEVICE != 'cuda' or torch.cuda.get_device_properties(0).total_memory > 30 * 2**30
def budget(big):
    return big if BIG_GPU else big // 4

def length_batches(lengths, max_tokens, max_bs=64):
    # 길이순으로 묶어 묶음마다 (가장 긴 길이 × 개수) ≤ max_tokens
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    b, out = [], []
    for i in order:
        if b and (len(b) + 1) * max(lengths[i], 1) > max_tokens or len(b) >= max_bs:
            out.append(b); b = []
        b.append(i)
    if b: out.append(b)
    return out

def save_json(obj, name):
    p = os.path.join(X1_DIR, name)
    json.dump(obj, open(p, 'w'), default=float)

def load_json(name):
    p = os.path.join(X1_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None

def run_parts(name, n, fn, part=10000):
    # 트랙 B 지점 점수를 조각마다 드라이브에 저장 (끊겨도 이어서)
    out = np.full(n, np.nan, dtype=np.float32); t0 = time.perf_counter()
    for i in range(0, n, part):
        pp = os.path.join(X1_DIR, f'{name}_part{i // part:03d}.npy')
        if os.path.exists(pp):
            v = np.load(pp)
        else:
            v = np.asarray(fn(range(i, min(i + part, n))), dtype=np.float32)
            np.save(pp, v)
        out[i:i + len(v)] = v
        print(f'  {name} {min(i + part, n):,}/{n:,} | {(time.perf_counter() - t0) / 60:.1f}분')
    return out

def latency(fn, inputs, warm=10):
    for z in inputs[:warm]: fn(z)
    ms = []
    for z in inputs:
        t0 = time.perf_counter(); fn(z)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))
"""

X1_LT = r"""
LT_ID, LT_MAX = 'KRLabsOrg/lettucedect-large-modernbert-en-v1', 4096

def lt_load():
    from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
    cfg = AutoConfig.from_pretrained(LT_ID)
    if hasattr(cfg, 'reference_compile'): cfg.reference_compile = False
    tok = AutoTokenizer.from_pretrained(LT_ID)
    model = AutoModelForTokenClassification.from_pretrained(LT_ID, config=cfg, attn_implementation='sdpa').to(DEVICE).eval()
    return tok, model

@torch.no_grad()
def lt_run(tok, model, pairs, max_tokens=None):
    # (RAGTruth 원 프롬프트, 답) 쌍마다 답 토큰의 글자 구간·환각 확률, 프롬프트 잘림 여부.
    # LettuceDetect 학습·추론과 같은 입력: tokenizer(prompt, answer, truncation='only_first', max_length=4096)
    max_tokens = max_tokens or budget(64000)
    lens = [min(LT_MAX, (len(p) + len(a)) // 3 + 8) for p, a in pairs]
    out = [None] * len(pairs)
    for b in length_batches(lens, max_tokens):
        enc = tok([pairs[j][0] for j in b], [pairs[j][1] for j in b], truncation='only_first', max_length=LT_MAX,
                  padding=True, return_offsets_mapping=True, return_tensors='pt')
        seqs = [enc.sequence_ids(k) for k in range(len(b))]
        offs = enc['offset_mapping'].tolist()
        n_tok = enc['attention_mask'].sum(1).tolist()
        z = model(input_ids=enc['input_ids'].to(DEVICE), attention_mask=enc['attention_mask'].to(DEVICE)).logits.float()
        pr = torch.softmax(z, -1)[..., 1].cpu().numpy()
        for k, j in enumerate(b):
            idx = [u for u, s in enumerate(seqs[k]) if s == 1 and offs[k][u][1] > offs[k][u][0]]
            out[j] = ([tuple(offs[k][u]) for u in idx], pr[k, idx], n_tok[k] >= LT_MAX)
    return out
"""

X1_HH = r"""
HH_ID = 'vectara/hallucination_evaluation_model'

def hh_load():
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(HH_ID, trust_remote_code=True).to(DEVICE).eval()

@torch.no_grad()
def hh_run(model, pairs, max_tokens=None):
    # (근거, 문장 또는 문장 앞부분) 쌍마다 환각 점수 = 1 − 일치 확률 (모델 카드의 predict 그대로)
    max_tokens = max_tokens or budget(48000)
    lens = [(len(p) + len(h)) // 3 + 32 for p, h in pairs]
    out = np.zeros(len(pairs), dtype=np.float32)
    for b in length_batches(lens, max_tokens):
        out[b] = 1.0 - model.predict([pairs[j] for j in b]).float().cpu().numpy()
    return out
"""

X1_HD = r"""
HD_ID = 'AimonLabs/hallucination-detection-model'

def hd_load():
    # hdm2 패키지의 로더(백본 + LoRA + 토큰 점수 헤드). 상식(common knowledge) 분류기는 쓰지 않는다.
    # fp32로 불러 X1-0 점검과 채점을 같은 정밀도로 한다
    from hdm2.utils.load_models import load_model_components
    return load_model_components(repo_id=HD_ID, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def hd_run(model, tok, prompt, response):
    # hdm2 _detect_hallucinations와 같은 입력: tokenizer(prompt + ' ' + context, response), context = '' (RAGTruth 원 프롬프트에 근거가 들어 있음)
    enc = tok(prompt + ' ' + '', response, return_tensors='pt', return_offsets_mapping=True)
    start = enc.sequence_ids(0).index(1)
    offs = enc['offset_mapping'][0].tolist()[start:]
    dt = next(model.parameters()).dtype
    dev = model.backbone.device
    op = model(input_ids=enc['input_ids'].to(dev), attention_mask=enc['attention_mask'].to(dev, dtype=dt))
    pr = torch.softmax(op['token_logits'][0].float(), -1)[start:, 1].cpu().numpy()
    keep = [u for u, (s, e) in enumerate(offs) if e > s]
    return [tuple(offs[u]) for u in keep], pr[keep]

def causal_check(run, items, n=20, per=5, seed=3):
    # X1-0: 전체 응답 1회 통과와 지점까지 잘라 넣은 입력에서, 지점까지 끝난 같은 토큰의 확률 차이 최댓값
    rs = random.Random(seed); worst = 0.0; n_cmp = 0
    for x in rs.sample(items, min(n, len(items))):
        text = RESP_TEXT[x['id']]
        full = dict(zip(*run(PROMPT[x['id']], text)))
        for q in rs.sample(PTS_BY[x['id']], min(per, len(PTS_BY[x['id']]))):
            pre = dict(zip(*run(PROMPT[x['id']], text[:q['cut']])))
            for sp, p in pre.items():
                if sp in full and sp[1] < q['cut']:          # 마지막 토큰은 잘린 단어일 수 있어 제외
                    worst = max(worst, abs(float(p) - float(full[sp]))); n_cmp += 1
    return worst, n_cmp
"""

X1_METRICS = r"""
from sklearn.metrics import roc_auc_score

def resp_max(x, S):
    return max(S[x['id']] or [NO_SCORE])

def conformal_resp(cal, S, alpha=0.10):
    mx = np.sort([resp_max(x, S) for x in cal if x['onset'] is None]); n = len(mx); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(mx[k - 1]) + 1e-6 if k <= n else float('inf')

def track_a(lab, name, default_thr):
    # 완성 응답: 응답 AUROC(응답 점수 = 문장 점수 최댓값), 완성 문장 AUROC, 응답 단위 서비스 10%, 도구 기본 임계값
    cal, dev = LAB_SETS[lab]; S = SA[name]
    y = np.array([x['onset'] is not None for x in dev]); s = np.array([resp_max(x, S) for x in dev])
    ys, ss = [], []
    for x in dev:
        for i, sn in enumerate(x['sents']):
            if SENT_OK[x['id']][i]: ys.append(sn['y']); ss.append(S[x['id']][i])
    thr = conformal_resp(cal, S)
    return {'resp_auc': roc_auc_score(y, s), 'sent_auc': roc_auc_score(ys, ss), 'thr_svc': thr,
            'svc_recall': float((s[y] >= thr).mean()), 'svc_clean_flag': float((s[~y] >= thr).mean()),
            'def_thr': default_thr, 'def_recall': float((s[y] >= default_thr).mean()),
            'def_clean_flag': float((s[~y] >= default_thr).mean())}

def boot_auc(lab, a, b, n=2000, seed=2027):
    dev = LAB_SETS[lab][1]; y = np.array([x['onset'] is not None for x in dev])
    sa = np.array([resp_max(x, SA[a]) for x in dev]); sb = np.array([resp_max(x, SA[b]) for x in dev])
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].all() or not y[i].any(): continue
        d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    return roc_auc_score(y, sa) - roc_auc_score(y, sb), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def track_b(lab, name):
    # 스트리밍: 문장 끝 + 8토큰 지점, 서비스 10%(엄격 탐지·지연)와 ARL₀=100 엄격 재현율
    cal, dev = LAB_SETS8[lab]; sc = f'{name}|8'
    thr = conformal_flag(cal, sc); det, dl = detect_vec(dev, sc, thr)
    clean = float(np.mean([any(s >= thr for _, s in series(x, sc)) for x in dev if x['onset'] is None]))
    st = evaluate(dev, sc, calibrate(cal, sc, 100))
    return {'thr_svc': thr, 'svc_recall': float(det.mean()), 'svc_delay_med': float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
            'svc_clean_flag': clean, 'strict_recall_arl100': st['recall'], 'strict_delay_med': st['delay_tok_median'],
            'realized_arl0': st['realized_ARL0_dev']}, det

def boot_det(da, db, n=2000, seed=2027):
    rng = np.random.default_rng(seed); N = len(da)
    d = [(da[i].mean() - db[i].mean()) for i in (rng.integers(0, N, N) for _ in range(n))]
    return float(da.mean() - db.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def verdict(lo, hi):
    return '앞섬' if lo > 0 else ('뒤짐(오염 영향 분리 불가)' if hi < 0 else '차이 없음')
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# X1 — 외부 공개 판정기 예비 비교 (LettuceDetect · HHEM-2.1 · HDM-2, dev-L)

## 왜 하나

지금까지 직접 비교한 상대는 MiniCheck와 NLI 모델뿐이다(3주차·S2). 최근 공개된 RAG 환각 판정기 세 개와 우리 배포 판정기(F1 base)를 **같은 응답·같은 정답·같은 보정 방식**으로 비교한다.
세 도구는 RAGTruth train으로 학습됐거나(LettuceDetect 확인, HDM-2 2차 출처) 학습 데이터가 공개되지 않았다(HHEM-2.1). dev-L은 RAGTruth train에서 떼어 낸 세트라 **상대에게 유리한 조건**이다. 공정한 본 비교는 S5에서 test로 한 번 한다. 이 노트북은 test를 열지 않는다.

## 무엇을 하나

| | 입력 | 점수 |
|---|---|---|
| F1 base (우리, 배포) | 근거 조각 top-8 + 문장 앞부분 | 저장 점수 `final/scores_base.npy` (logit) |
| LettuceDetect-large v1 (ModernBERT, 0.4B) | RAGTruth 원 프롬프트 + 응답 (학습 형식, 4,096토큰) | 토큰 환각 확률 → 문장 안 최댓값 |
| HHEM-2.1-open (flan-t5-base) | (근거, 문장 또는 문장 앞부분) | 1 − 일치 확률 |
| HDM-2-3B (디코더 + LoRA) | RAGTruth 원 프롬프트 + 응답 | 문맥 환각 토큰 확률 → 문장 안 최댓값 (상식 분류기 미사용) |

- **트랙 A (완성 응답, 상대의 본래 사용법)**: 응답을 다 받은 뒤 판정. 응답 AUROC, 완성 문장 AUROC, 응답 단위 서비스 10% 탐지율, 도구 기본 임계값(0.5)에서의 표시율·탐지율. F1 base는 문장 끝 점수.
- **트랙 B (스트리밍, 우리 서비스 조건)**: 문장 끝 + 8토큰 지점마다 **그 지점까지의 응답만** 보고 판정. LettuceDetect는 양방향 모델이라 지점마다 잘라 넣고, HHEM은 문장 앞부분으로, HDM-2는 디코더(앞만 보는 모델)라 1회 통과 점수를 쓴다(X1-0으로 확인). 서비스 10% 엄격 탐지율·지연과 ARL₀=100 엄격 재현율.

## 사전 정의 기준 (결과 보기 전, SSOT §6 X1에 고정)

- 임계값은 cal-L, 수치는 dev-L. 원래 정답이 주, C1 보조.
- **X1-0**: HDM-2 1회 통과와 지점까지 잘라 넣은 입력에서 같은 토큰 확률의 최대 차이 ≤ 0.01 (20응답). 미달이면 HDM-2 트랙 B는 보고하지 않는다.
- **해석 규칙**: 트랙 A 응답 AUROC와 트랙 B 8토큰 서비스 탐지율 각각, F1 base − 상대의 응답 단위 쌍체 부트스트랩(2,000회) 95% 하한 > 0이면 "상대에게 유리한 조건에서도 앞섬", 상한 < 0이면 "dev-L에서는 뒤짐(오염 영향 분리 불가)", 그 밖은 "차이 없음".
- 모델 접근·로드가 실패하면 그 도구는 제외로 기록한다(기준 변경 없음). **이 결과로 판정기·임계값을 바꾸지 않는다.**

> **런타임: A100** (40GB 이상). T4(16GB)도 되지만 묶음 크기가 자동으로 4분의 1이 되어 느리고, 판정 지연 수치는 T4 기준이 된다. 드라이브에 `r0/`, `r4/`, `r4b/labels_train.json`, `final/scores_base.npy`.
> HDM-2는 접근 승인이 필요하다: [모델 페이지](https://huggingface.co/AimonLabs/hallucination-detection-model)에서 약관에 동의하고, Colab 보안 비밀에 `HF_TOKEN`을 넣는다(없으면 HDM-2만 제외).
> HDM-2 라이선스는 CC BY-NC-SA 4.0(비상업)이라 연구 비교에만 쓴다. 결과는 드라이브 `Colab Notebooks/HalluGate/x1/`와 `x1_results.zip`.
""")

md("## 1. 환경 준비")
code(ENV)
md("## 2. 데이터와 평가 세트 (F1과 같음)")
code(DATA)
code(PREP)
code(LOAD)
code(CORR)
md("## 3. 평가 함수 (F1·I2와 같음)")
code(SERIES)
code(EVAL_DEFS)
code(CONF)

md("""## 4. 판정 지점과 F1 base 점수

트랙 B 지점(문장 끝 + 8토큰)마다 새 번호를 매긴 응답 사본을 만들고, F1 base 저장 점수를 그 번호로 옮긴다. 트랙 A의 F1 base 문장 점수는 문장 끝 점수다.""")
code(X1_COMMON + r"""
SC['F_base'] = np.load(os.path.join(FINAL_DIR, 'scores_base.npy'))
PTS, MAP8 = build_points(CAL + DEV, RESP_TEXT)
PTS_BY = {}
for q in PTS: PTS_BY.setdefault(q['id'], []).append(q)
LAB_SETS8 = {lab: (remap8(c, MAP8), remap8(d, MAP8)) for lab, (c, d) in LAB_SETS.items()}
SC['Fb'] = to_points(CAL + DEV, 'F_base', MAP8, len(PTS))
assert np.isfinite(SC['Fb']).all(), 'F1 base 점수가 없는 지점이 있음'
print(f'트랙 B 지점 {len(PTS):,}개 (응답당 {len(PTS) / len(CAL + DEV):.1f})')

_chk = conformal_flag(LAB_SETS8['orig'][0], 'Fb|8')
assert abs(_chk - 3.968751) < 1e-3, f'F1 재실행 8토큰 임계값 재현 실패: {_chk}'
print(f'F1 base 8토큰 cal-L 서비스 임계값 재현: {_chk:.4f}')

SA, SENT_OK, AVAIL, LAT = {}, {}, {}, {}
for x in CAL + DEV:
    d = dict(x['pts2'])
    SENT_OK[x['id']] = [sn['t1'] in d for sn in x['sents']]          # 문장 AUROC는 F1 base 점수가 있는 문장만 (모든 도구 공통)
SA['Fb'] = {x['id']: [float(SC['F_base'][dict(x['pts2'])[sn['t1']]]) if ok else NO_SCORE
                      for sn, ok in zip(x['sents'], SENT_OK[x['id']])] for x in CAL + DEV}
AVAIL['Fb'] = {'A': True, 'B': True, 'note': '저장 점수'}
LAT['Fb'] = {'note': 'F1 재실행 A100 80GB p95 40.7ms (SSOT §5 F1)'}
print('문장 AUROC 대상 문장', sum(sum(v) for v in SENT_OK.values()))
""")

md("""## 5. LettuceDetect-large v1

트랙 A는 응답마다 1회, 트랙 B는 지점마다 그 지점까지 잘라 넣는다(양방향 모델이라 뒤 토큰을 보지 않게). 점수는 조각마다 저장한다.""")
code(X1_LT + r"""
try:
    lt_tok, lt_model = lt_load()
    ALL = CAL + DEV
    sa = load_json('lt_sents.json')
    if sa is None:
        res = lt_run(lt_tok, lt_model, [(PROMPT[x['id']], RESP_TEXT[x['id']]) for x in ALL])
        sa = {x['id']: sent_scores(x, RESP_TEXT[x['id']], sp, pr) for x, (sp, pr, _) in zip(ALL, res)}
        sa['_trunc'] = float(np.mean([tr for _, _, tr in res]))
        save_json(sa, 'lt_sents.json')
    LT_TRUNC = sa.pop('_trunc'); SA['LT'] = sa
    print(f'트랙 A 완료 | 프롬프트 잘림 {LT_TRUNC:.2%}')

    BY = {x['id']: x for x in ALL}
    def lt_points(idx):
        qs = [PTS[i] for i in idx]
        res = lt_run(lt_tok, lt_model, [(PROMPT[q['id']], RESP_TEXT[q['id']][:q['cut']]) for q in qs])
        return [point_scores(BY[q['id']], RESP_TEXT[q['id']], sp, pr, [q])[0] for q, (sp, pr, _) in zip(qs, res)]
    SC['LT'] = run_parts('lt', len(PTS), lt_points)
    LQ = random.Random(2027).sample(PTS, 200)
    LAT['LT'] = dict(zip(['p50', 'p95'], latency(lambda q: lt_run(lt_tok, lt_model, [(PROMPT[q['id']], RESP_TEXT[q['id']][:q['cut']])]), LQ)))
    AVAIL['LT'] = {'A': True, 'B': True, 'note': f'프롬프트 잘림 {LT_TRUNC:.2%}'}
    print('LettuceDetect 판정 1회', LAT['LT'])
    del lt_model; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    import traceback; traceback.print_exc()
    AVAIL['LT'] = {'A': 'LT' in SA, 'B': False, 'note': f'실패: {type(e).__name__}: {e}'}
""")

md("""## 6. HHEM-2.1-open

트랙 A는 (근거, 문장), 트랙 B는 (근거, 문장 앞부분). 같은 쌍은 한 번만 채점한다.""")
code(X1_HH + r"""
try:
    hh = hh_load()
    ALL = CAL + DEV
    sa = load_json('hh_sents.json')
    if sa is None:
        pairs, where = [], []
        for x in ALL:
            text = RESP_TEXT[x['id']]
            for i, sn in enumerate(x['sents']):
                h = text[sn['a']:sn['b']].strip()
                if h: pairs.append((REFS[x['ref_key']], h)); where.append((x['id'], i))
        v = hh_run(hh, pairs)
        sa = {x['id']: [NO_SCORE] * len(x['sents']) for x in ALL}
        for (cid, i), s in zip(where, v): sa[cid][i] = float(s)
        save_json(sa, 'hh_sents.json')
    SA['HH'] = sa
    print('트랙 A 완료')

    BY = {x['id']: x for x in ALL}
    def hh_points(idx):
        qs = [PTS[i] for i in idx]
        hyp = [RESP_TEXT[q['id']][BY[q['id']]['sents'][q['si']]['a']:q['cut']].strip() for q in qs]
        uniq = sorted({(BY[q['id']]['ref_key'], h) for q, h in zip(qs, hyp) if h})
        v = dict(zip(uniq, hh_run(hh, [(REFS[k], h) for k, h in uniq])))
        return [float(v[(BY[q['id']]['ref_key'], h)]) if h else NO_SCORE for q, h in zip(qs, hyp)]
    SC['HH'] = run_parts('hh', len(PTS), hh_points)
    LQ = random.Random(2027).sample(PTS, 200)
    LAT['HH'] = dict(zip(['p50', 'p95'], latency(
        lambda q: hh_run(hh, [(REFS[BY[q['id']]['ref_key']], RESP_TEXT[q['id']][BY[q['id']]['sents'][q['si']]['a']:q['cut']].strip() or '.')]), LQ)))
    AVAIL['HH'] = {'A': True, 'B': True, 'note': ''}
    print('HHEM 판정 1회', LAT['HH'])
    del hh; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    import traceback; traceback.print_exc()
    AVAIL['HH'] = {'A': 'HH' in SA, 'B': False, 'note': f'실패: {type(e).__name__}: {e}'}
""")

md("""## 7. HDM-2-3B

디코더라 토큰 확률이 앞 토큰에만 의존한다. 먼저 X1-0으로 확인하고, 응답마다 1회 통과해 트랙 A·B 점수를 함께 만든다.""")
code(X1_HD + r"""
X1_0 = None
try:
    hd_model, hd_tok = hd_load()
    ALL = CAL + DEV
    run = lambda p, r: hd_run(hd_model, hd_tok, p, r)
    worst, n_cmp = causal_check(run, ALL)
    X1_0 = worst <= 0.01
    print(f'=== X1-0 인과성 점검: 최대 차이 {worst:.4g} ({n_cmp:,}개 토큰 비교) → {"충족" if X1_0 else "미달 → HDM-2 트랙 B 미보고"} ===')

    sa = load_json('hd_sents.json'); pb = os.path.join(X1_DIR, 'hd_points.npy')
    if sa is None or not os.path.exists(pb):
        sa, pts_sc, t0 = {}, np.full(len(PTS), np.nan, dtype=np.float32), time.perf_counter()
        for n, x in enumerate(ALL):
            text = RESP_TEXT[x['id']]
            sp, pr = run(PROMPT[x['id']], text)
            sa[x['id']] = sent_scores(x, text, sp, pr)
            for q, v in zip(PTS_BY[x['id']], point_scores(x, text, sp, pr, PTS_BY[x['id']])):
                pts_sc[MAP8[(x['id'], q['t'])]] = v
            if (n + 1) % 500 == 0: print(f'  HDM-2 {n + 1:,}/{len(ALL):,} | {(time.perf_counter() - t0) / 60:.1f}분')
        save_json(sa, 'hd_sents.json'); np.save(pb, pts_sc)
    SA['HD'] = sa
    if X1_0: SC['HD'] = np.load(pb)
    LQ = random.Random(2027).sample(ALL, 100)
    LAT['HD'] = dict(zip(['p50', 'p95'], latency(lambda x: run(PROMPT[x['id']], RESP_TEXT[x['id']]), LQ)))
    LAT['HD']['note'] = '응답 전체 1회 통과(배치 1). 스트리밍 서비스에서는 KV 캐시로 토큰마다 이어 계산해야 함'
    AVAIL['HD'] = {'A': True, 'B': bool(X1_0), 'note': f'X1-0 최대 차이 {worst:.4g}'}
    print('HDM-2 응답 1회 통과', LAT['HD'])
    del hd_model; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    import traceback; traceback.print_exc()
    AVAIL['HD'] = {'A': 'HD' in SA, 'B': False, 'note': f'실패: {type(e).__name__}: {e}'}
""")

md("""## 8. 결과와 사전 규칙 판정 (dev-L, 임계값 cal-L)""")
code(X1_METRICS + r"""
NAMES = {'Fb': 'F1 base (우리)', 'LT': 'LettuceDetect-large', 'HH': 'HHEM-2.1-open', 'HD': 'HDM-2-3B'}
DEFAULT_THR = {'Fb': 3.968751, 'LT': 0.5, 'HH': 0.5, 'HD': 0.5}    # F1 base는 배포 임계값
ROWS_A, ROWS_B, DET, CMP = [], [], {}, []
for lab in ['orig', 'C1']:
    for m in NAMES:
        if AVAIL.get(m, {}).get('A') and m in SA:
            ROWS_A.append({'labels': lab, 'model': NAMES[m], **track_a(lab, m, DEFAULT_THR[m])})
        if AVAIL.get(m, {}).get('B') and m in SC:
            r, det = track_b(lab, m); DET[(lab, m)] = det
            ROWS_B.append({'labels': lab, 'model': NAMES[m], **r})
    for m in ['LT', 'HH', 'HD']:
        if AVAIL.get(m, {}).get('A') and m in SA:
            d, lo, hi = boot_auc(lab, 'Fb', m)
            CMP.append({'labels': lab, 'vs': NAMES[m], 'metric': '트랙 A 응답 AUROC', 'diff': d, 'lo': lo, 'hi': hi, 'verdict': verdict(lo, hi)})
        if (lab, m) in DET:
            d, lo, hi = boot_det(DET[(lab, 'Fb')], DET[(lab, m)])
            CMP.append({'labels': lab, 'vs': NAMES[m], 'metric': '트랙 B 8토큰 서비스 탐지율', 'diff': d, 'lo': lo, 'hi': hi, 'verdict': verdict(lo, hi)})
tab_a, tab_b, tab_c = pd.DataFrame(ROWS_A), pd.DataFrame(ROWS_B), pd.DataFrame(CMP)
pd.set_option('display.width', 250)
print('■ 트랙 A — 완성 응답'); print(tab_a.round(3).to_string(index=False))
print('\n■ 트랙 B — 스트리밍 (문장 끝 + 8토큰)'); print(tab_b.round(3).to_string(index=False))
print('\n■ 사전 규칙 판정 (F1 base − 상대, 응답 단위 쌍체 부트스트랩 95%)'); print(tab_c.round(3).to_string(index=False))
print('\n■ 판정 1회 지연'); print(json.dumps(LAT, ensure_ascii=False, indent=1, default=float))
print('■ 도구 상태'); print(json.dumps(AVAIL, ensure_ascii=False, indent=1))

# 과제별 (원래 정답, 트랙 A 응답 AUROC · 트랙 B 서비스 탐지율)
TASK_ROWS = []
for m in NAMES:
    for task in TASKS:
        row = {'model': NAMES[m], 'task': task}
        dev_t = [x for x in LAB_SETS['orig'][1] if x['task_type'] == task]
        if m in SA and AVAIL.get(m, {}).get('A'):
            y = [x['onset'] is not None for x in dev_t]
            row['resp_auc'] = roc_auc_score(y, [resp_max(x, SA[m]) for x in dev_t])
        if ('orig', m) in DET:
            thr = conformal_flag(LAB_SETS8['orig'][0], f'{m}|8')
            det, _ = detect_vec([x for x in LAB_SETS8['orig'][1] if x['task_type'] == task], f'{m}|8', thr)
            row['svc_recall_8'] = det.mean()
        TASK_ROWS.append(row)
tab_t = pd.DataFrame(TASK_ROWS)
print('\n■ 과제별 (원래 정답)'); print(tab_t.round(3).to_string(index=False))
""")

md("## 9. 저장과 zip 다운로드")
code(r"""
LOCAL = '/content/x1_local'; os.makedirs(LOCAL, exist_ok=True)
for name, t in [('x1_trackA.csv', tab_a), ('x1_trackB.csv', tab_b), ('x1_compare.csv', tab_c), ('x1_tasks.csv', tab_t)]:
    t.to_csv(os.path.join(LOCAL, name), index=False)
summ = {'X1_0': X1_0, 'avail': AVAIL, 'latency': LAT, 'n_points': len(PTS), 'compare': CMP}
json.dump(summ, open(os.path.join(LOCAL, 'x1_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL):
    try: shutil.copy(os.path.join(LOCAL, f), os.path.join(X1_DIR, f))
    except OSError as e: print('드라이브 저장 실패', f, e)
zp = shutil.make_archive('/content/x1_results', 'zip', LOCAL)
print('zip:', zp)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'x1_external_judges.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
