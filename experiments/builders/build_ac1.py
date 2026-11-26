"""AC1 노트북 생성: 자동 교정 — 판정기 v1.0이 경고한 문장을 짚어 LLM이 고쳐 쓰면 환각 응답이 줄어드는가,
그리고 판정기의 경고가 "모든 응답을 다시 검토"보다 싸게 같은 효과를 내는가 (SSOT §6 AC1, 11/26 고정).
데이터는 RAGTruth dev-L 원본 응답(train 분할)만. 판정 코드는 L3·M1, 충실성(R4)·정답성(T1b) 평가는 기존 코드를 그대로 쓴다.
RAGTruth test 분할은 열지 않는다."""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec); argv = sys.argv; sys.argv = ['import']
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


M1 = _load('build_m1')
TB = _load('build_t1b')
L3, T1 = M1.L3, TB.T1
SRC = {'judge': M1.SRC['judge']}
V1 = json.load(open(os.path.join(ROOT, 'configs', 'judge_combo_v1.json'), encoding='utf-8'))
COMBO = {'F_mean': V1['standardize']['F']['mean'], 'F_std': V1['standardize']['F']['std'],
         'I_mean': V1['standardize']['I']['mean'], 'I_std': V1['standardize']['I']['std'], 'threshold': V1['threshold']}
assert all(abs(COMBO[k] - L3.COMBO[k]) < 1e-9 for k in ['F_mean', 'F_std', 'I_mean', 'I_std'])     # I3 상수와 같음
assert abs(COMBO['threshold'] - 2.7385038673533924) < 1e-12                                       # v1.0 임계값 B

# T1b 정답성 평가(지시문·요청·파싱)를 그대로
_g = TB.T1B_GEN
GRADE = _g[_g.index('GRADE_SYSTEM'): _g.index('def judge_requests_b')]
assert 'def grade_params' in GRADE and 'def parse_grade' in GRADE

# ---- 아래 AC1_* 문자열은 smoke_ac1.py가 가짜 데이터로 그대로 실행해 확인한다 ----
AC1_DATA = r"""
SEED_AC, PER_TASK = 2040, 150                       # 사전 고정 (SSOT §6 AC1)
TASKS = ['QA', 'Summary', 'Data2txt']

def sample_items(r0_dev, responses, sources):
    # dev-L(R0) 원본 응답에서 과제별 150개. 전부 train 분할인지 확인
    by_id = {str(r['id']): r for r in responses}
    rs = random.Random(SEED_AC)
    items = []
    for t in TASKS:
        ids = sorted({str(x['id']) for x in r0_dev if x['task_type'] == t})
        for rid in rs.sample(ids, min(PER_TASK, len(ids))):
            r = by_id[rid]; src = sources[r['source_id']]
            assert r['split'] == 'train', f'test 분할 응답이 섞임: {rid}'
            si = src['source_info'] if isinstance(src['source_info'], dict) else {}
            items.append({'rid': rid, 'sid': r['source_id'], 'task': t, 'gen_model': r.get('model'), 'text': r['response'],
                          'ref': build_reference(src),
                          'q': si.get('question', '') if t == 'QA' else None, 'passages': si.get('passages', '') if t == 'QA' else None})
    assert len({x['rid'] for x in items}) == len(items)
    return items
"""

AC1_FLAG = r"""
MARK = '[CHECK]'

def flagged_starts(pts, thr):
    # 경보 지점(조합 점수 ≥ 임계값)이 하나라도 있는 문장의 시작 글자 위치 = 경고 문장
    return sorted({int(p[0]) for p in pts if p[4] >= thr})

def mark(text, starts):
    # 경고 문장 앞에 [CHECK] 표시 (본문 글자는 그대로)
    out, last = [], 0
    for a in sorted(starts):
        out.append(text[last:a]); out.append(MARK + ' '); last = a
    out.append(text[last:])
    return ''.join(out)
"""

AC1_REWRITE = r"""
REWRITER, REWRITE_MAX, LABELER = 'claude-haiku-5-5', 2048, 'claude-opus-5-5'
_KEEP = '''Keep the original format (line breaks, lists), tone, and length as much as possible. Do not add any information that is not in the REFERENCE.
Output ONLY the revised response text, with no explanation, heading, or quotation marks.'''
TGT_SYSTEM = '''You revise a response so that every statement is supported by the REFERENCE.
In the RESPONSE, some sentences are marked with [CHECK]: an automatic checker found that they may contain information that is not supported by, or contradicts, the REFERENCE.
For each marked sentence, compare it with the REFERENCE. If it contains unsupported or contradicting content, correct that content using the REFERENCE or remove it. If the marked sentence is actually fully supported, keep it unchanged.
Do not change unmarked sentences except for minimal grammar fixes needed after your edits. Remove all [CHECK] marks.
''' + _KEEP
BLIND_SYSTEM = '''You revise a response so that every statement is supported by the REFERENCE.
Compare each sentence of the RESPONSE with the REFERENCE. If a sentence contains information that is not supported by, or contradicts, the REFERENCE, correct that content using the REFERENCE or remove it. Keep supported sentences unchanged.
''' + _KEEP

def rw_msg(it, body):
    return f"TASK: {TASK_DESC[it['task']]}\n\nREFERENCE:\n<<<\n{it['ref']}\n>>>\n\nRESPONSE:\n<<<\n{body}\n>>>"

def rw_params(system, it, body):
    return {'model': REWRITER, 'max_tokens': REWRITE_MAX, 'system': system, 'messages': [{'role': 'user', 'content': rw_msg(it, body)}]}

def rewrite_requests(items, flags):
    # tgt: 경고 응답만(표시해서) / blind: 모든 응답(표시 없이). 판정이 없는 응답은 빼고
    R = {}
    for it in items:
        st = flags.get(it['rid'])
        if st is None: continue
        if st: R[f"t-{it['rid']}"] = rw_params(TGT_SYSTEM, it, mark(it['text'], st))
        R[f"b-{it['rid']}"] = rw_params(BLIND_SYSTEM, it, it['text'])
    return R

def clean_rewrite(g):
    # 고쳐 쓴 글 (잘림·거절·빈 글은 None). 남은 [CHECK]와 감싼 <<< >>>는 지운다
    if not g or g.get('stop') in ('max_tokens', 'refusal'): return None
    t = g['text'].replace(MARK + ' ', '').replace(MARK, '').strip()
    if t.startswith('<<<'): t = t[3:]
    if t.endswith('>>>'): t = t[:-3]
    t = t.strip()
    return t if t and sentences(t) else None
"""

AC1_EVAL = r"""
import hashlib
ARMS = ['orig', 'tgt', 'blind']

LOSS_SYSTEM = '''You compare an ORIGINAL response with a REVISED version of it. The REFERENCE is the only source of truth.
Question: does the REVISED version remove or weaken any information that appears in the ORIGINAL and IS supported by the REFERENCE?
- Removing or correcting information that is NOT supported by, or contradicts, the REFERENCE is NOT a loss.
- Rewording that keeps the meaning is NOT a loss.
Return ONLY a JSON object: {"loss": "yes" | "no", "reason": "at most 20 words"}'''

def loss_params(it, orig, rev):
    u = (f"TASK: {TASK_DESC[it['task']]}\n\nREFERENCE:\n<<<\n{it['ref']}\n>>>\n\n"
         f"ORIGINAL:\n<<<\n{orig}\n>>>\n\nREVISED:\n<<<\n{rev}\n>>>")
    return {'model': LABELER, 'max_tokens': 300, 'system': LOSS_SYSTEM, 'messages': [{'role': 'user', 'content': u}]}

def parse_loss(text):
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        v = str(d.get('loss', '')).strip().lower()
        return {'yes': True, 'no': False}.get(v)
    except Exception:
        return None

def arm_texts(items, flags, RW):
    # 응답마다 세 팔의 글. 고쳐 쓰기가 실패한 응답은 뺀다 (경고 없는 응답의 tgt = 원본)
    T = {}
    for it in items:
        st = flags.get(it['rid'])
        if st is None: continue
        b = clean_rewrite(RW.get(f"b-{it['rid']}"))
        t = clean_rewrite(RW.get(f"t-{it['rid']}")) if st else it['text']
        if b is None or t is None: continue
        T[it['rid']] = {'orig': it['text'], 'tgt': t, 'blind': b}
    return T

def hkey(kind, rid, text):
    # 같은 응답의 같은 글은 한 번만 평가 (어느 팔인지는 평가 요청에 드러나지 않음)
    return f"{kind}-{rid}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"

def eval_requests(items, T):
    F, Q, L = {}, {}, {}
    by = {it['rid']: it for it in items}
    for rid, arms in T.items():
        it = by[rid]; case = {'task': it['task'], 'ref_label': it['ref']}
        for a in ARMS:
            txt = arms[a]
            F[hkey('f', rid, txt)] = label_params(case, txt)
            if it['task'] == 'QA':
                Q[hkey('q', rid, txt)] = grade_params({'q': it['q'], 'own': it['passages']}, txt)
            if a != 'orig' and txt != arms['orig']:
                L[hkey('l', rid, txt)] = loss_params(it, arms['orig'], txt)
    return F, Q, L

def hal_of(FR, key, text):
    r = FR.get(key)
    if r is None: return None
    labs, ok = parse(r['text'], len(sentences(text)))
    return ('H' in [v['label'] for v in labs.values()]) if ok else None
"""

AC1_METRICS = r"""
def build_rows(items, flags, T, FR, QR, LR, rescore=None):
    rows = []
    for it in items:
        rid = it['rid']; st = flags.get(rid)
        row = {'rid': rid, 'task': it['task'], 'gen_model': it['gen_model'], 'scored': st is not None,
               'flagged': bool(st), 'n_flag_sent': len(st or []), 'n_sent': len(sentences(it['text'])), 'in_T': rid in T}
        if rid in T:
            arms = T[rid]
            for a in ARMS:
                txt = arms[a]
                row[f'hal_{a}'] = hal_of(FR, hkey('f', rid, txt), txt)
                row[f'len_{a}'] = len(txt)
                row[f'changed_{a}'] = txt != arms['orig']
                if it['task'] == 'QA':
                    q = QR.get(hkey('q', rid, txt)); row[f'grade_{a}'] = parse_grade(q['text']) if q else None
                if a != 'orig':
                    if txt == arms['orig']:
                        row[f'loss_{a}'] = False                     # 바뀌지 않으면 손실 없음
                    else:
                        l = LR.get(hkey('l', rid, txt)); row[f'loss_{a}'] = parse_loss(l['text']) if l else None
            if rescore is not None and rid in rescore:
                row['reflag_tgt'] = rescore[rid]
        rows.append(row)
    D = pd.DataFrame(rows)
    need = [f'hal_{a}' for a in ARMS] + ['loss_tgt', 'loss_blind']
    ok = D['in_T'].astype(bool) & D.reindex(columns=need).notna().all(axis=1)
    qa = (D.task == 'QA').values
    gq = D.reindex(columns=[f'grade_{a}' for a in ARMS]).notna().all(axis=1).values
    D['valid'] = ok.values & (~qa | gq)
    return D

def boot(x, n=2000, seed=SEED_AC):
    # 응답 단위 쌍체 부트스트랩 95% 구간
    x = np.asarray(x, float)
    if len(x) == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    vals = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def judge_ac1(D):
    d = D[D.valid].copy(); n = len(d)
    h = {a: d[f'hal_{a}'].astype(float) for a in ARMS}
    p = {a: float(h[a].mean()) if n else float('nan') for a in ARMS}
    out = {'n': n, 'n_items': len(D), 'n_flagged_valid': int(d.flagged.sum()), 'n_flagged_all': int(D.flagged.sum()), 'hal': p}
    for a in ['tgt', 'blind']:
        x = h[a] - h['orig']
        out[f'diff_{a}'], out[f'diff_{a}_ci'] = float(x.mean()), boot(x)
        out[f'rel_{a}'] = out[f'diff_{a}'] / p['orig'] if p['orig'] > 0 else float('nan')
    q = d[d.task == 'QA']
    out['qa_n'] = len(q)
    yes = {a: (q[f'grade_{a}'] == 'yes').astype(float) for a in ARMS}
    out['yes'] = {a: float(yes[a].mean()) if len(q) else float('nan') for a in ARMS}
    for a in ['tgt', 'blind']:
        x = yes[a] - yes['orig']
        out[f'yes_diff_{a}'], out[f'yes_diff_{a}_ci'] = float(x.mean()) if len(q) else float('nan'), boot(x)
    for a in ['tgt', 'blind']:
        ch = d[d[f'changed_{a}'].astype(bool)]
        out[f'n_changed_{a}'] = len(ch)
        out[f'loss_{a}'] = float(ch[f'loss_{a}'].astype(float).mean()) if len(ch) else 0.0
    out['rewrite_frac'] = float(d.flagged.mean()) if n else float('nan')
    out['gain_tgt'], out['gain_blind'] = p['orig'] - p['tgt'], p['orig'] - p['blind']
    out['AC1_0'] = bool(n >= 405 and out['n_flagged_valid'] >= 60)
    out['AC1_1'] = bool(out['rel_tgt'] <= -0.15 and out['diff_tgt_ci'][1] < 0)
    out['AC1_2'] = bool(out['yes_diff_tgt'] >= -0.05 and out['yes_diff_tgt_ci'][0] >= -0.10 and out['loss_tgt'] <= 0.15)
    out['AC1_3'] = bool(out['rewrite_frac'] <= 0.30 and out['gain_tgt'] >= 0.5 * out['gain_blind'])
    out['AC1'] = out['AC1_0'] and out['AC1_1'] and out['AC1_2'] and out['AC1_3']
    # 보고만
    out['by_task'] = {t: {a: float(g[f'hal_{a}'].astype(float).mean()) for a in ARMS} | {'n': len(g), 'flag_frac': float(g.flagged.mean())}
                      for t, g in d.groupby('task')}
    out['hal_orig_flagged'] = float(d[d.flagged].hal_orig.astype(float).mean()) if out['n_flagged_valid'] else float('nan')
    out['hal_orig_unflagged'] = float(d[~d.flagged].hal_orig.astype(float).mean()) if n - out['n_flagged_valid'] else float('nan')
    out['len_ratio'] = {a: float((d[f'len_{a}'] / d.len_orig).median()) for a in ['tgt', 'blind']}
    if 'reflag_tgt' in d:
        r = d[d.flagged & d.changed_tgt.astype(bool)].reflag_tgt.dropna()
        out['reflag_tgt'] = float(r.astype(float).mean()) if len(r) else float('nan')
    return out
"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# AC1 — 자동 교정: 판정기가 경고한 문장을 LLM이 고쳐 쓰면 환각이 줄어드는가

## 왜 하나

지금 게이트웨이는 환각을 **경고만** 하고 응답을 바꾸지 않는다. 경고가 서비스 가치가 되려면 "경고 → 고치기"까지 이어져야 한다.
두 가지를 확인한다. ① 판정기가 짚은 문장을 LLM이 근거와 대조해 고쳐 쓰면 **환각 응답이 실제로 줄어드는가**.
② 그 효과가 "모든 응답을 다시 검토시키기"와 비교해 **판정기 덕분에 싸게** 얻어지는가(판정기의 가치).

## 무엇을 하나

- **응답**: RAGTruth **dev-L 원본 응답**(train 분할, 판정기 학습·보정에 쓰이지 않은 근거) 과제별 150개 = 450개, `Random(2040)`
- **판정**: 배포 판정기 **v1.0**(F1 base + I2 조합, 8토큰 + 문장 끝, **임계값 B 2.7385**). 경보 지점이 있는 문장 = 경고 문장
- **세 팔**
  - `orig`: 원본 그대로
  - `tgt`: **경고 응답만** `claude-haiku-5-5`가 고쳐 씀 — 경고 문장 앞에 `[CHECK]` 표시, 그 문장만 근거와 대조해 고치거나 뺌
  - `blind`: **모든 응답**을 같은 모델이 표시 없이 "근거와 대조해 근거 없는 내용을 고쳐라"로 고쳐 씀 (대조군)
- **평가**(`claude-opus-5-5`, 어느 팔인지 가림, 같은 글은 한 번만): 충실성(R4 문장 S/H/A → H 하나라도 있으면 환각 응답), 정답성(QA만, T1b yes/partial/no), 정보 손실(고친 글만, "근거가 뒷받침하는 원본 정보를 빼거나 약화했나")

## 사전 정의 기준 (SSOT §6 AC1, 11/26 고정)

- **AC1-0 유효성**: 세 팔 모두 고쳐 쓰기·평가 성공 응답 ≥ 405, 경고 응답 ≥ 60. 미달이면 판정 보류
- **AC1-1 (주, 효과)**: 환각 응답 비율 상대 변화 (tgt − orig)/orig ≤ **−15%**, 그리고 차이의 95% 상한 < 0
- **AC1-2 (부작용)**: QA `yes` tgt − orig ≥ −0.05(하한 ≥ −0.10), 그리고 tgt에서 고친 응답 중 정보 손실 ≤ 15%
- **AC1-3 (판정기의 가치)**: 고친 응답 비율 ≤ 30%, 그리고 감소폭 (orig − tgt) ≥ 0.5 × (orig − blind)
- **채택 = AC1-0~3** → 게이트웨이 "응답 끝 교정" 기능을 SPEC부터 설계(선택 기능). AC1-1·2만 충족이면 "교정 효과는 있으나 판정기 안내의 이점은 확인 안 됨"
- 응답 단위 쌍체 부트스트랩 2,000회(시드 2040). **API 상한 $25**(제출 전 예상 > $22면 멈춤)

## 실행 방법

1. 왼쪽 열쇠 아이콘(보안 비밀)에 `ANTHROPIC_API_KEY`, 노트북 접근 켜기
2. 런타임 유형 **GPU**(A100·L4·T4 아무거나 — 판정기 채점용, 속도는 재지 않음)
3. **모두 실행**. 배치는 보통 수십 분~1시간. 끊기면 다시 모두 실행 — 드라이브 `ac1/`에서 이어서 한다
4. 끝나면 `ac1_results.zip`(원문 없음)이 내려받아진다
""")

md("## 1. 환경 준비")
code("SRC = " + repr(SRC) + r"""
import os, sys, json, time, random, subprocess, shutil, pickle
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'anthropic', 'sentence-transformers', 'tiktoken', 'pysbd', '-U', 'transformers>=4.48'], check=True)
subprocess.run('git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || true', shell=True)
import numpy as np, pandas as pd, torch
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/hg/hallugate', exist_ok=True)
open('/content/hg/hallugate/__init__.py', 'w').write('')
for n, s in SRC.items(): open(f'/content/hg/hallugate/{n}.py', 'w').write(s)
sys.path.insert(0, '/content/hg')
from hallugate.judge import StudentJudge, read_config, input_mode

OUT = '/content/drive/MyDrive/Colab Notebooks/HalluGate'
F_DIR, I_DIR = os.path.join(OUT, 'final', 'judge_base'), os.path.join(OUT, 'i2', 'judge_i2')
R0_DIR, AC1_DIR = os.path.join(OUT, 'r0'), os.path.join(OUT, 'ac1')
os.makedirs(AC1_DIR, exist_ok=True)
for d in [F_DIR, I_DIR]:
    assert os.path.exists(os.path.join(d, 'judge_config.json')), f'{d}/judge_config.json 없음'
assert os.path.exists(os.path.join(R0_DIR, 'r0_items.pkl')), 'r0/r0_items.pkl 없음 (dev-L 분할)'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'
COMBO = """ + repr(COMBO) + r"""      # 판정기 v1.0 (configs/judge_combo_v1.json)
THR = COMBO['threshold']
def save_json(o, p): json.dump(o, open(p + '.tmp', 'w'), ensure_ascii=False); os.replace(p + '.tmp', p)
def load_json(p): return json.load(open(p)) if os.path.exists(p) else None
print('device', GPU, '| 임계값 B', THR)
""")

md("## 2. dev-L 원본 응답 450개 (과제별 150)")
code(L3.L3_INPUTS + AC1_DATA + r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
R0P = pickle.load(open(os.path.join(R0_DIR, 'r0_items.pkl'), 'rb'))
ITEMS = sample_items(R0P['items']['devL'], responses, sources)
print(f'응답 {len(ITEMS)}개 |', pd.Series([x['task'] for x in ITEMS]).value_counts().to_dict(),
      '| 생성 모델', pd.Series([x['gen_model'] for x in ITEMS]).value_counts().to_dict())
""")

md("## 3. 판정기 v1.0으로 경고 문장 찾기 (GPU)")
code(L3.L3_CORE + M1.M1_LABEL + M1.M1_SCORE + AC1_FLAG + r"""
jF, jI = make_judge(F_DIR, DEVICE), make_judge(I_DIR, DEVICE)
assert jF.mode == 'retrieval' and jI.mode == 'full', (jF.mode, jI.mode)
FN = judge_par if DEVICE == 'cuda' else judge_seq

def score_texts(pairs, path):
    # pairs: key -> (근거, 글). 결과: key -> 판정 지점 목록 (드라이브에 이어서 저장)
    S = load_json(path) or {}
    t0, refs = time.time(), {}
    for k, (ref, txt) in [(k, v) for k, v in pairs.items() if k not in S]:
        if ref not in refs: refs = {ref: (jF.prepare(ref), jI.prepare(ref))}
        S[k] = [list(p) for p in score_response(jF, jI, refs[ref], txt, FN)]
        if len(S) % 25 == 0:
            save_json(S, path); print(f'  판정 {len(S)}/{len(pairs)} ({time.time() - t0:.0f}s)', flush=True)
    save_json(S, path)
    return S

SC = score_texts({it['rid']: (it['ref'], it['text']) for it in sorted(ITEMS, key=lambda x: x['sid'])}, os.path.join(AC1_DIR, 'scores_orig.json'))
FLAGS = {it['rid']: flagged_starts(SC[it['rid']], THR) for it in ITEMS if it['rid'] in SC}
nf = sum(bool(v) for v in FLAGS.values())
print(f'경고 응답 {nf}/{len(FLAGS)} ({nf / len(FLAGS):.3f}) | 과제별', {t: round(float(np.mean([bool(FLAGS[x["rid"]]) for x in ITEMS if x["task"] == t])), 3) for t in TASKS})
""")

md("## 4. 고쳐 쓰기 (Haiku) — 형식 확인 → 비용 추정 → 배치")
code(T1.T1_API + AC1_REWRITE + AC1_EVAL + GRADE + r"""
BUDGET_USD, STOP_USD = 25.0, 22.0
def est_cost(reqs, out_tok):
    tot = 0.0
    for p in reqs.values():
        n_in = (len(p.get('system', '')) + sum(len(m['content']) for m in p['messages'])) / 3.5
        o = out_tok(p) if callable(out_tok) else out_tok
        pin, pout = PRICE[p['model']]; tot += n_in / 1e6 * pin + o / 1e6 * pout
    return tot

REQ_RW = rewrite_requests(ITEMS, FLAGS)
by_rid = {it['rid']: it for it in ITEMS}
EST_RW = est_cost(REQ_RW, lambda p: len(p['messages'][0]['content']) / 3.5 * 0.4)
# 평가 예상(고쳐 쓰기 전): 충실성 = 원본 + 경고 응답 + blind 전부, 정답성 = QA × 3, 손실 = 경고 응답 + blind 전부
n_items, n_qa = len(FLAGS), sum(1 for x in ITEMS if x['task'] == 'QA')
avg_ref = np.mean([len(x['ref']) + len(x['text']) for x in ITEMS]) / 3.5
EST_EV = ((n_items * 2 + nf) * ((avg_ref + 2500) * PRICE[LABELER][0] + 400 * PRICE[LABELER][1])
          + n_qa * 3 * (avg_ref * PRICE[LABELER][0] + 80 * PRICE[LABELER][1])
          + (n_items + nf) * ((avg_ref * 1.5) * PRICE[LABELER][0] + 80 * PRICE[LABELER][1])) / 1e6
print(f'고쳐 쓰기 {len(REQ_RW)}건 예상 ${EST_RW:.2f} + 평가 약 ${EST_EV:.2f} (멈춤 기준 ${STOP_USD}, 상한 ${BUDGET_USD})')
assert EST_RW + EST_EV <= STOP_USD, '예상 비용이 멈춤 기준을 넘음 — 멈춤'

# 형식 확인: tgt·blind 1건씩 바로 (수 센트)
_k = next(k for k in REQ_RW if k.startswith('t-'))
_g = rec(A.messages.create(**REQ_RW[_k])); _it = by_rid[_k[2:]]
print('[tgt 표시된 입력]', mark(_it['text'], FLAGS[_it['rid']])[:400].replace('\n', ' '))
print('[tgt 고친 글]', (clean_rewrite(_g) or '(무효)')[:400].replace('\n', ' '), '| stop', _g['stop'])
_g = rec(A.messages.create(**REQ_RW['b-' + _it['rid']]))
print('[blind 고친 글]', (clean_rewrite(_g) or '(무효)')[:400].replace('\n', ' '), '| stop', _g['stop'])

RW = run_batch(REQ_RW, os.path.join(AC1_DIR, 'rewrite.json'))
TX = arm_texts(ITEMS, FLAGS, RW)
print(f"세 팔 확보 {len(TX)}/{len(FLAGS)} | 고친 글 무효 tgt {sum(1 for k in REQ_RW if k.startswith('t-') and clean_rewrite(RW.get(k)) is None)}, "
      f"blind {sum(1 for k in REQ_RW if k.startswith('b-') and clean_rewrite(RW.get(k)) is None)} | "
      f"원본과 같은 글 tgt {sum(1 for r, t in TX.items() if FLAGS[r] and t['tgt'] == t['orig'])}, blind {sum(1 for t in TX.values() if t['blind'] == t['orig'])}")
""")

md("## 5. 평가 (Opus, 어느 팔인지 가림) — 충실성·정답성(QA)·정보 손실")
code(r"""
REQ_F, REQ_Q, REQ_L = eval_requests(ITEMS, TX)
EST2 = est_cost(REQ_F, 400) + est_cost(REQ_Q, 80) + est_cost(REQ_L, 80)
SPENT = cost(RW)
print(f'평가 요청: 충실성 {len(REQ_F)}, 정답성 {len(REQ_Q)}, 손실 {len(REQ_L)} | 예상 ${EST2:.2f} + 사용 ${SPENT:.2f}')
assert SPENT + EST2 <= BUDGET_USD - 1, '예상 비용이 상한을 넘음 — 멈춤'
FR = run_batch(REQ_F, os.path.join(AC1_DIR, 'faith.json'))
QR = run_batch(REQ_Q, os.path.join(AC1_DIR, 'grade.json'))
LR = run_batch(REQ_L, os.path.join(AC1_DIR, 'loss.json'))
# 충실성 파싱 실패는 바로 1회 재요청 (최대 50건)
text_of_key = {hkey('f', r, t[a]): t[a] for r, t in TX.items() for a in ARMS}
bad = [k for k in REQ_F if k in FR and not parse(FR[k]['text'], len(sentences(text_of_key[k])))[1]]
for k in bad[:50]:
    try: FR[k] = rec(A.messages.create(**REQ_F[k]))
    except Exception as e: print(f'  {k}: {type(e).__name__}'); break
save_json(FR, os.path.join(AC1_DIR, 'faith.json'))
print(f'충실성 파싱 실패 {len(bad)} → 재요청 후', sum(1 for k in REQ_F if k in FR and not parse(FR[k]["text"], len(sentences(text_of_key[k])))[1]))
""")

md("## 6. (보고) 고친 글을 판정기로 다시 채점 — 경고가 사라졌나")
code(r"""
RS = score_texts({r: (by_rid[r]['ref'], t['tgt']) for r, t in TX.items() if FLAGS[r] and t['tgt'] != t['orig']},
                 os.path.join(AC1_DIR, 'scores_tgt.json'))
RESCORE = {r: bool(flagged_starts(p, THR)) for r, p in RS.items()}
print(f'고친 tgt {len(RESCORE)}개 중 다시 경고 {sum(RESCORE.values())}')
""")

md("## 7. 사전 기준 판정")
code(AC1_METRICS + r"""
D = build_rows(ITEMS, FLAGS, TX, FR, QR, LR, RESCORE)
R = judge_ac1(D)
f = lambda ci: f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'
print(f"유효 {R['n']}/{R['n_items']} | 경고 응답 {R['n_flagged_valid']} (전체 {R['n_flagged_all']}) | 고친 응답 비율 {R['rewrite_frac']:.3f}")
print(f"환각 응답 비율: orig {R['hal']['orig']:.3f} | tgt {R['hal']['tgt']:.3f} | blind {R['hal']['blind']:.3f}")
print(f"  tgt − orig {R['diff_tgt']:+.3f} {f(R['diff_tgt_ci'])} (상대 {R['rel_tgt']:+.1%}) | blind − orig {R['diff_blind']:+.3f} {f(R['diff_blind_ci'])} (상대 {R['rel_blind']:+.1%})")
print(f"QA yes ({R['qa_n']}개): orig {R['yes']['orig']:.3f} | tgt {R['yes']['tgt']:.3f} | blind {R['yes']['blind']:.3f} | tgt − orig {R['yes_diff_tgt']:+.3f} {f(R['yes_diff_tgt_ci'])}")
print(f"정보 손실: tgt {R['loss_tgt']:.3f} ({R['n_changed_tgt']}개 고침) | blind {R['loss_blind']:.3f} ({R['n_changed_blind']}개)")
print(f"\n=== AC1-0 유효성(유효 ≥ 405, 경고 응답 ≥ 60) → {'충족' if R['AC1_0'] else '미달(판정 보류)'} ===")
print(f"=== AC1-1 효과: 상대 {R['rel_tgt']:+.1%} (≤ −15%), 상한 {R['diff_tgt_ci'][1]:+.3f} (< 0) → {'충족' if R['AC1_1'] else '미달'} ===")
print(f"=== AC1-2 부작용: yes {R['yes_diff_tgt']:+.3f} (≥ −0.05), 하한 {R['yes_diff_tgt_ci'][0]:+.3f} (≥ −0.10), 손실 {R['loss_tgt']:.3f} (≤ 0.15) → {'충족' if R['AC1_2'] else '미달'} ===")
print(f"=== AC1-3 판정기의 가치: 고친 비율 {R['rewrite_frac']:.3f} (≤ 0.30), 감소폭 tgt {R['gain_tgt']:.3f} vs blind의 절반 {0.5 * R['gain_blind']:.3f} → {'충족' if R['AC1_3'] else '미달'} ===")
print(f"=== AC1 {'채택' if R['AC1'] else '미채택'} ===")
print('(보고) 과제별', json.dumps(R['by_task'], ensure_ascii=False))
print(f"(보고) 원본 환각 비율: 경고 응답 {R['hal_orig_flagged']:.3f} / 경고 없는 응답 {R['hal_orig_unflagged']:.3f} | 길이 비율 {R['len_ratio']} | 고친 tgt 다시 경고 {R.get('reflag_tgt')}")
""")

md("## 8. 저장")
code(r"""
usage = {f: round(cost(load_json(os.path.join(AC1_DIR, f))), 2) for f in ['rewrite.json', 'faith.json', 'grade.json', 'loss.json']
         if os.path.exists(os.path.join(AC1_DIR, f))}
print('API 사용량(배치 단가) $', usage, '합계', round(sum(usage.values()), 2))
LOCAL = '/content/ac1_local'; os.makedirs(LOCAL, exist_ok=True)
D.to_csv(os.path.join(LOCAL, 'ac1_rows.csv'), index=False)            # 원문 없음
json.dump(dict(R, cost_usd=usage, gpu=GPU, threshold=THR, rewriter=REWRITER, labeler=LABELER, seed=SEED_AC),
          open(os.path.join(LOCAL, 'ac1_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for f in os.listdir(LOCAL): shutil.copy(os.path.join(LOCAL, f), os.path.join(AC1_DIR, f))
zp = shutil.make_archive('/content/ac1_results', 'zip', LOCAL)
from google.colab import files
files.download(zp)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
if __name__ == '__main__':
    out = os.path.join(NB_DIR, 'ac1_auto_correct.ipynb')
    json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
    print('wrote', out, len(cells), 'cells')
