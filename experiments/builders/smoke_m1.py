"""M1 노트북의 요청 구성·정답 파싱·판정·지표 함수를 가짜 근거·가짜 API 결과로 맥 CPU에서 돌려 보는 smoke test.
판정기 가중치 폴더(F_DIR, I_DIR)가 있으면 실제 판정기로 점수를 내고, 없으면 가짜 점수를 쓴다. API는 호출하지 않는다.
실행: python experiments/builders/smoke_m1.py [F_DIR] [I_DIR]"""
import ast, importlib.util, json, os, random, sys, tempfile, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('build_m1', os.path.join(HERE, 'build_m1.py'))
B = importlib.util.module_from_spec(spec); _argv = sys.argv; spec.loader.exec_module(B); sys.argv = _argv
from hallugate.judge import StudentJudge, read_config, input_mode
from hallugate.store import SQLiteStore
from hallugate.context import rag_messages, rag_reference

# 0) 노트북 코드 셀이 모두 파이썬으로 읽히는지
for i, c in enumerate(B.cells):
    if c['cell_type'] == 'code':
        ast.parse(c['source'] if isinstance(c['source'], str) else ''.join(c['source']))
print('코드 셀', sum(c['cell_type'] == 'code' for c in B.cells), '개 구문 확인')

g = {'np': np, 'torch': torch, 'json': json, 'time': time, 'random': random, 'os': os, 'COMBO': B.COMBO,
     'StudentJudge': StudentJudge, 'read_config': read_config, 'input_mode': input_mode,
     'SQLiteStore': SQLiteStore, 'rag_messages': rag_messages, 'rag_reference': rag_reference}
for part in [B.L3.L3_INPUTS, B.L3.L3_CORE, B.M1_FLOW, B.M1_LABEL, B.M1_SCORE, B.M1_METRICS]:
    exec(part, g)
assert 'Subtle baseless info' in g['SYSTEM'] and set(g['TASK_DESC']) == {'QA', 'Summary', 'Data2txt'}

F_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Downloads/f1_final_judges/judge_base')
I_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/Downloads/i2_judge/judge_i2')
REAL = os.path.exists(F_DIR) and os.path.exists(I_DIR)


class FakeEmb:
    """단어 해시 주머니 임베딩 (정규화)."""
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, **kw):
        v = np.zeros((len(texts), 256), np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                v[i, hash(w.strip('.,?')) % 256] += 1
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


if REAL:
    jF, jI = g['make_judge'](F_DIR, 'cpu'), g['make_judge'](I_DIR, 'cpu')
    assert jF.mode == 'retrieval' and jI.mode == 'full'
    emb = jF.embedder
    thr_f = read_config(F_DIR)['thresholds']['8']['svc10_orig']
else:
    emb, thr_f = FakeEmb(), 3.968751

# 1) 요청 구성: QA는 컬렉션 검색, 나머지는 원 프롬프트
eiffel = ('passage 1: The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed in 1889 and is 330 metres tall. '
          'It was designed by the company of Gustave Eiffel for the 1889 World Fair. ') * 3
python = ('passage 1: Python is a high-level programming language created by Guido van Rossum and first released in 1991. '
          'It emphasizes code readability and uses significant indentation. ') * 3
sources = {
    's1': {'task_type': 'QA', 'source_info': {'question': 'How tall is the Eiffel Tower?', 'passages': eiffel}, 'prompt': 'IGNORED'},
    's2': {'task_type': 'QA', 'source_info': {'question': 'Who created the Python programming language?', 'passages': python}, 'prompt': 'IGNORED'},
    's3': {'task_type': 'Summary', 'source_info': 'The city council approved a new park budget of 2 million dollars on Monday. Work starts in May.',
           'prompt': 'Summarize the following news within 50 words:\nThe city council approved a new park budget of 2 million dollars on Monday. Work starts in May.'},
    's4': {'task_type': 'Data2txt', 'source_info': {'name': 'Cafe Blue', 'stars': 4.5, 'review_info': [{'text': 'Great coffee.'}]},
           'prompt': 'Write an objective overview about the following local business: {"name": "Cafe Blue"}'},
}
db = os.path.join(tempfile.mkdtemp(), 'm1.sqlite3')
store = SQLiteStore(db, emb, 120, 20)
assert g['build_collection'](store, 'm1-devL', list(sources), sources) == 2
cases = g['make_cases']('devL', list(sources), sources, store, 'm1-devL')
assert len(cases) == 8 and {c['gen'] for c in cases.values()} == {'haiku', 'sonnet'}
qa = cases['devL-haiku-s1']
assert qa['flow'] == 'collection' and qa['gold_in'] is True and '<documents>' in qa['system']
assert qa['messages'] == [{'role': 'user', 'content': 'How tall is the Eiffel Tower?'}]
assert qa['ref_label'].startswith('Question: How tall') and qa['ref_judge'] in qa['ref_label']
assert cases['devL-sonnet-s2']['gold_in'] is True
sm = cases['devL-haiku-s3']
assert sm['flow'] == 'direct' and sm['system'] is None and sm['messages'][0]['content'] == sources['s3']['prompt']
assert sm['ref_judge'] == g['build_reference'](sources['s3'])
p = g['gen_params'](qa); assert p['system'] == qa['system'] and p['model'] == 'claude-haiku-5-5' and p['max_tokens'] == 8000
assert 'system' not in g['gen_params'](sm)
json.dumps(cases)                                                  # 드라이브 저장 가능

# 2) 정답 요청·파싱
texts = {
    'devL-haiku-s1': 'The Eiffel Tower is 330 metres tall. It was completed in 1889.',
    'devL-sonnet-s1': 'The Eiffel Tower is 330 metres tall. It is the most visited paid monument in the world and was painted gold in 2020.',
    'devL-haiku-s2': 'Python was created by Guido van Rossum. It was first released in 1991.',
    'devL-sonnet-s2': 'Python was created by Guido van Rossum in 1991.',
    'devL-haiku-s3': 'The council approved a 2 million dollar park budget. Work starts in May.',
    'devL-sonnet-s3': '',
    'devL-haiku-s4': 'Cafe Blue is a cafe rated 4.5 stars. Reviewers praise its coffee.',
}
gens = {c: {'text': t, 'stop': 'end_turn', 'model': cases[c]['model'], 'in': 100, 'out': 30} for c, t in texts.items()}
gens['devL-sonnet-s4'] = {'text': '', 'stop': 'refusal', 'model': 'claude-sonnet-5-5', 'in': 100, 'out': 0}
lp = g['label_params'](cases['devL-sonnet-s1'], texts['devL-sonnet-s1'])
assert lp['model'] == 'claude-opus-5-5' and lp['max_tokens'] == 4096 and lp['system'] == g['SYSTEM']
um = lp['messages'][0]['content']
assert 'RESPONSE (2 numbered sentences)' in um and '[2] It is the most visited' in um and 'Question: How tall' in um
assert g['gen_status'](gens['devL-sonnet-s3']) == 'empty' and g['gen_status'](gens['devL-sonnet-s4']) == 'refusal'
assert g['gen_status'](None) == 'nogen' and g['gen_status'](gens['devL-haiku-s1']) == 'valid'

def fake_opus(c):
    n = len(g['sentences'](texts[c]))
    labs = [{'i': i + 1, 'label': 'S'} for i in range(n)]
    if c == 'devL-sonnet-s1':
        labs[1] = {'i': 2, 'label': 'H', 'type': 'evident_baseless', 'phrase': 'painted gold', 'reason': 'not in reference'}
    if c == 'devL-haiku-s4':
        labs[1] = {'i': 2, 'label': 'A', 'type': 'subtle_baseless', 'phrase': 'praise', 'reason': 'x'}
    return 'Here you go: ' + json.dumps({'sentences': labs})

labels = {}
for c in texts:
    if g['gen_status'](gens[c]) != 'valid': continue
    labs, ok = g['parse'](fake_opus(c), len(g['sentences'](texts[c])))
    assert ok, c
    if c != 'devL-sonnet-s2': labels[c] = labs                   # 정답 없음 1건
assert g['parse']('not json', 2) == ({}, False)

# 3) 판정: 실제 가중치면 실제 점수, 아니면 가짜 점수
valid = [c for c in cases if g['gen_status'](gens.get(c)) == 'valid']
scores = {}
if REAL:
    refs = {}
    for c in valid:
        r = cases[c]['ref_judge']
        if r not in refs: refs = {r: (jF.prepare(r), jI.prepare(r))}
        scores[c] = [list(x) for x in g['score_response'](jF, jI, refs[r], texts[c], g['judge_seq'])]
    pts = g['points_of'](texts['devL-sonnet-s1'])
    assert len(scores['devL-sonnet-s1']) == len(pts) and all(x[0] == a for x, (a, _) in zip(scores['devL-sonnet-s1'], pts))
    sc = np.array(scores['devL-sonnet-s1'])
    assert np.allclose(sc[:, 4], [g['combine'](f, i) for f, i in sc[:, 2:4]])
    print('실제 점수 (응답별 조합 최대)', {c: round(max(x[4] for x in v), 2) for c, v in scores.items()})
else:
    for c in valid:
        pts = g['points_of'](texts[c])
        scores[c] = [[a, cut, 0.0, 0.0, 0.0] for a, cut in pts]
    print('가중치 없음 — 가짜 점수 사용')
# 지표 검증용으로 두 응답의 점수를 고정: haiku-s1(멀쩡)은 첫 지점에서 경보, sonnet-s1(환각)은 둘째 문장에서 경보
thr = B.COMBO['threshold']
s_h1 = [list(x) for x in scores['devL-haiku-s1']]; s_h1[0][4] = thr + 1; s_h1[0][2] = thr_f + 1
s_s1 = [list(x) for x in scores['devL-sonnet-s1']]
for x in s_s1: x[4] = min(x[4], thr - 1)
s_s1[-1][4] = thr + 0.5
scores2 = {**scores, 'devL-haiku-s1': s_h1, 'devL-sonnet-s1': s_s1}
for c in ['devL-haiku-s2', 'devL-haiku-s3', 'devL-haiku-s4']:
    scores2[c] = [x[:4] + [min(x[4], thr - 1)] for x in scores[c]]

# 4) 응답 표·지표
df = g['response_rows'](cases, gens, labels, scores2, thr, thr_f)
st = df.set_index('cid').status.to_dict()
assert st['devL-sonnet-s3'] == 'empty' and st['devL-sonnet-s4'] == 'refusal' and st['devL-sonnet-s2'] == 'nolabel'
row = df.set_index('cid').loc
assert row['devL-haiku-s1', 'flag_c'] and row['devL-haiku-s1', 'clean'] and row['devL-haiku-s1', 'flag_f']
assert row['devL-sonnet-s1', 'n_H'] == 1 and not row['devL-sonnet-s1', 'clean'] and row['devL-sonnet-s1', 'strict_det']
assert row['devL-haiku-s4', 'clean'] and not row['devL-haiku-s4', 'strict_clean'] and row['devL-haiku-s4', 'n_A'] == 1
rep = g['report'](df)
cc = rep['clean_combo']
assert (cc['k'], cc['n']) == (1, 4), cc                           # 멀쩡 4개(h-s1, h-s2, h-s3, h-s4) 중 1개 표시
assert (rep['hal_any']['k'], rep['hal_any']['n']) == (1, 1) and rep['hal_strict']['k'] == 1
assert rep['strict_clean_combo']['n'] == 3
fp = rep['flag_precision_H']; assert (fp['k'], fp['n']) == (1, 2) and abs(fp['rate'] - 0.5) < 1e-9
assert rep['qa_gold_in'] == 1.0 and rep['by_gen']['haiku']['n'] == 4
assert rep['status']['ok'] == 5 and rep['max_tokens_stops'] == 0
json.dumps(rep, default=float)
print('표시율', g['fmt'](cc), '| 과제별', {t: g['fmt'](d) for t, d in rep['by_task'].items()})

# 5) 등각 재보정·Wilson
assert abs(g['conformal'](np.arange(1, 10)) - (9 + 1e-6)) < 1e-9
assert g['conformal'](np.arange(1, 5)) == float('inf')
mx = np.random.RandomState(0).normal(size=500)
t = g['conformal'](mx); assert (mx >= t).mean() <= 0.10
p_, lo, hi = g['wilson'](15, 100); assert abs(p_ - 0.15) < 1e-12 and lo < 0.15 < hi and 0.08 < lo and hi < 0.24
print('smoke_m1 통과', '(실제 판정기)' if REAL else '(가짜 점수)')
