"""T1 smoke test (맥 CPU, API·실제 데이터 없음): 가짜 RAGTruth QA + 가짜 임베더 + 가짜 API 결과로 노트북과 같은 T1_* 코드를 끝까지 실행.
- 다이어트 답을 판정자가 항상 SAME으로 보고 환각 차이 0 → 채택이어야 함
- 다이어트 답이 항상 WORSE → T1-2 미달
- test 분할 근거는 뽑히지 않아야 함
python experiments/builders/smoke_t1.py"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np
import build_t1 as B

n = 0
for c in B.cells:
    if c['cell_type'] == 'code':
        ast.parse(c['source']); n += 1
print(f'[1] t1: 코드 셀 {n}개 문법 통과')

rng = random.Random(0)
words = 'alpha beta gamma delta river mountain city bridge harbor engine market forest valley ocean'.split()
sources, responses = [], []
for k in range(260):
    split = 'test' if k % 10 == 0 else 'train'
    passages = ' '.join(rng.choice(words) for _ in range(150)) + f' topic{k} answer{k} is {k}.'
    sources.append({'source_id': str(k), 'task_type': 'QA' if k % 7 else 'Summary',
                    'source_info': {'question': f'What is topic{k} answer{k}?', 'passages': passages}})
    responses.append({'source_id': str(k), 'split': split})
G = {'np': np, 'json': json, 'os': os, 'random': random}
exec(B.T1_DATA, G)
POOL = G['train_qa_sources'](responses, sources)
assert all(int(s['source_id']) % 10 and int(s['source_id']) % 7 for s in POOL)          # test·Summary 제외
ITEMS = G['make_items'](POOL)
assert len(ITEMS) == 200 and ITEMS[0]['own'] in ITEMS[0]['ref_L'] and len(ITEMS[0]['ref_L']) > 7 * len(ITEMS[0]['own'])
assert [x['sid'] for x in G['make_items'](POOL)] == [x['sid'] for x in ITEMS]           # 재현
print(f'[2] train QA {len(POOL)}개 중 {len(ITEMS)}개, test·Summary 제외 확인')


class HashEmb:
    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace('?', ' ').replace('.', ' ').split():
                out[i, sum(map(ord, w)) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


exec(B.T1_TRIM, G)
T = G['trim_all'](ITEMS, G['Trimmer'](HashEmb(), 120, 20), lambda t: len(t.split()))
L = T[T.cond == 'L']
assert (L.ref_tok_k4 < L.ref_tok_full).all() and (T[T.cond == 'S'].chunks_total <= 4).all()
print(f"[3] L 근거 토큰 감소 중앙값 {float((1 - L.ref_tok_k4 / L.ref_tok_full).median()):.2f}, 원래 지문 재현(k4) {L.own_recall_k4.mean():.2f}")
REQ = G['gen_requests'](ITEMS)
assert len(REQ) == 800 and REQ['g-0-L-trim']['model'] == 'claude-haiku-5-5' and 'reference documents' in REQ['g-0-L-trim']['system']

import pysbd
G['SEG'] = pysbd.Segmenter(language='en', clean=False, char_span=True)
G['PRICE'] = {'claude-haiku-5-5': (0.05, 0.25), 'claude-opus-5-5': (2.0, 10.0)}
exec(B.M1.M1_LABEL + B.M1.M1_SCORE + B.M1.M1_METRICS, G)


def gen(k, p):
    full = k.endswith('full')
    return {'text': 'The answer is here. It is supported.', 'stop': 'end_turn', 'model': 'x',
            'in': 4000 if full and '-L-' in k else 600, 'out': 50}


GR = {k: gen(k, p) for k, p in REQ.items()}
P, F = G['judge_requests'](ITEMS, GR)
assert len(P) == 400 and len(F) == 800 and P['p-0-L']['model'] == 'claude-opus-5-5'
assert ITEMS[0]['own'] in P['p-0-L']['messages'][0]['content'] and ITEMS[0]['ref_L'] not in P['p-0-L']['messages'][0]['content']
FR = {k: {'text': json.dumps({'sentences': [{'i': 1, 'label': 'S'}, {'i': 2, 'label': 'S'}]}), 'model': 'x', 'in': 1, 'out': 1} for k in F}
exec(B.T1_METRICS, G)
PR_same = {k: {'text': '{"better": "same"}', 'model': 'x', 'in': 1, 'out': 1} for k in P}
R = G['judge_t1'](G['build_rows'](ITEMS, GR, PR_same, FR, T))
assert R['T1'] and R['L']['input_reduction_median'] > 0.8 and R['L']['worse'] == 0, R
# 다이어트 답이 항상 나쁨: 다이어트가 1번이면 "2", 아니면 "1"
PR_worse = {}
for k in P:
    _, i, cond = k.split('-')
    PR_worse[k] = {'text': json.dumps({'better': '2' if G['trim_first'](int(i), cond) else '1'}), 'model': 'x', 'in': 1, 'out': 1}
R2 = G['judge_t1'](G['build_rows'](ITEMS, GR, PR_worse, FR, T))
assert R2['L']['worse'] == 1.0 and not R2['T1_2'] and not R2['T1']
print(f"[4] SAME → T1 {R['T1']} (감소 {R['L']['input_reduction_median']:.2f}), 항상 WORSE → T1-2 {R2['T1_2']}")
print('smoke_t1 통과')
