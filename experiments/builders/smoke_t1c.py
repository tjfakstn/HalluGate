"""T1c smoke test (맥 CPU, API·실제 데이터 없음): 가짜 데이터로 노트북과 같은 코드를 끝까지 실행.
- T1·T1b 질문이 섞이지 않는지, trim 팔이 top_k 8 조각인지
- 모두 yes·환각 같음 → 채택 / trim 환각만 늘어남 → T1c-3 미달
python experiments/builders/smoke_t1c.py"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np
import pysbd
import build_t1c as C
TB, T1, M1 = C.TB, C.T1, C.M1

n = sum(1 for c in C.cells if c['cell_type'] == 'code' and ast.parse(c['source']))
print(f'[1] t1c: 코드 셀 {n}개 문법 통과')
rng = random.Random(0)
words = 'alpha beta gamma delta river mountain city bridge harbor engine market forest valley ocean'.split()
sources, responses = [], []
for k in range(1000):
    sources.append({'source_id': str(k), 'task_type': 'QA' if k % 7 else 'Summary',
                    'source_info': {'question': f'What is topic{k} answer{k}?',
                                    'passages': ' '.join(rng.choice(words) for _ in range(150)) + f' topic{k} answer{k} is {k}.'}})
    responses.append({'source_id': str(k), 'split': 'test' if k % 10 == 0 else 'train'})
G = {'np': np, 'json': json, 'os': os, 'random': random}
exec(T1.T1_DATA + TB.T1B_DATA + C.T1C_DATA, G)
POOL = G['train_qa_sources'](responses, sources)
ITEMS, SEEN = G['make_items_c'](POOL)
t1 = {x['sid'] for x in G['make_items'](POOL)}
t1b = {x['sid'] for x in G['make_items_b'](POOL, t1)}
assert G['SEED_B'] == 2038 and SEEN == t1 | t1b and len(SEEN) == 400                       # 시드 원복, T1b 재현
assert len(ITEMS) == 200 and not {x['sid'] for x in ITEMS} & SEEN
print(f'[2] 풀 {len(POOL)}, T1·T1b {len(SEEN)}개 제외 후 {len(ITEMS)}개')


class HashEmb:
    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace('?', ' ').replace('.', ' ').split():
                out[i, sum(map(ord, w)) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


exec(T1.T1_TRIM, G); G['TOP_K'] = G['TOP_K_C']
T = G['trim_all'](ITEMS, G['Trimmer'](HashEmb(), 120, 20), lambda t: len(t.split()))
assert all(len(it['trim_L']) == 8 for it in ITEMS)                                            # top_k 8
G['SEG'] = pysbd.Segmenter(language='en', clean=False, char_span=True)
G['PRICE'] = {'claude-haiku-5-5': (0.05, 0.25), 'claude-opus-5-5': (2.0, 10.0)}
exec(M1.M1_LABEL + M1.M1_SCORE + M1.M1_METRICS + TB.T1B_GEN, G)
REQ = G['gen_requests_b'](ITEMS, False)
GR = {k: {'text': 'The answer is here. It is supported.', 'stop': 'end_turn', 'model': 'x',
          'in': 4000 if not k.endswith('trim') else 2000, 'out': 50} for k in REQ}
Q, F = G['judge_requests_b'](ITEMS, GR)
exec(T1.T1_METRICS + TB.T1B_METRICS + C.T1C_METRICS, G)
QR = {k: {'text': '{"grade": "yes"}', 'model': 'x', 'in': 1, 'out': 1} for k in Q}


def faith(bad):
    return {k: {'text': json.dumps({'sentences': [{'i': 1, 'label': 'H' if bad(k) else 'S'}, {'i': 2, 'label': 'S'}]}),
                'model': 'x', 'in': 1, 'out': 1} for k in F}


R = G['judge_t1c'](G['build_rows_b'](ITEMS, GR, QR, faith(lambda k: False), T))
assert R['T1c'] and abs(R['input_reduction_median'] - 0.5) < 1e-9 and 'own_recall_k8' in R, R
R2 = G['judge_t1c'](G['build_rows_b'](ITEMS, GR, QR, faith(lambda k: k.endswith('trim') and int(k.split('-')[1]) % 10 == 0), T))
assert R2['T1c_0'] and R2['T1c_2'] and not R2['T1c_3'] and not R2['T1c']
print(f"[3] 같음 → 채택 {R['T1c']} (절감 {R['input_reduction_median']:.2f}) | trim 환각 +10%p → T1c-3 {R2['T1c_3']}")
print('smoke_t1c 통과')
