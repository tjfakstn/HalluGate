"""T1b smoke test (맥 CPU, API·실제 데이터 없음): 가짜 RAGTruth QA + 가짜 임베더 + 가짜 API 결과로 노트북과 같은 코드를 끝까지 실행.
- T1 질문이 섞이지 않는지, 세 팔 요청(temperature 0, max_tokens 1024)
- 세 팔 모두 yes → 채택 / trim만 자주 no → T1b-2 미달 / 대조군이 흔들림 → T1b-0 미달(판정 보류)
python experiments/builders/smoke_t1b.py"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np
import pysbd
import build_t1b as B

n = sum(1 for c in B.cells if c['cell_type'] == 'code' and ast.parse(c['source']))
print(f'[1] t1b: 코드 셀 {n}개 문법 통과')

rng = random.Random(0)
words = 'alpha beta gamma delta river mountain city bridge harbor engine market forest valley ocean'.split()
sources, responses = [], []
for k in range(700):
    sources.append({'source_id': str(k), 'task_type': 'QA' if k % 7 else 'Summary',
                    'source_info': {'question': f'What is topic{k} answer{k}?',
                                    'passages': ' '.join(rng.choice(words) for _ in range(150)) + f' topic{k} answer{k} is {k}.'}})
    responses.append({'source_id': str(k), 'split': 'test' if k % 10 == 0 else 'train'})
G = {'np': np, 'json': json, 'os': os, 'random': random}
exec(B.T1.T1_DATA + B.T1B_DATA, G)
POOL = G['train_qa_sources'](responses, sources)
T1_SIDS = {x['sid'] for x in G['make_items'](POOL)}
ITEMS = G['make_items_b'](POOL, T1_SIDS)
assert len(ITEMS) == 200 and not {x['sid'] for x in ITEMS} & T1_SIDS
assert [x['sid'] for x in G['make_items_b'](POOL, T1_SIDS)] == [x['sid'] for x in ITEMS]
print(f'[2] 풀 {len(POOL)}, T1 제외 후 {len(ITEMS)}개, 겹침 없음')


class HashEmb:
    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace('?', ' ').replace('.', ' ').split():
                out[i, sum(map(ord, w)) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


exec(B.T1.T1_TRIM, G)
T = G['trim_all'](ITEMS, G['Trimmer'](HashEmb(), 120, 20), lambda t: len(t.split()))
G['SEG'] = pysbd.Segmenter(language='en', clean=False, char_span=True)
G['PRICE'] = {'claude-haiku-5-5': (0.05, 0.25), 'claude-opus-5-5': (2.0, 10.0)}
exec(B.M1.M1_LABEL + B.M1.M1_SCORE + B.M1.M1_METRICS + B.T1B_GEN, G)
REQ = G['gen_requests_b'](ITEMS, True)
r0 = REQ['g-0-full']
assert len(REQ) == 600 and r0['temperature'] == 0.0 and r0['max_tokens'] == 1024 and r0 == REQ['g-0-full2']
assert 'temperature' not in G['gen_requests_b'](ITEMS, False)['g-0-trim']
GR = {k: {'text': 'The answer is here. It is supported.', 'stop': 'end_turn', 'model': 'x',
          'in': 4000 if not k.endswith('trim') else 1000, 'out': 50} for k in REQ}
Q, F = G['judge_requests_b'](ITEMS, GR)
assert len(Q) == 600 and len(F) == 600 and ITEMS[0]['own'] in Q['q-0-trim']['messages'][0]['content']
exec(B.T1.T1_METRICS + B.T1B_METRICS, G)
FR = {k: {'text': json.dumps({'sentences': [{'i': 1, 'label': 'S'}, {'i': 2, 'label': 'S'}]}), 'model': 'x', 'in': 1, 'out': 1} for k in F}


def run(grade_fn):
    QR = {k: {'text': json.dumps({'grade': grade_fn(k)}), 'model': 'x', 'in': 1, 'out': 1} for k in Q}
    return G['judge_t1b'](G['build_rows_b'](ITEMS, GR, QR, FR, T))


R = run(lambda k: 'yes')
assert R['T1b'] and R['n'] == 200 and R['input_reduction_median'] > 0.7, R
R2 = run(lambda k: 'no' if k.endswith('trim') and int(k.split('-')[1]) % 4 == 0 else 'yes')     # trim만 25% 나쁨
assert R2['T1b_0'] and not R2['T1b_2'] and abs(R2['yes_diff'] + 0.25) < 1e-9
R3 = run(lambda k: 'no' if k.endswith('full2') and int(k.split('-')[1]) % 5 == 0 else 'yes')   # 대조군 흔들림
assert not R3['T1b_0'] and not R3['T1b']
print(f"[3] 모두 yes → 채택 {R['T1b']} | trim 25% 나쁨 → T1b-2 {R2['T1b_2']} | 대조군 흔들림 → T1b-0 {R3['T1b_0']}")
print('smoke_t1b 통과')
