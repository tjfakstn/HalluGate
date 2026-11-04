"""I1 노트북의 새 셀(입력 구성·학습 입력·채택 판정)을 가짜 데이터로 끝까지 돌려 보는 smoke test.
노트북과 같은 코드 문자열을 build_i1.py에서 가져와 exec한다. 데이터·드라이브·GPU 없이 CPU에서 돈다.
토크나이저는 DeBERTa-v3-base 것을 내려받는다(약 3MB). 실행: python experiments/builders/smoke_i1.py"""
import importlib.util, os, random, copy, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('build_i1', os.path.join(HERE, 'build_i1.py'))
B = importlib.util.module_from_spec(spec)
_argv = sys.argv; spec.loader.exec_module(B); sys.argv = _argv          # 노트북도 다시 생성됨

from transformers import AutoTokenizer, AutoModelForSequenceClassification

g = {'torch': torch, 'np': np, 'random': random, 'copy': copy, 'AutoTokenizer': AutoTokenizer,
     'MAX_LEN': 512, 'K': 8, 'CHUNK_WORDS': 40, 'DEVICE': 'cpu', 'AMP_DTYPE': torch.bfloat16, 'Z0': {}}
rng = random.Random(0)
WORDS = 'river city bloom india museum bridge north train school price year market festival temple'.split()

# 가짜 근거 문서 2개 (짧은 것 / 512를 넘는 긴 것)
REFS = {'r_short': ' '.join(rng.choice(WORDS) for _ in range(90)), 'r_long': ' '.join(rng.choice(WORDS) for _ in range(1400))}
class FakeRetriever:
    def encode(self, texts, **kw):
        v = torch.randn(len(texts), 16); return torch.nn.functional.normalize(v, dim=-1)
retriever = FakeRetriever()
chunk_cache = {}
def get_chunks(key):
    if key not in chunk_cache:
        w = REFS[key].split()
        ch = [' '.join(w[i:i + 40]) for i in range(0, max(len(w), 1), 40)]
        chunk_cache[key] = (ch, retriever.encode(ch))
    return chunk_cache[key]
g.update(retriever=retriever, get_chunks=get_chunks, ctx_of=lambda key, idx: ' ... '.join(get_chunks(key)[0][j] for j in idx))

# points()는 R4c LABELS 셀의 정의를 그대로 쓴다
exec(B.LABELS.split('\n\n\nWIN_TOK')[0], g)

# 가짜 응답: 문장 3개, 토큰 = 단어
def fake_item(i, key, onset):
    sents_txt = [' '.join(rng.choice(WORDS) for _ in range(n)) + '.' for n in (7, 12, 9)]
    text = ' '.join(sents_txt)
    ts, te, pos = [], [], 0
    for w in text.split(' '):
        s = text.index(w, pos); ts.append(s); te.append(s + len(w)); pos = s + len(w)
    sents, t = [], 0
    for st in sents_txt:
        n = len(st.split(' ')); a = ts[t]; b = te[t + n - 1]
        sents.append({'a': a, 'b': b, 't0': t, 't1': t + n - 1, 'y': 0}); t += n
    sent_of = {tt: sn['t0'] for sn in sents for tt in range(sn['t0'], sn['t1'] + 1)}
    return {'id': f'x{i}', 'task_type': ['QA', 'Summary'][i % 2], 'ref_key': key, 'text': text, 'ts': ts, 'te': te,
            'n_tok': len(ts), 'onset': onset, 'sents': sents, 'sent_of': sent_of, 'ends': {sn['t1'] for sn in sents}}

items = [fake_item(i, ['r_short', 'r_long'][i % 2], (None if i % 3 else 15)) for i in range(60)]
# R0식 질의 번호: (근거, 텍스트)로 묶음
qidx, Q_KEYS, Q_TEXTS = {}, [], []
for x in items:
    pts = []
    for t, txt in g['points'](x, 2):
        if not txt: continue
        k = (x['ref_key'], txt)
        if k not in qidx: qidx[k] = len(Q_KEYS); Q_KEYS.append(k[0]); Q_TEXTS.append(txt)
        pts.append((t, qidx[k]))
    x['pts2'] = pts
g.update(Q_KEYS=Q_KEYS, Q_TEXTS=Q_TEXTS)

exec('BACKBONES_PATH = "cross-encoder/nli-deberta-v3-base"\n' + B.I1_INPUT, g)
tok = g['I1_TOK']

# 1) 학습 지점 앞 문장: LABELS와 같은 순서, 첫 문장은 빈 값
P_TXT = [txt for x in items for t, txt in g['points'](x, 2) if txt]
P_KEY = [x['ref_key'] for x in items for t, txt in g['points'](x, 2) if txt]
P_PREV = g['prev_of_train'](items)
assert len(P_PREV) == len(P_TXT)
assert P_PREV[0] == '' and any(P_PREV)

# 2) 예산 채우기: 잘림 없음, 긴 문서는 조각이 8개보다 많을 수 있고 짧은 문서는 전부
P_B, P_SEL = g['build_inputs'](P_KEY, P_PREV, P_TXT)
lens = [len(tok(g['ctx_of'](k, s), b)['input_ids']) for k, s, b in zip(P_KEY, P_SEL, P_B)]
assert max(lens) <= 512, max(lens)
n_short = len(get_chunks('r_short')[0])
assert all(len(s) == n_short for k, s in zip(P_KEY, P_SEL) if k == 'r_short')
assert all(s == sorted(s) for s in P_SEL)
long_n = [len(s) for k, s in zip(P_KEY, P_SEL) if k == 'r_long']
print(f'입력 길이 최대 {max(lens)}, 긴 문서 조각 수 {min(long_n)}~{max(long_n)}, 짧은 문서 {n_short}개 전부')
assert tok.sep_token in P_B[[i for i, p in enumerate(P_PREV) if p][0]]

# 3) 평가 질의 새 번호: 같은 (근거, 텍스트)라도 앞 문장이 다르면 다른 번호, 지점은 그대로
Q2_KEYS, Q2_PREV, Q2_TEXTS, Q2_MAP = g['build_eval_queries'](items)
assert len(Q2_KEYS) >= len(Q_KEYS)
re_items = g['remap'](items, Q2_MAP)
for x, y in zip(items, re_items):
    assert [t for t, _ in x['pts2']] == [t for t, _ in y['pts2']]
    for (t, q), (_, q2) in zip(x['pts2'], y['pts2']):
        assert Q2_TEXTS[q2] == Q_TEXTS[q]
# 첫 문장 지점은 앞 문장 없음, 두 번째 문장 지점의 앞 문장 = 첫 문장 전체
x, y = items[0], re_items[0]
t_second = x['sents'][1]['t0'] + 1
q2 = dict(y['pts2'])[t_second]
assert Q2_PREV[q2] == x['text'][x['sents'][0]['a']:x['sents'][0]['b']].strip(), Q2_PREV[q2]
assert Q2_PREV[dict(y['pts2'])[x['sents'][0]['t1']]] == ''

# 4) 학습 입력·채점 함수: 작은 임의 모델로 enc_ids·score_q2·calsub_metric 실행
exec(B.SERIES, g); exec(B.EVAL_DEFS, g)
Q2_B, Q2_SEL = g['build_inputs'](Q2_KEYS, Q2_PREV, Q2_TEXTS)
g.update(P_KEY=P_KEY, P_SEL=P_SEL, P_B=P_B, Q2_KEYS=Q2_KEYS, Q2_B=Q2_B, Q2_SEL=Q2_SEL, Q2_MAP=Q2_MAP,
         CALSUB=items[:30], MAX_LEN=512)
exec(B.I1_TRAIN_OVERRIDE, g)
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
cfg = DebertaV2Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                      max_position_embeddings=512, num_labels=1, pad_token_id=tok.pad_token_id)
model = DebertaV2ForSequenceClassification(cfg).eval()
enc = g['enc_ids'](tok, [0, 1, 2])
assert enc['input_ids'].shape[1] <= 512
v = g['score_q2'](tok, model, list(range(len(Q2_KEYS))))
assert v.shape == (len(Q2_KEYS),) and np.isfinite(v).all()
rec, d4 = g['calsub_metric'](tok, model)
print(f'calsub_metric 동작: 재현율 {rec:.3f}, 지연 {d4}')

# 5) 채택 판정: 기준선 = R0 번호, I1 = 새 번호. 같은 점수를 주면 차이 0이어야 한다
exec(B.DECIDE, g)
SC = g['SC']
SC['F_base'] = np.array([hash(t) % 1000 / 100 for t in Q_TEXTS], dtype=np.float32)
SC['I1'] = np.array([hash(t) % 1000 / 100 for t in Q2_TEXTS], dtype=np.float32)
cal, cal2 = items[:40], re_items[:40]
sets_b = {'orig': (cal, items[40:]), 'C1': (cal, items[40:])}
sets_i = {'orig': (cal2, re_items[40:]), 'C1': (cal2, re_items[40:])}
r = g['compare'](sets_b, sets_i, '8', n_boot=200)
assert abs(r['orig']['delta']) < 1e-9 and r['orig']['ci_lo'] <= 0 <= r['orig']['ci_hi'], r['orig']
print('compare 동일 점수 → Δ 0 확인:', {k: round(v, 3) for k, v in r['orig'].items()})

# 6) 노트북 셀 문법 확인 (IPython 매직 줄 제외)
import ast, json
nb = json.load(open(os.path.join(B.NB_DIR, 'i1_input_fill_prev.ipynb')))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        ast.parse('\n'.join(l for l in c['source'].split('\n') if not l.lstrip().startswith('!')))
print('smoke test 통과')
