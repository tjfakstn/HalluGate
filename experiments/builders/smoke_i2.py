"""I2 노트북의 새 셀(입력 구성·학습 루프·채택 판정)을 가짜 데이터·작은 ModernBERT로 끝까지 돌려 보는 smoke test.
노트북과 같은 코드 문자열을 build_i2.py에서 가져와 exec한다. CPU, 토크나이저만 내려받음. 실행: python experiments/builders/smoke_i2.py"""
import importlib.util, os, random, copy, sys, json, ast, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('build_i2', os.path.join(HERE, 'build_i2.py'))
B = importlib.util.module_from_spec(spec); _argv = sys.argv; spec.loader.exec_module(B); sys.argv = _argv

from transformers import AutoTokenizer, ModernBertConfig, ModernBertForSequenceClassification
g = {'torch': torch, 'np': np, 'random': random, 'copy': copy, 'time': time, 'DEVICE': 'cpu', 'Z0': {}}
rng = random.Random(0)
WORDS = 'river city bloom india museum bridge north train school price year market festival temple'.split()
REFS = {'r_short': ' '.join(rng.choice(WORDS) for _ in range(90)), 'r_long': ' '.join(rng.choice(WORDS) for _ in range(2500))}
g['REFS'] = REFS
exec(B.LABELS.split('\n\n\nWIN_TOK')[0], g)          # points()

def fake_item(i, key, onset):
    sents_txt = [' '.join(rng.choice(WORDS) for _ in range(n)) + '.' for n in (7, 12, 9)]
    text = ' '.join(sents_txt)
    ts, te, pos = [], [], 0
    for w in text.split(' '):
        s = text.index(w, pos); ts.append(s); te.append(s + len(w)); pos = s + len(w)
    sents, t = [], 0
    for st in sents_txt:
        n = len(st.split(' ')); sents.append({'a': ts[t], 'b': te[t + n - 1], 't0': t, 't1': t + n - 1, 'y': 0}); t += n
    sent_of = {tt: sn['t0'] for sn in sents for tt in range(sn['t0'], sn['t1'] + 1)}
    return {'id': f'x{i}', 'task_type': ['QA', 'Summary'][i % 2], 'ref_key': key, 'text': text, 'ts': ts, 'te': te,
            'n_tok': len(ts), 'onset': onset, 'sents': sents, 'sent_of': sent_of, 'ends': {sn['t1'] for sn in sents}}

items = [fake_item(i, ['r_short', 'r_long'][i % 2], (None if i % 3 else 15)) for i in range(60)]
qidx, Q_TEXTS = {}, []
for x in items:
    x['pts2'] = []
    for t, txt in g['points'](x, 2):
        if not txt: continue
        k = (x['ref_key'], txt)
        if k not in qidx: qidx[k] = len(Q_TEXTS); Q_TEXTS.append(txt)
        x['pts2'].append((t, qidx[k]))
RESP_TEXT = {x['id']: x['text'] for x in items}
eval_items = [{k: v for k, v in x.items() if k != 'text'} for x in items]   # R0 항목에는 응답 본문이 없다

exec(B.I2_POINTS, g)
g['MAX_LEN2'] = 512                                     # 작은 모델용
tok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')

# 1) 학습 지점 앞부분: LABELS와 같은 순서, 판정 텍스트 끝과 일치
P_TXT = [txt for x in items for t, txt in g['points'](x, 2) if txt]
TP = g['train_prefixes'](items)
assert len(TP) == len(P_TXT) and all(p.rstrip().endswith(t[-20:].rstrip()) for (_, p), t in zip(TP, P_TXT))
assert TP[-1][1] == items[-1]['text'][:items[-1]['te'][-1]]

# 2) 평가 지점 새 번호, 채점 대상(문장 끝 + 4토큰), 응답 앞부분이 응답 처음부터 시작
E_KEYS, E_TEXTS, E_NEED, E_MAP = g['build_eval_points'](eval_items, RESP_TEXT)
assert len(E_KEYS) == sum(len(x['pts2']) for x in items)
re_items = g['remap'](eval_items, E_MAP)
for x, y in zip(items, re_items):
    for (t, _), (_, e) in zip(x['pts2'], y['pts2']):
        assert E_TEXTS[e] == x['text'][:x['te'][t]]
        assert E_NEED[e] == (t in x['ends'] or (t - x['sent_of'][t] + 1) % 4 == 0)
enc = g['encode'](tok, [(E_KEYS[e], E_TEXTS[e]) for e in range(3)])
assert enc['input_ids'].shape[1] <= 512
print(f'평가 지점 {len(E_KEYS)}개, 채점 대상 {int(E_NEED.sum())}개, 긴 근거 입력 {enc["input_ids"].shape}')

# 3) 학습 루프·채점·체크포인트 선택: 작은 임의 ModernBERT로 몇 단계
exec(B.SERIES, g); exec(B.EVAL_DEFS, g)
CALSUB2 = re_items[:30]
g.update(TP=TP, Y_ALL=np.array([i % 3 == 0 for i in range(len(TP))], dtype=np.float32), E_KEYS=E_KEYS, E_TEXTS=E_TEXTS,
         CALSUB2=CALSUB2, CALSUB_E=sorted({e for x in CALSUB2 for t, e in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % 8 == 0}),
         N_TRAIN=64)
exec(B.I2_TRAIN, g)
cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       max_position_embeddings=512, num_labels=1, pad_token_id=tok.pad_token_id, reference_compile=False)
model = ModernBertForSequenceClassification(cfg)
model, hist = g['train_i2'](tok, model, 42)
assert len(hist) == 1 + g['N_EVALS'], hist
v = g['score_pairs'](tok, model, [(E_KEYS[e], E_TEXTS[e]) for e in np.flatnonzero(E_NEED)[:40]])
assert np.isfinite(v).all()
print('학습 루프 동작, cal-sub 이력', [(s, round(r, 3)) for s, r in hist])

# 4) 채택 판정 함수: 같은 점수면 차이 0
exec(B.DECIDE, g)
SC = g['SC']
SC['F_base'] = np.array([hash(t) % 1000 / 100 for t in Q_TEXTS], dtype=np.float32)
# I2 점수 = 같은 지점의 F_base 점수 → 차이 0
SC['I2'] = np.full(len(E_KEYS), np.nan, dtype=np.float32)
for x, y in zip(eval_items, re_items):
    for (t, q), (_, e) in zip(x['pts2'], y['pts2']):
        if E_NEED[e]: SC['I2'][e] = SC['F_base'][q]
sb = {'orig': (eval_items[:40], eval_items[40:]), 'C1': (eval_items[:40], eval_items[40:])}
si = {'orig': (re_items[:40], re_items[40:]), 'C1': (re_items[:40], re_items[40:])}
for k in ['end', '4', '8', '16']:
    r = g['compare'](sb, si, k, n_boot=100)
    assert abs(r['orig']['delta']) < 1e-9 and r['orig']['thr_base'] == r['orig']['thr_i1'], (k, r['orig'])
print('compare: 같은 점수 → 네 간격 모두 Δ 0')

nb = json.load(open(os.path.join(B.NB_DIR, 'i2_longctx_judge.ipynb')))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        ast.parse('\n'.join(l for l in c['source'].split('\n') if not l.lstrip().startswith('!')))
print('smoke test 통과')
