"""AC1 smoke test (맥 CPU, API·GPU·실제 데이터 없음): 가짜 데이터로 노트북과 같은 코드를 끝까지 실행.
- 표본: dev-L(R0) 응답만, 과제별 150, test 분할이 섞이면 멈춤
- 경고 문장 표시·고쳐 쓰기 요청(tgt는 경고 응답만, blind는 전부)·같은 글 평가 한 번
- 시나리오: tgt가 경고 환각을 고침 → 채택 / tgt가 아무것도 못 고침 → AC1-1 미달 / blind가 훨씬 많이 고침 → AC1-3 미달
python experiments/builders/smoke_ac1.py"""
import ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import build_ac1 as C

n = sum(1 for c in C.cells if c['cell_type'] == 'code' and ast.parse(c['source']))
print(f'[1] ac1: 코드 셀 {n}개 문법 통과 | 임계값 {C.COMBO["threshold"]:.4f}')

# ---- 가짜 RAGTruth: 응답마다 문장 4개, 'WRONG' 문장 = 판정기가 잡는 환각, 'HIDDEN' 문장 = 못 잡는 환각 ----
rng = random.Random(0)
sources, responses, r0 = {}, [], []
tasks = ['QA', 'Summary', 'Data2txt']
for k in range(900):
    t = tasks[k % 3]; sid = f's{k}'
    sources[sid] = {'source_id': sid, 'task_type': t,
                    'source_info': {'question': f'What is item {k}?', 'passages': f'Item {k} is red. It costs {k} dollars.'} if t == 'QA'
                    else (f'Article {k} says the bridge opened in 1990.' if t == 'Summary' else {'name': f'Shop {k}', 'city': 'Austin'})}
    kind = rng.random()
    s = ['The first fact is fine.', 'The second fact is fine.', 'The third fact is fine.', 'The last fact is fine.']
    if kind < 0.25: s[1] = 'This WRONG claim is made up.'
    elif kind < 0.70: s[2] = 'This HIDDEN claim is made up.'
    responses.append({'id': str(k), 'source_id': sid, 'split': 'test' if k >= 800 else 'train', 'model': 'm', 'response': ' '.join(s)})
    if k < 800: r0.append({'id': str(k), 'task_type': t, 'ref_key': sid})

G = {'np': np, 'pd': pd, 'json': json, 'os': os, 'random': random}
exec(C.L3.L3_INPUTS + C.AC1_DATA, G)
ITEMS = G['sample_items'](r0, responses, sources)
assert len(ITEMS) == 450 and all(sum(x['task'] == t for x in ITEMS) == 150 for t in tasks)
assert all(int(x['rid']) < 800 for x in ITEMS) and ITEMS[0]['ref']
bad = r0 + [{'id': '850', 'task_type': 'QA', 'ref_key': 's850'}]
G['PER_TASK'] = 300
try:
    G['sample_items'](bad, responses, sources); raise SystemExit('test 분할 응답을 못 걸러냄')
except AssertionError:
    pass
G['PER_TASK'] = 150
print(f'[2] 표본 {len(ITEMS)}개(과제별 150), test 분할 섞이면 멈춤 확인')

exec(C.M1.M1_LABEL + C.M1.M1_SCORE + C.AC1_FLAG, G)
THR = C.COMBO['threshold']
def fake_pts(text):
    # 판정 지점 대신 문장마다 1개: 'WRONG' 문장만 임계값 이상
    return [[a, b, 0.0, 0.0, 4.0 if 'WRONG' in s else 0.5] for a, b, s in G['sentences'](text)]
FLAGS = {it['rid']: G['flagged_starts'](fake_pts(it['text']), THR) for it in ITEMS}
nf = sum(bool(v) for v in FLAGS.values())
it = next(x for x in ITEMS if FLAGS[x['rid']])
mk = G['mark'](it['text'], FLAGS[it['rid']])
assert mk.count('[CHECK] ') == 1 and '[CHECK] This WRONG' in mk and mk.replace('[CHECK] ', '') == it['text']
print(f'[3] 경고 응답 {nf}/450, 표시: {mk[:70]}...')

G['PRICE'] = {'claude-haiku-5-5': (0.05, 0.25), 'claude-opus-5-5': (2.0, 10.0)}
exec(C.AC1_REWRITE + C.AC1_EVAL + C.GRADE, G)
REQ = G['rewrite_requests'](ITEMS, FLAGS)
assert sum(k.startswith('t-') for k in REQ) == nf and sum(k.startswith('b-') for k in REQ) == 450
assert '[CHECK]' in REQ['t-' + it['rid']]['messages'][0]['content'] and '[CHECK]' not in REQ['b-' + it['rid']]['messages'][0]['content']
assert REQ['t-' + it['rid']]['model'] == 'claude-haiku-5-5' and G['LABELER'] == 'claude-sonnet-5-5'
print(f'[4] 고쳐 쓰기 요청 tgt {nf}(경고 응답만) + blind 450')

def fix(text, words):
    return ' '.join('This fact is fixed.' if any(w in s for w in words) else s for _, _, s in G['sentences'](text))

def rewrites(tgt_words, blind_words):
    RW = {}
    for k in REQ:
        x = next(i for i in ITEMS if i['rid'] == k[2:])
        w = tgt_words if k.startswith('t-') else blind_words
        RW[k] = {'text': '[CHECK] ' + fix(x['text'], w) if k.startswith('t-') else fix(x['text'], w), 'stop': 'end_turn', 'model': 'x', 'in': 1, 'out': 1}
    return RW

def evaluate(RW, loss_all=False):
    TX = G['arm_texts'](ITEMS, FLAGS, RW)
    F, Q, L = G['eval_requests'](ITEMS, TX)
    texts = {}
    for r, t in TX.items():
        for a in G['ARMS']:
            texts[G['hkey']('f', r, t[a])] = t[a]
    FR = {k: {'text': json.dumps({'sentences': [{'i': i + 1, 'label': 'H' if ('WRONG' in s or 'HIDDEN' in s) else 'S'}
                                                 for i, (_, _, s) in enumerate(G['sentences'](texts[k]))]}), 'model': 'x', 'in': 1, 'out': 1} for k in F}
    QR = {k: {'text': '{"grade": "yes"}', 'model': 'x', 'in': 1, 'out': 1} for k in Q}
    LR = {k: {'text': '{"loss": "%s"}' % ('yes' if loss_all else 'no'), 'model': 'x', 'in': 1, 'out': 1} for k in L}
    exec(C.AC1_METRICS, G)
    return TX, F, Q, L, G['judge_ac1'](G['build_rows'](ITEMS, FLAGS, TX, FR, QR, LR))

TX, F, Q, L, R = evaluate(rewrites(['WRONG'], ['WRONG']))
n_tgt_changed = sum(1 for r, t in TX.items() if t['tgt'] != t['orig'])
assert all('[CHECK]' not in t['tgt'] for t in TX.values()) and n_tgt_changed == nf
assert len(F) == 450 + nf and len(Q) == 150 + sum(1 for x in ITEMS if x['task'] == 'QA' and FLAGS[x['rid']])   # 같은 글은 한 번만
assert R['AC1'] and R['n'] == 450 and R['hal']['tgt'] == R['hal']['blind'] < R['hal']['orig'], R
print(f"[5] tgt·blind 모두 경고 환각 고침 → 채택 {R['AC1']} | 환각 {R['hal']} | 상대 {R['rel_tgt']:+.2f} | 평가 요청 충실성 {len(F)}·정답성 {len(Q)}·손실 {len(L)}")

R2 = evaluate(rewrites([], ['WRONG']))[-1]
assert R2['AC1_0'] and not R2['AC1_1'] and not R2['AC1']
R3 = evaluate(rewrites(['WRONG'], ['WRONG', 'HIDDEN']))[-1]
assert R3['AC1_1'] and not R3['AC1_3'] and R3['gain_tgt'] < 0.5 * R3['gain_blind']
R4 = evaluate(rewrites(['WRONG'], ['WRONG']), loss_all=True)[-1]
assert R4['AC1_1'] and not R4['AC1_2']
print(f"[6] tgt 못 고침 → AC1-1 {R2['AC1_1']} | blind가 더 고침 → AC1-3 {R3['AC1_3']} (tgt {R3['gain_tgt']:.3f} < blind 절반 {0.5 * R3['gain_blind']:.3f}) | 손실 100% → AC1-2 {R4['AC1_2']}")

RWb = rewrites(['WRONG'], ['WRONG']); RWb['b-' + ITEMS[0]['rid']]['stop'] = 'max_tokens'
TXb = G['arm_texts'](ITEMS, FLAGS, RWb)
assert ITEMS[0]['rid'] not in TXb and len(TXb) == 449
print('[7] 잘린 고쳐 쓰기는 그 응답을 뺌')
print('smoke_ac1 통과')
