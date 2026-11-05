"""X1 노트북의 새 셀(지점 구성·토큰→문장 점수·각 도구 실행 함수·인과성 점검·지표)을 가짜 데이터로 끝까지 돌려 보는 smoke test.
노트북과 같은 코드 문자열을 build_x1.py에서 가져와 exec한다. CPU, ModernBERT 토크나이저만 내려받음(임의 초기화 작은 모델).
실행: python experiments/builders/smoke_x1.py"""
import importlib.util, os, random, copy, sys, json, time, tempfile
import numpy as np, torch, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('build_x1', os.path.join(HERE, 'build_x1.py'))
B = importlib.util.module_from_spec(spec); _argv = sys.argv; spec.loader.exec_module(B); sys.argv = _argv

rng = random.Random(0)
WORDS = 'river city bloom india museum bridge north train school price year market festival temple'.split()
g = {'np': np, 'torch': torch, 'random': random, 'copy': copy, 'time': time, 'os': os, 'json': json, 'pd': pd,
     'DEVICE': 'cpu', 'Z0': {}, 'X1_DIR': tempfile.mkdtemp(), 'TASKS': ['QA', 'Summary']}
exec(B.SERIES, g); exec(B.EVAL_DEFS, g); exec(B.CONF, g); exec(B.X1_COMMON, g); exec(B.X1_METRICS, g)

def fake_item(i, hall):
    sents_txt = [' '.join(rng.choice(WORDS) for _ in range(n)) + '.' for n in (7, 12, 9)]
    text = ' '.join(sents_txt)
    ts, te, pos = [], [], 0
    for w in text.split(' '):
        s = text.index(w, pos); ts.append(s); te.append(s + len(w)); pos = s + len(w)
    sents, t = [], 0
    for st in sents_txt:
        n = len(st.split(' ')); sents.append({'a': ts[t], 'b': te[t + n - 1], 't0': t, 't1': t + n - 1, 'y': 0}); t += n
    onset = None
    if hall:
        onset = sents[1]['t0'] + 3; sents[1]['y'] = 1
    sent_of = {tt: sn['t0'] for sn in sents for tt in range(sn['t0'], sn['t1'] + 1)}
    x = {'id': f'x{i}', 'task_type': ['QA', 'Summary'][i % 2], 'ref_key': f'r{i % 3}', 'ts': ts, 'te': te,
         'n_tok': len(ts), 'onset': onset, 'sents': sents, 'sent_of': sent_of, 'ends': {sn['t1'] for sn in sents}}
    return x, text

items, RESP_TEXT = [], {}
for i in range(80):
    x, text = fake_item(i, i % 3 == 0); items.append(x); RESP_TEXT[x['id']] = text
# R0와 같은 질의 번호(2토큰 간격 + 문장 끝)와 F1 base 가짜 점수: 환각 문장의 발생 이후 지점만 높게
fb = []
for x in items:
    x['pts2'] = []
    for sn in x['sents']:
        for t in range(sn['t0'], sn['t1'] + 1):
            if (t - sn['t0'] + 1) % 2 == 0 or t == sn['t1']:
                hot = x['onset'] is not None and sn['y'] and t >= x['onset']
                x['pts2'].append((t, len(fb))); fb.append(rng.gauss(5 if hot else 0, 1))
g['SC']['F_base'] = np.array(fb, dtype=np.float32)
CAL, DEV = items[:40], items[40:]
g.update(RESP_TEXT=RESP_TEXT, PROMPT={x['id']: 'Answer from passages: ' + ' '.join(WORDS) for x in items},
         REFS={f'r{k}': ' '.join(rng.choice(WORDS) for _ in range(60)) for k in range(3)},
         LAB_SETS={'orig': (CAL, DEV), 'C1': (CAL, DEV)})

# 1) 지점: 문장 끝 + 8토큰, 응답 사본의 pts2는 지점만 남고 새 번호
PTS, MAP8 = g['build_points'](CAL + DEV, RESP_TEXT)
for x in items:
    want = [t for t, _ in x['pts2'] if t in x['ends'] or (t - x['sent_of'][t] + 1) % 8 == 0]
    assert [q['t'] for q in PTS if q['id'] == x['id']] == want
PTS_BY = {}
for q in PTS: PTS_BY.setdefault(q['id'], []).append(q)
g.update(PTS=PTS, MAP8=MAP8, PTS_BY=PTS_BY)
g['LAB_SETS8'] = {lab: (g['remap8'](c, MAP8), g['remap8'](d, MAP8)) for lab, (c, d) in g['LAB_SETS'].items()}
g['SC']['Fb'] = g['to_points'](CAL + DEV, 'F_base', MAP8, len(PTS))
assert np.isfinite(g['SC']['Fb']).all()
d0 = dict(items[0]['pts2']); q0 = PTS_BY['x0'][0]
assert g['SC']['Fb'][MAP8[('x0', q0['t'])]] == g['SC']['F_base'][d0[q0['t']]]

# 2) 토큰 → 문장 점수: 문장 경계 공백 토큰(" word")은 다음 문장에 배정, 지점 점수는 지점 전에 끝난 현재 문장 토큰만
x, text = items[0], RESP_TEXT['x0']
spans = [(s - (1 if s > 0 else 0), e) for s, e in zip(x['ts'], x['te'])]      # 앞 공백 포함 토큰
probs = np.linspace(0, 1, len(spans))
ss = g['sent_scores'](x, text, spans, probs)
for i, sn in enumerate(x['sents']):
    assert abs(ss[i] - probs[sn['t1']]) < 1e-6, (i, ss[i])                       # 증가 확률 → 문장 마지막 토큰
pq = [{'si': 1, 'cut': x['te'][x['sents'][1]['t0'] + 2]}, {'si': 1, 'cut': x['te'][x['sents'][0]['t1']]}]
ps = g['point_scores'](x, text, spans, probs, pq)
assert abs(ps[0] - probs[x['sents'][1]['t0'] + 2]) < 1e-6 and ps[1] == g['NO_SCORE']

# 3) LettuceDetect 실행 함수: 진짜 ModernBERT 토크나이저 + 임의 초기화 작은 토큰 분류기
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForTokenClassification
exec(B.X1_LT, g)
tok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_labels=2, pad_token_id=tok.pad_token_id, reference_compile=False)
torch.manual_seed(0); lt = ModernBertForTokenClassification(cfg).eval()
pairs = [(g['PROMPT'][x['id']], RESP_TEXT[x['id']]) for x in items[:6]] + [(g['PROMPT']['x0'], RESP_TEXT['x0'][:PTS[3]['cut']])]
res = g['lt_run'](tok, lt, pairs, max_tokens=2000)
single = [g['lt_run'](tok, lt, [p])[0] for p in pairs]
for (sp, pr, tr), (sp1, pr1, tr1), (_, a) in zip(res, single, pairs):
    assert sp == sp1 and np.allclose(pr, pr1, atol=1e-4), '묶음 채점이 단건과 다름 (패딩 처리)'
    assert ''.join(a[s:e] for s, e in sp).replace(' ', '') == a.replace(' ', ''), '답 토큰 구간이 답 전체를 덮지 않음'
    assert not tr and len(sp) == len(pr)
g['LT_MAX'] = 64
assert g['lt_run'](tok, lt, [(' '.join(WORDS * 20), RESP_TEXT['x0'])])[0][2], '프롬프트 잘림 표시 실패'
g['LT_MAX'] = 4096
SA = {'Fb': {x['id']: [float(g['SC']['F_base'][dict(x['pts2'])[sn['t1']]]) for sn in x['sents']] for x in items}}
SA['LT'] = {x['id']: g['sent_scores'](x, RESP_TEXT[x['id']], *g['lt_run'](tok, lt, [(g['PROMPT'][x['id']], RESP_TEXT[x['id']])])[0][:2]) for x in items}
BY = {x['id']: x for x in items}
def lt_points(idx):
    qs = [PTS[i] for i in idx]
    out = g['lt_run'](tok, lt, [(g['PROMPT'][q['id']], RESP_TEXT[q['id']][:q['cut']]) for q in qs])
    return [g['point_scores'](BY[q['id']], RESP_TEXT[q['id']], sp, pr, [q])[0] for q, (sp, pr, _) in zip(qs, out)]
g['SC']['LT'] = g['run_parts']('lt', len(PTS), lt_points, part=150)
again = g['run_parts']('lt', len(PTS), lambda idx: 1 / 0, part=150)                 # 저장 조각에서 이어서
assert np.array_equal(again, g['SC']['LT'])

# 4) HHEM 실행 함수: predict를 흉내 낸 가짜 모델(일치 확률 = 가설 단어가 근거에 있는 비율)
exec(B.X1_HH, g)
class FakeHH:
    def predict(self, pairs):
        return torch.tensor([np.mean([w in p for w in h.split()]) for p, h in pairs])
hp = [(g['REFS']['r0'], 'river city'), (g['REFS']['r0'], 'zzz qqq')]
v = g['hh_run'](FakeHH(), hp, max_tokens=50)
assert abs(v[1] - 1.0) < 1e-6 and v[0] <= 1.0
SA['HH'] = {x['id']: [float(s) for s in g['hh_run'](FakeHH(), [(g['REFS'][x['ref_key']], RESP_TEXT[x['id']][sn['a']:sn['b']]) for sn in x['sents']])] for x in items}
g['SC']['HH'] = np.array([g['hh_run'](FakeHH(), [(g['REFS'][BY[q['id']]['ref_key']], RESP_TEXT[q['id']][BY[q['id']]['sents'][q['si']]['a']:q['cut']])])[0] for q in PTS], dtype=np.float32)

# 5) HDM-2 실행 함수와 X1-0 점검: 앞 토큰만 보는 가짜 모델은 통과, 뒤 토큰을 보는 가짜 모델은 미달
exec(B.X1_HD, g)
class FakeHD(torch.nn.Module):
    def __init__(self, causal):
        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1)); self.causal = causal
        self.device = torch.device('cpu'); self.backbone = self
    def forward(self, input_ids, attention_mask):
        h = (input_ids.float() % 97) / 97.0
        c = torch.cumsum(h, -1) if self.causal else torch.flip(torch.cumsum(torch.flip(h, [-1]), -1), [-1])
        z = torch.sin(c)[..., None]
        return {'token_logits': torch.cat([-z, z], -1)}
for causal, ok in [(True, True), (False, False)]:
    m = FakeHD(causal)
    run = lambda p, r: g['hd_run'](m, tok, p, r)
    worst, n_cmp = g['causal_check'](run, items, n=10, per=3)
    assert n_cmp > 0 and ((worst <= 0.01) == ok), (causal, worst)
m = FakeHD(True); run = lambda p, r: g['hd_run'](m, tok, p, r)
sp, pr = run(g['PROMPT']['x0'], RESP_TEXT['x0'])
assert ''.join(RESP_TEXT['x0'][s:e] for s, e in sp).replace(' ', '') == RESP_TEXT['x0'].replace(' ', '')
SA['HD'], hd_pts = {}, np.full(len(PTS), np.nan, dtype=np.float32)
for x in items:
    sp, pr = run(g['PROMPT'][x['id']], RESP_TEXT[x['id']])
    SA['HD'][x['id']] = g['sent_scores'](x, RESP_TEXT[x['id']], sp, pr)
    for q, v in zip(PTS_BY[x['id']], g['point_scores'](x, RESP_TEXT[x['id']], sp, pr, PTS_BY[x['id']])):
        hd_pts[MAP8[(x['id'], q['t'])]] = v
assert np.isfinite(hd_pts).all()
g['SC']['HD'] = hd_pts
# 인과 모델이면 1회 통과 지점 점수 = 지점까지 잘라 넣은 입력의 지점 점수
for q in PTS[:30]:
    sp, pr = run(g['PROMPT'][q['id']], RESP_TEXT[q['id']][:q['cut']])
    assert abs(g['point_scores'](BY[q['id']], RESP_TEXT[q['id']], sp, pr, [q])[0] - hd_pts[MAP8[(q['id'], q['t'])]]) < 1e-5

# 6) 지표·부트스트랩·판정 규칙
g['SA'] = SA
g['SENT_OK'] = {x['id']: [True] * len(x['sents']) for x in items}
for m_ in ['Fb', 'LT', 'HH', 'HD']:
    ra = g['track_a']('orig', m_, 0.5)
    rb, det = g['track_b']('orig', m_)
    assert 0 <= ra['resp_auc'] <= 1 and 0 <= rb['svc_recall'] <= 1 and len(det) == sum(x['onset'] is not None for x in DEV)
    print(f"{m_:3} 트랙 A 응답 AUROC {ra['resp_auc']:.3f} 문장 {ra['sent_auc']:.3f} 서비스 {ra['svc_recall']:.2f} | "
          f"트랙 B 서비스 {rb['svc_recall']:.2f} 지연 {rb['svc_delay_med']} 엄격 {rb['strict_recall_arl100']:.2f}")
ra = g['track_a']('orig', 'Fb', 3.97)
assert ra['resp_auc'] > 0.9, 'F1 base 가짜 점수는 환각을 거의 다 구분해야 함'
d, lo, hi = g['boot_auc']('orig', 'Fb', 'HH', n=200)
assert lo <= d <= hi
_, dA = g['track_b']('orig', 'Fb'); _, dB = g['track_b']('orig', 'LT')
d, lo, hi = g['boot_det'](dA, dB, n=200)
assert lo <= d <= hi
assert g['verdict'](0.01, 0.1) == '앞섬' and g['verdict'](-0.1, -0.01).startswith('뒤짐') and g['verdict'](-0.1, 0.1) == '차이 없음'
print('smoke_x1 통과')
