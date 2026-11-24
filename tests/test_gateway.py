"""게이트웨이 단위·통합 테스트 (모델 다운로드 없이 실행 가능).

  cd gateway && python -m pytest -q
"""
import json
import os
import random
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallugate import Settings, create_app                      # noqa: E402
from hallugate.context import extract                            # noqa: E402
from hallugate.judge import LexicalJudge                         # noqa: E402
from hallugate.splitter import IncrementalSplitter               # noqa: E402
import mock_upstream                                             # noqa: E402

REF = 'The company was founded in 1998 in Seoul. It reported revenue of 2 billion dollars in 2023.'
GOOD = 'The company was founded in 1998 in Seoul. It reported revenue of 2 billion dollars in 2023.'
BAD_2ND = 'The company was founded in 1998 in Seoul. Its chief executive flew rockets across Mars yesterday. It reported revenue of 2 billion dollars in 2023.'


# ---------------- 문장 분할 ----------------
@pytest.mark.parametrize('text', [
    GOOD, BAD_2ND,
    'Dr. Kim arrived at 3.5 p.m. on Jan. 5. He left early!  Then what?\n\nNew paragraph here.',
    'No terminal punctuation at all',
])
def test_splitter_streaming_matches_whole(text):
    rng = random.Random(0)
    sp, got, i = IncrementalSplitter(), [], 0
    while i < len(text):
        j = i + rng.randint(1, 6)
        got += sp.feed(text[i:j])
        i = j
    got += sp.flush()
    assert ''.join(s.text for s in got) == text                    # 글자 하나도 잃지 않음
    assert [s.start for s in got] == [sum(len(x.text) for x in got[:k]) for k in range(len(got))]
    whole = IncrementalSplitter()
    ref = whole.feed(text) + whole.flush()
    assert [s.clean for s in got] == [s.clean for s in ref]        # 조각 단위로 받아도 한 번에 받은 것과 같음


# ---------------- 컨텍스트 추출 ----------------
def test_extract_field_and_strip():
    body = {'model': 'x', 'messages': [], 'verification': {'context': ['a', 'b'], 'mode': 'hold', 'action': 'cut'}}
    up, ref, mode, action = extract(body)
    assert 'verification' not in up and ref == 'a\n\nb' and (mode, action) == ('hold', 'cut')


def test_extract_tag_and_off():
    body = {'messages': [{'role': 'user', 'content': 'Q?\n<context>doc one</context> <context>doc two</context>'}]}
    assert extract(body)[1:3] == ('doc one\n\ndoc two', 'pass')
    assert extract({'messages': [{'role': 'user', 'content': 'hi'}]})[2] == 'off'
    with pytest.raises(ValueError):
        extract({'messages': [], 'verification': {'context': 'x', 'mode': 'bogus'}})


# ---------------- 게이트웨이 통합 (가짜 LLM 연결) ----------------
@pytest.fixture
def gw(tmp_path):
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path=str(tmp_path / 'm.jsonl'), stride_tokenizer='words')
    app = create_app(s, judge=LexicalJudge(threshold=0.5), http_client=up)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gw'), s


def body(resp, mode, action='flag', stream=True):
    return {'model': 'mock', 'stream': stream, 'messages': [{'role': 'user', 'content': 'q'}],
            'verification': {'context': REF, 'mode': mode, 'action': action},
            'mock_response': resp, 'mock_ttft_ms': 0, 'mock_token_ms': 0}


async def run(client, b):
    r = await client.post('/v1/chat/completions', json=b)
    assert r.status_code == 200
    text, events, finish = '', [], None
    for line in r.text.splitlines():
        if not line.startswith('data:') or line.endswith('[DONE]'):
            continue
        c = json.loads(line[5:])
        if c.get('verification'):
            events.append(('v', c['verification']))
        for ch in c['choices']:
            d = ch['delta'].get('content')
            if d:
                text += d
                events.append(('t', d))
            finish = ch.get('finish_reason') or finish
    return text, events, finish


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pass', 'hold'])
async def test_stream_text_intact_and_verdicts(gw, mode):
    client, s = gw
    text, ev, finish = await run(client, body(BAD_2ND, mode))
    assert text == BAD_2ND and finish == 'stop'
    vs = [e for k, e in ev if k == 'v']
    assert [v['sentence_idx'] for v in vs] == [0, 1, 2]
    assert [v['hallucinated'] for v in vs] == [False, True, False]
    rec = json.loads(open(s.metrics_path).read().splitlines()[-1])
    assert rec['mode'] == mode and len(rec['sentences']) == 3
    if mode == 'hold':
        # hold: 각 문장의 판정이 그 문장 본문보다 먼저 나가야 함 → 누출 0
        order = [k for k, _ in ev]
        assert order == ['v', 't', 'v', 't', 'v', 't']
        assert all(x.get('leak_chars', 0) == 0 for x in rec['sentences'])


@pytest.mark.asyncio
async def test_hold_cut_blocks_bad_sentence(gw):
    client, _ = gw
    text, ev, finish = await run(client, body(BAD_2ND, 'hold', 'cut'))
    assert finish == 'content_filter'
    assert text == 'The company was founded in 1998 in Seoul. '     # 틀린 문장은 한 글자도 안 나감
    assert 'rockets' not in text


# ---------------- 실제 서버 (시간 간격이 있는 스트리밍) ----------------
# ASGITransport는 응답을 한 번에 모아서 주므로, 토큰 간격이 중요한 테스트는 실제 서버로 한다.
def _free_port():
    import socket
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def live(tmp_path):
    import threading, time as _t, uvicorn
    servers = []

    def start(app):
        port = _free_port()
        srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error'))
        th = threading.Thread(target=srv.run, daemon=True); th.start()
        while not srv.started:
            _t.sleep(0.02)
        servers.append(srv)
        return f'http://127.0.0.1:{port}'

    up = start(mock_upstream.app)
    s = Settings(upstream_base_url=up + '/v1', metrics_path=str(tmp_path / 'm.jsonl'), stride_tokenizer='words')
    gw_url = start(create_app(s, judge=LexicalJudge(threshold=0.5, delay_ms=5)))
    yield gw_url, s
    for srv in servers:
        srv.should_exit = True


async def run_live(url, b):
    """(시각ms, 종류, 값) 이벤트 목록"""
    import time as _t
    out, t0 = [], _t.perf_counter()
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream('POST', url + '/v1/chat/completions', json=b) as r:
            async for line in r.aiter_lines():
                if not line.startswith('data:') or line.endswith('[DONE]'):
                    continue
                ch = json.loads(line[5:]); now = (_t.perf_counter() - t0) * 1000
                if ch.get('verification'):
                    out.append((now, 'v', ch['verification']))
                for x in ch['choices']:
                    if x['delta'].get('content'):
                        out.append((now, 't', x['delta']['content']))
                    if x.get('finish_reason'):
                        out.append((now, 'f', x['finish_reason']))
    return out


@pytest.mark.asyncio
async def test_live_pass_cut_stops_after_verdict(live):
    url, _ = live
    b = body(BAD_2ND, 'pass', 'cut'); b['mock_token_ms'] = 20
    ev = await run_live(url, b)
    text = ''.join(v for _, k, v in ev if k == 't')
    assert ('f', 'content_filter') in [(k, v) for _, k, v in ev]
    assert 'rockets' in text and '2023' not in text        # 이미 나간 건 못 막지만 이후는 끊김


@pytest.mark.asyncio
async def test_live_pass_verdict_arrives_without_waiting_next_token(live):
    url, s = live
    b = body(BAD_2ND, 'pass'); b['mock_token_ms'] = 50
    ev = await run_live(url, b)
    rec = json.loads(open(s.metrics_path).read().splitlines()[-1])
    lags = [x['t_verdict'] - x['t_complete'] for x in rec['sentences']]
    assert max(lags) < 40, lags                              # 다음 토큰(50ms)을 기다리지 않고 바로 판정 전달


@pytest.mark.asyncio
async def test_live_hold_delays_but_no_leak(live):
    url, s = live
    for mode in ['off', 'hold']:
        b = body(BAD_2ND, mode); b['mock_token_ms'] = 20
        if mode == 'off':
            b.pop('verification')
        ev = await run_live(url, b)
        text = ''.join(v for _, k, v in ev if k == 't')
        assert text == BAD_2ND
        first = next(t for t, k, _ in ev if k == 't')
        if mode == 'off':
            off_first = first
        else:
            assert first > off_first + 100                   # 첫 문장이 다 만들어질 때까지 보류됨


@pytest.mark.asyncio
async def test_off_is_passthrough(gw):
    client, _ = gw
    b = body(GOOD, 'pass')
    b.pop('verification')
    r = await client.post('/v1/chat/completions', json=b)
    assert r.headers['x-verification-mode'] == 'off'
    assert 'verification' not in r.text


@pytest.mark.asyncio
async def test_non_stream(gw):
    client, _ = gw
    r = await client.post('/v1/chat/completions', json=body(BAD_2ND, 'hold', 'cut', stream=False))
    d = r.json()
    assert d['choices'][0]['finish_reason'] == 'content_filter'
    assert d['choices'][0]['message']['content'] == 'The company was founded in 1998 in Seoul. '
    assert [x['hallucinated'] for x in d['verification']['sentences']] == [False, True]


@pytest.mark.asyncio
async def test_openai_sdk_compatible(gw):
    from openai import AsyncOpenAI
    client, _ = gw
    oai = AsyncOpenAI(base_url='http://gw/v1', api_key='x', http_client=client)
    stream = await oai.chat.completions.create(
        model='mock', messages=[{'role': 'user', 'content': 'q'}], stream=True,
        extra_body={'verification': {'context': REF, 'mode': 'hold'}, 'mock_response': BAD_2ND,
                    'mock_ttft_ms': 0, 'mock_token_ms': 0})
    text, flags = '', []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
        v = getattr(chunk, 'verification', None)
        if v:
            flags.append(v['hallucinated'])
    assert text == BAD_2ND and flags == [False, True, False]


# ---------------- 학습된 판정기 로딩 경로 (작은 임의 모델) ----------------
def test_student_judge_loads(tmp_path):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from hallugate.judge import StudentJudge
    words = 'the company was founded in seoul revenue billion dollars reported it'.split()
    (tmp_path / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '...', '.'] + words))
    BertTokenizerFast(str(tmp_path / 'vocab.txt')).save_pretrained(tmp_path)
    BertForSequenceClassification(BertConfig(vocab_size=len(words) + 7, hidden_size=16, num_hidden_layers=1,
                                             num_attention_heads=2, intermediate_size=32, num_labels=1)
                                  ).save_pretrained(tmp_path)

    class Emb:
        def encode(self, xs, **k):
            return torch.nn.functional.normalize(torch.randn(len(xs), 8), dim=-1)

    j = StudentJudge(str(tmp_path), embedder=Emb(), k=8, device='cpu')
    ref = j.prepare(REF * 5)
    v = j.judge(ref, 'The company was founded in 1998.')
    assert isinstance(v.score, float) and v.judge_ms > 0


# ---------------- 앞부분 재판정 (stride, SPEC §5.1) ----------------
def test_splitter_pending_tracks_current_sentence():
    sp = IncrementalSplitter()
    sp.feed('First sentence here. Second sen')
    off, buf = sp.pending()
    assert off == len('First sentence here. ') and buf == 'Second sen'


def test_extract_stride():
    body = {'messages': [], 'verification': {'context': 'x', 'stride': 4}}
    assert extract(body, with_stride=True)[4] == 4
    assert extract({'messages': [], 'verification': {'context': 'x'}}, default_stride=2, with_stride=True)[4] == 2
    with pytest.raises(ValueError):
        extract({'messages': [], 'verification': {'context': 'x', 'stride': -1}}, with_stride=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pass', 'hold'])
async def test_stride_text_intact_and_prefix_verdicts(gw, mode):
    client, s = gw
    b = body(BAD_2ND, mode); b['verification']['stride'] = 2
    text, ev, finish = await run(client, b)
    assert text == BAD_2ND and finish == 'stop'                    # 본문 보존 (G2)
    vs = [e for k, e in ev if k == 'v']
    assert {'prefix', 'sentence'} <= {v['kind'] for v in vs}
    assert all(v['start'] < v['upto'] and 'released_at_verdict' in v for v in vs)
    # 두 번째 문장은 앞부분 판정에서 이미 환각으로 잡힘 (문장 판정보다 먼저)
    first_alarm = next(v for v in vs if v['hallucinated'])
    assert first_alarm['kind'] == 'prefix' and first_alarm['sentence_idx'] == 1
    rec = json.loads(open(s.metrics_path).read().splitlines()[-1])
    assert rec['stride'] == 2 and len(rec['checks']) > 0


@pytest.mark.asyncio
async def test_stride_hold_releases_safe_prefix_before_sentence_end(gw):
    client, _ = gw
    b = body(GOOD, 'hold'); b['verification']['stride'] = 2
    text, ev, _ = await run(client, b)
    assert text == GOOD
    # 첫 문장의 문장 판정('sentence') 이전에, 안전한 앞부분이 이미 전달돼야 한다
    first_sent_verdict = next(i for i, (k, e) in enumerate(ev) if k == 'v' and e['kind'] == 'sentence')
    texts_before = [e for k, e in ev[:first_sent_verdict] if k == 't']
    assert texts_before, '문장 판정 전에 아무것도 전달되지 않음'
    # 첫 전달은 문장 전체가 아니라 앞부분이어야 한다 (문장 끝까지 기다리지 않음)
    assert len(texts_before[0]) < len('The company was founded in 1998 in Seoul. ')


@pytest.mark.asyncio
async def test_stride_hold_cut_stops_inside_bad_sentence(gw):
    client, _ = gw
    b = body(BAD_2ND, 'hold', 'cut'); b['verification']['stride'] = 2
    text, ev, finish = await run(client, b)
    assert finish == 'content_filter'
    assert text == 'The company was founded in 1998 in Seoul. '     # 환각 문장은 한 글자도 나가지 않음
    cut = [e for k, e in ev if k == 'v' and e.get('cut')]
    assert cut and cut[0]['kind'] == 'prefix'


@pytest.mark.asyncio
async def test_live_stride_hold_reduces_wait(live):
    # 실제 서버: 앞부분 재판정을 켜면 hold의 첫 글자 지연이 문장 길이보다 짧아진다
    url, _ = live
    firsts = {}
    for stride in [0, 2]:
        b = body(BAD_2ND, 'hold'); b['mock_token_ms'] = 20; b['verification']['stride'] = stride
        ev = await run_live(url, b)
        assert ''.join(v for _, k, v in ev if k == 't') == BAD_2ND
        firsts[stride] = next(t for t, k, _ in ev if k == 't')
    assert firsts[2] < firsts[0] - 60, firsts


# ---------------- S4: 판정기 설정 파일, 토큰 단위 재판정, 통과 청크, fail-open ----------------
def _tiny_bert(path, words):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    (path / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '...', '.'] + words))
    BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    BertForSequenceClassification(BertConfig(vocab_size=len(words) + 7, hidden_size=16, num_hidden_layers=1,
                                             num_attention_heads=2, intermediate_size=32, num_labels=1)
                                  ).save_pretrained(path)


def test_load_judge_reads_config_full_mode(tmp_path):
    # I2 노트북 형식(input.retrieval = null) → 근거 전체 + 응답 앞부분 입력, 간격별 임계값, 기본 간격 8
    from hallugate.judge import load_judge
    _tiny_bert(tmp_path, 'the company was founded in seoul revenue billion dollars reported it'.split())
    cfg = {'input': {'text_a': 'full reference', 'text_b': 'response prefix', 'retrieval': None}, 'max_len': 64,
           'default': {'kind': '8', 'threshold': 'svc10_orig'}, 'threshold': 1.5,
           'thresholds': {'end': {'svc10_orig': 0.5}, '8': {'svc10_orig': 1.5}, '4': {'svc10_orig': 2.5}}}
    (tmp_path / 'judge_config.json').write_text(json.dumps(cfg))
    j = load_judge(Settings(judge_model_dir=str(tmp_path), judge_device='cpu'))
    assert j.mode == 'full' and j.default_stride == 8 and j.max_len == 64
    assert (j.threshold_for(8), j.threshold_for(4), j.threshold_for(0), j.threshold_for(16)) == (1.5, 2.5, 0.5, 1.5)
    ref = j.prepare(REF)
    assert ref.emb is None and ref.text == REF                       # 근거 검색 없음
    a = j.judge(ref, 'It reported revenue.', prior='The company was founded in 1998 in Seoul. ')
    b = j.judge(ref, 'It reported revenue.')
    assert a.score != b.score                                        # 응답 앞부분(prior)이 입력에 들어감
    # 환경 변수로 임계값을 고정하면 간격별 임계값은 쓰지 않는다
    j2 = load_judge(Settings(judge_model_dir=str(tmp_path), judge_device='cpu', judge_threshold=9.0))
    assert j2.threshold_for(4) == 9.0


def test_word_tokens_cover_text():
    from hallugate.tokens import WordTokens
    t = 'The company  was\nfounded. '
    offs = WordTokens().offsets(t)
    assert ''.join(t[a:b] for a, b in offs) == t and offs[0] == (0, 3)


class RecordingJudge(LexicalJudge):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def judge(self, ref, text, prior='', threshold=None):
        self.calls.append((text, prior, threshold))
        return super().judge(ref, text, prior, threshold)


def _gw_with(tmp_path, judge, **kw):
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path=str(tmp_path / 'm.jsonl'), stride_tokenizer='words', **kw)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(s, judge=judge, http_client=up)),
                             base_url='http://gw'), s


@pytest.mark.asyncio
async def test_token_stride_points_and_prior(tmp_path):
    # 문장 시작부터 토큰 2개마다 앞부분 판정, 판정 텍스트는 토큰 경계에서 끝나고, prior는 그 문장 앞까지의 응답
    j = RecordingJudge(threshold=2.0)                                   # 환각 판정 없음
    client, _ = _gw_with(tmp_path, j)
    b = body(GOOD, 'hold'); b['verification']['stride'] = 2
    text, ev, _ = await run(client, b)
    assert text == GOOD
    prefixes = [(t, p) for t, p, _ in j.calls]
    assert ('The company', '') in prefixes and ('The company was founded', '') in prefixes
    assert ('It reported', 'The company was founded in 1998 in Seoul. ') in prefixes
    assert ('The company was', '') not in prefixes                   # 홀수 토큰 지점은 판정하지 않음


@pytest.mark.asyncio
async def test_default_stride_and_threshold_from_judge(tmp_path):
    class J(RecordingJudge):
        default_stride = 4
        def threshold_for(self, stride, key=None):
            return {0: 0.7, 4: 0.6}[stride]
    j = J()
    client, s = _gw_with(tmp_path, j)
    b = body(GOOD, 'hold'); b['verification'].pop('mode')
    r = await client.post('/v1/chat/completions', json=b)
    assert r.headers['x-verification-mode'] == 'hold' and r.headers['x-verification-stride'] == '4'
    assert {th for _, _, th in j.calls} == {0.6}
    rec = json.loads(open(s.metrics_path).read().splitlines()[-1])
    assert rec['threshold'] == 0.6 and rec['verified'] is True
    h = (await client.get('/health')).json()
    assert h['stride'] == 4 and h['threshold'] == 0.6 and h['mode'] == 'hold'
    r = await client.post('/v1/chat/completions', json=body(GOOD, 'hold', stream=False))
    assert r.json()['verification']['threshold'] == 0.7             # 비스트리밍은 문장 끝 임계값


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pass', 'hold'])
async def test_tool_calls_and_usage_pass_through(gw, mode):
    client, _ = gw
    b = body(GOOD, mode); b['mock_tool_call'] = 'lookup'; b['stream_options'] = {'include_usage': True}
    r = await client.post('/v1/chat/completions', json=b)
    chunks = [json.loads(l[5:]) for l in r.text.splitlines() if l.startswith('data:') and not l.endswith('[DONE]')]
    text = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks if c['choices'])
    tools = [c['choices'][0]['delta']['tool_calls'][0] for c in chunks if c['choices'] and c['choices'][0]['delta'].get('tool_calls')]
    finish = [c['choices'][0]['finish_reason'] for c in chunks if c['choices'] and c['choices'][0]['finish_reason']]
    assert text == GOOD
    assert tools[0]['function']['name'] == 'lookup' and tools[1]['function']['arguments'] == '{"q": 1}'
    assert finish == ['tool_calls']
    assert chunks[-1]['usage']['total_tokens'] == 10 and chunks[-1]['choices'] == []   # usage는 맨 끝


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pass', 'hold'])
async def test_fail_open_on_judge_error(tmp_path, mode):
    client, s = _gw_with(tmp_path, LexicalJudge(threshold=0.5, fail=lambda t: 'rockets' in t))
    text, ev, finish = await run(client, body(BAD_2ND, mode, 'cut'))
    assert text == BAD_2ND and finish == 'stop'                      # 본문은 끝까지, 끊지 않음
    vs = [e for k, e in ev if k == 'v']
    un = [v for v in vs if v.get('kind') == 'unverified']
    assert len(un) == 1 and un[0]['verified'] is False and un[0]['reason'] == 'judge_error'
    assert [v['sentence_idx'] for v in vs if v.get('kind') == 'sentence'] == [0]   # 장애 뒤로는 판정 안 함
    rec = json.loads(open(s.metrics_path).read().splitlines()[-1])
    assert rec['verified'] is False and rec['unverified_reason'] == 'judge_error'


@pytest.mark.asyncio
async def test_fail_open_on_timeout(tmp_path):
    client, _ = _gw_with(tmp_path, LexicalJudge(threshold=0.5, delay_ms=300), judge_timeout_ms=50)
    text, ev, _ = await run(client, body(BAD_2ND, 'hold'))
    assert text == BAD_2ND
    assert [e['reason'] for k, e in ev if k == 'v' and e.get('kind') == 'unverified'] == ['judge_timeout']


@pytest.mark.asyncio
async def test_non_english_unverified(gw):
    client, _ = gw
    ko = '이 회사는 1998년 서울에서 설립되었다. 2023년 매출은 20억 달러였다.'
    text, ev, _ = await run(client, body(ko, 'hold'))
    assert text == ko
    vs = [e for k, e in ev if k == 'v']
    assert len(vs) == 1 and vs[0]['reason'] == 'non_english'
    r = await client.post('/v1/chat/completions', json=body(ko, 'hold', stream=False))
    v = r.json()['verification']
    assert v['verified'] is False and v['reason'] == 'non_english' and v['sentences'] == []


# 실제 배포 가중치로 고정 예제 점수 일치(A11). 가중치는 저장소에 없으므로 JUDGE_PARITY_DIR을 지정했을 때만 실행
@pytest.mark.skipif(not os.environ.get('JUDGE_PARITY_DIR'), reason='JUDGE_PARITY_DIR 미지정 (가중치 필요)')
def test_parity_fixture():
    from hallugate.judge import load_judge
    d = os.environ['JUDGE_PARITY_DIR']
    fx = json.load(open(os.path.join(d, 'parity_fixture.json')))
    j = load_judge(Settings(judge_model_dir=d, judge_device='cpu'))
    diffs = [abs(j.judge(j.prepare(it['reference']), it['text']).score - it['score_cpu_fp32']) for it in fx['items']]
    assert max(diffs) <= 0.2, max(diffs)


# ---------------- 컬렉션·RAG (SPEC §2.5·2.6, A9) ----------------
class HashEmb:
    """테스트용 임베더: 단어 해시 묶음 벡터 (같은 단어가 많을수록 가깝다)."""
    def encode(self, texts, **kw):
        import numpy as np
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace('.', ' ').replace('?', ' ').split():
                out[i, sum(map(ord, w)) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


DOC_A = 'The company was founded in 1998 in Seoul. It reported revenue of 2 billion dollars in 2023.'
DOC_B = 'Penguins live mostly in the southern hemisphere and cannot fly but swim very well.'


def _store(tmp_path):
    from hallugate.store import SQLiteStore
    return SQLiteStore(str(tmp_path / 'c.sqlite3'), HashEmb(), chunk_words=12, overlap=2)


def test_store_add_search_delete(tmp_path):
    st = _store(tmp_path)
    a = st.add('kb', DOC_A, title='company', doc_id='a')
    st.add('kb', DOC_B, title='penguin', doc_id='b')
    assert a['n_chunks'] >= 2 and [d['doc_id'] for d in st.list('kb')] == ['a', 'b']
    hit = st.search('kb', 'When was the company founded in Seoul?', 2)
    assert hit[0].doc_id == 'a' and hit[0].score >= hit[1].score
    st.add('kb', 'Replaced text about owls.', doc_id='b')                 # 같은 id → 교체
    assert len(st.list('kb')) == 2 and st.list('kb')[1]['n_chunks'] == 1
    assert st.delete('kb', 'a') and not st.delete('kb', 'a')
    assert st.collections() == [{'collection': 'kb', 'n_docs': 1}] and not st.exists('nope')
    with pytest.raises(ValueError):
        st.add('kb', '   ')


class CapJudge(LexicalJudge):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.refs = []

    def prepare(self, reference):
        self.refs.append(reference)
        return super().prepare(reference)


def _rag_gw(tmp_path, judge=None, **kw):
    sent = []

    async def hook(req):
        sent.append(json.loads(req.content))
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock',
                           event_hooks={'request': [hook]})
    s = Settings(upstream_base_url='http://mock/v1', metrics_path=str(tmp_path / 'm.jsonl'), stride_tokenizer='words',
                 collection_top_k=2, **kw)
    st = _store(tmp_path)
    app = create_app(s, judge=judge or CapJudge(threshold=0.5), http_client=up, store=st)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gw'), s, st, sent


def rag_body(resp, mode='hold', stream=True, coll='kb', question='When was the company founded and what was its revenue?'):
    return {'model': 'mock', 'stream': stream, 'messages': [{'role': 'system', 'content': 'Be brief.'},
                                                             {'role': 'user', 'content': question}],
            'verification': {'collection': coll, 'mode': mode}, 'mock_response': resp,
            'mock_ttft_ms': 0, 'mock_token_ms': 0}


@pytest.mark.asyncio
async def test_collection_api_auth_and_crud(tmp_path):
    client, *_ = _rag_gw(tmp_path, admin_token='secret')
    url = '/v1/collections/kb/documents'
    assert (await client.post(url, json={'text': DOC_A})).status_code == 401
    H = {'Authorization': 'Bearer secret'}
    r = await client.post(url, json={'documents': [{'text': DOC_A, 'doc_id': 'a'}, {'text': DOC_B, 'title': 'p'}]}, headers=H)
    assert r.status_code == 200 and [d['n_chunks'] > 0 for d in r.json()['data']] == [True, True]
    assert (await client.post(url, json={'text': ''}, headers=H)).status_code == 400
    assert len((await client.get(url, headers=H)).json()['data']) == 2
    assert (await client.get('/v1/collections', headers=H)).json()['data'] == [{'collection': 'kb', 'n_docs': 2}]
    assert (await client.delete(url + '/a', headers=H)).json()['deleted'] is True
    assert (await client.delete(url + '/a', headers=H)).status_code == 404
    h = (await client.get('/health')).json()
    assert h['collections'] and h['collections_auth']


@pytest.mark.asyncio
async def test_collections_disabled(tmp_path):
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path='', stride_tokenizer='words', collection_db='')
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(s, judge=LexicalJudge(), http_client=up)),
                               base_url='http://gw')
    assert (await client.get('/v1/collections')).status_code == 503
    assert (await client.post('/v1/chat/completions', json=rag_body(GOOD))).status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pass', 'hold'])
async def test_rag_same_evidence_for_llm_and_judge(tmp_path, mode):
    # A9: LLM 프롬프트에 넣은 근거 = 판정 근거, 본문 보존, 사용한 근거를 본문보다 먼저 알림
    j = CapJudge(threshold=0.5)
    client, s, st, sent = _rag_gw(tmp_path, judge=j)
    st.add('kb', DOC_A, title='company', doc_id='a'); st.add('kb', DOC_B, title='penguin', doc_id='b')
    text, ev, finish = await run(client, rag_body(BAD_2ND, mode))
    assert text == BAD_2ND and finish == 'stop'
    vs = [e for k, e in ev if k == 'v']
    assert ev[0][0] == 'v' and vs[0]['kind'] == 'evidence' and vs[0]['collection'] == 'kb'
    evid = [e['text'] for e in vs[0]['evidence']]
    assert len(evid) == 2 and vs[0]['evidence'][0]['doc_id'] == 'a'
    msgs = sent[-1]['messages']
    assert msgs[0]['role'] == 'system' and all(t in msgs[0]['content'] for t in evid)
    assert msgs[1:] == rag_body(GOOD)['messages']                     # 서비스 메시지는 그대로
    assert 'verification' not in sent[-1]
    assert j.refs[-1] == '\n\n'.join(evid)                            # 판정 근거 = 프롬프트 근거
    sents = [v for v in vs if v.get('kind') == 'sentence']
    assert [v['hallucinated'] for v in sents] == [False, True, False]
    rec = json.loads(open(s.metrics_path).read().splitlines()[-1])
    assert rec['collection'] == 'kb' and rec['n_evidence'] == 2


@pytest.mark.asyncio
async def test_rag_non_stream_and_errors(tmp_path):
    client, _, st, sent = _rag_gw(tmp_path)
    st.add('kb', DOC_A, doc_id='a')
    r = await client.post('/v1/chat/completions', json=rag_body(BAD_2ND, 'hold', stream=False))
    v = r.json()['verification']
    assert v['collection'] == 'kb' and v['evidence'][0]['doc_id'] == 'a' and r.headers['x-verification-evidence'] == '2'
    assert [x['hallucinated'] for x in v['sentences']] == [False, True, False]
    assert (await client.post('/v1/chat/completions', json=rag_body(GOOD, coll='missing'))).status_code == 404
    # 요청에 근거를 직접 담으면 그것이 우선, 컬렉션 검색·프롬프트 삽입 안 함
    b = rag_body(GOOD); b['verification']['context'] = REF
    n = len(sent)
    await run(client, b)
    assert sent[n]['messages'] == b['messages']


# ---------------- 조합 판정기 (SPEC §4 ComboJudge, A13) ----------------
def _combo_dir(tmp_path, thresholds=None):
    words = 'the company was founded in seoul revenue billion dollars reported it'.split()
    member_cfg = {'input': {'retrieval': None}, 'max_len': 64, 'default': {'kind': 'end'}, 'threshold': 99.0}
    for name in ['f', 'i']:
        d = tmp_path / name; d.mkdir()
        _tiny_bert(d, words)
        (d / 'judge_config.json').write_text(json.dumps(member_cfg))
    cfg = {'type': 'combo', 'version': 'test', 'members': {'F': 'f', 'I': str(tmp_path / 'i')},   # 상대·절대 경로 모두
           'standardize': {'F': {'mean': -1.0, 'std': 2.0}, 'I': {'mean': 0.5, 'std': 4.0}},
           'default': {'kind': '8', 'threshold': 'svcB'},
           'thresholds': thresholds if thresholds is not None else {'8': {'svcB': 0.25, 'svc10_orig': 0.4}}, 'threshold': 0.3}
    (tmp_path / 'judge_config.json').write_text(json.dumps(cfg))
    return tmp_path


def test_combo_judge_standardized_mean(tmp_path):
    from hallugate.judge import ComboJudge, load_judge
    d = _combo_dir(tmp_path)
    j = load_judge(Settings(judge_model_dir=str(d), judge_device='cpu'))
    assert isinstance(j, ComboJudge) and j.mode == 'combo' and j.default_stride == 8
    assert (j.threshold_for(8), j.threshold_for(8, 'svc10_orig'), j.threshold_for(0), j.threshold_for(4)) == (0.25, 0.4, 0.3, 0.3)
    jf = load_judge(Settings(judge_model_dir=str(d / 'f'), judge_device='cpu'))
    ji = load_judge(Settings(judge_model_dir=str(d / 'i'), judge_device='cpu'))
    prior, text = 'The company was founded in 1998 in Seoul. ', 'It reported revenue.'
    sf = jf.judge(jf.prepare(REF), text, prior).score
    si = ji.judge(ji.prepare(REF), text, prior).score
    want = ((sf + 1.0) / 2.0 + (si - 0.5) / 4.0) / 2                 # 노트북 combine과 같은 식
    v = j.judge(j.prepare(REF), text, prior)
    assert abs(v.score - want) < 1e-5 and v.judge_ms > 0
    assert v.hallucinated == (v.score >= 0.3)                         # threshold 미지정이면 조합 기본 임계값
    assert j.judge(j.prepare(REF), text, prior, threshold=v.score - 1).hallucinated
    # 하위 판정기의 임계값(99)·간격(end)은 쓰지 않는다. 환경 변수 임계값은 조합에 적용
    j2 = load_judge(Settings(judge_model_dir=str(d), judge_device='cpu', judge_threshold=7.0))
    assert j2.threshold_for(8) == 7.0 and j2.threshold_for(0) == 7.0


def test_combo_judge_missing_member_fails(tmp_path):
    from hallugate.judge import load_judge
    d = _combo_dir(tmp_path)
    cfg = json.loads((d / 'judge_config.json').read_text()); cfg['members']['I'] = 'nope'
    (d / 'judge_config.json').write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError):
        load_judge(Settings(judge_model_dir=str(d), judge_device='cpu'))


class _Shift(LexicalJudge):
    """조합 게이트웨이 테스트용: 대체 판정기 점수에 상수를 더한다 (두 하위 판정기 점수를 다르게)."""
    def __init__(self, add, **kw):
        super().__init__(**kw); self.add = add

    def judge(self, ref, text, prior='', threshold=None):
        v = super().judge(ref, text, prior, threshold)
        v.score += self.add
        return v


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pass', 'hold'])
async def test_combo_judge_in_gateway(tmp_path, mode):
    from hallugate.judge import ComboJudge
    j = ComboJudge([('F', _Shift(0.0), 0.0, 1.0), ('I', _Shift(1.0), 1.0, 2.0)], thresholds={}, threshold=0.25)
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path='', stride_tokenizer='words', collection_db='')
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(s, judge=j, http_client=up)), base_url='http://gw')
    text, ev, finish = await run(client, body(BAD_2ND, mode))
    assert text == BAD_2ND and finish == 'stop'                       # 본문 보존
    vs = [e for k, e in ev if k == 'v']
    lex = LexicalJudge(); r = lex.prepare(REF)
    for v in vs:                                                      # 조합 점수 = (s/1 + (s+1−1)/2)/2 = 0.75·s
        assert abs(v['score'] - 0.75 * lex.judge(r, v['text']).score) < 1e-6
    assert [v['hallucinated'] for v in vs] == [False, True, False]
    h = (await client.get('/health')).json()
    assert h['judge'] == 'ComboJudge' and h['input_mode'] == 'combo' and h['threshold'] == 0.25


@pytest.mark.asyncio
async def test_combo_member_failure_fails_open(tmp_path):
    from hallugate.judge import ComboJudge
    bad = LexicalJudge(fail=lambda t: 'rockets' in t)
    j = ComboJudge([('F', LexicalJudge(), 0.0, 1.0), ('I', bad, 0.0, 1.0)], thresholds={}, threshold=0.5)
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path='', stride_tokenizer='words', collection_db='')
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(s, judge=j, http_client=up)), base_url='http://gw')
    text, ev, _ = await run(client, body(BAD_2ND, 'hold'))
    assert text == BAD_2ND                                            # 하위 판정기 하나가 실패해도 본문은 그대로
    assert any(e.get('verified') is False and e.get('reason') == 'judge_error' for k, e in ev if k == 'v')


def test_combo_config_v1_matches_analysis():
    # 저장소의 배포 조합 설정이 실험 결과 파일(I3 표준화 상수, K2 서비스 임계값 B)과 같은지
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, 'configs', 'judge_combo_v1.json')):
        pytest.skip('저장소 설정·분석 파일이 없는 환경(S3 노트북)')
    cfg = json.load(open(os.path.join(root, 'configs', 'judge_combo_v1.json')))
    i3 = json.load(open(os.path.join(root, 'docs', 'analysis', 'i3', 'i3_summary.json')))['standardize']
    k2 = json.load(open(os.path.join(root, 'docs', 'analysis', 'k2', 'k2b', 'k2_summary.json')))['thr']['old']
    assert cfg['type'] == 'combo' and set(cfg['members']) == set(cfg['standardize']) == {'F', 'I'}
    assert cfg['standardize']['F'] == i3['Fb'] and cfg['standardize']['I'] == i3['I2']
    assert cfg['thresholds']['8'] == {'svcB': k2['B'], 'svc10_orig': k2['A']} and cfg['threshold'] == k2['B']
    assert cfg['default'] == {'kind': '8', 'threshold': 'svcB'}


# ---------------- 응답 끝 2단 검증 (SPEC §5.3 post_check, A14) ----------------
import asyncio                                                    # noqa: E402
import re as _re                                                  # noqa: E402
from types import SimpleNamespace                                 # noqa: E402


class FakeClaude:
    """anthropic.AsyncAnthropic 대역: messages.create(**params) → 응답 객체. fn(params, k)가 텍스트를 만든다."""
    def __init__(self, fn, delay=0.0):
        self.calls, self.fn, self.delay = [], fn, delay
        self.messages = self

    async def create(self, **p):
        k = len(self.calls); self.calls.append(p)
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=self.fn(p, k))], stop_reason='end_turn',
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def rocket_labels(p, k):
    # 번호 붙은 문장 중 'rockets'가 든 문장만 H
    lines = _re.findall(r'^\[(\d+)\] (.*)$', p['messages'][0]['content'], _re.M)
    return json.dumps({'sentences': [{'i': int(i), 'label': 'H' if 'rockets' in t else 'S',
                                      **({'phrase': 'flew rockets', 'reason': 'not in reference'} if 'rockets' in t else {})}
                                     for i, t in lines]})


def _pc_client(tmp_path, checker=None, **kw):
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path=str(tmp_path / 'm.jsonl'), stride_tokenizer='words',
                 collection_db='', anthropic_api_key='', **kw)
    app = create_app(s, judge=LexicalJudge(threshold=0.5), http_client=up, post_checker=checker)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gw'), s


async def run_chunks(client, b):
    r = await client.post('/v1/chat/completions', json=b)
    assert r.status_code == 200, r.text
    out = []
    for line in r.text.splitlines():
        if line.startswith('data:'):
            out.append('DONE' if line.endswith('[DONE]') else json.loads(line[5:]))
    return r, out


def _pc_body(post_check=True, mode='hold', action='flag', stream=True):
    b = body(BAD_2ND, mode, action, stream)
    if post_check is not None:
        b['verification']['post_check'] = post_check
    return b


def _post(out):
    return [c['verification'] for c in out if c != 'DONE' and (c.get('verification') or {}).get('kind') == 'post_check']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pass', 'hold'])
async def test_post_check_stream(tmp_path, mode):
    from hallugate.postcheck import ClaudePostChecker
    fc = FakeClaude(rocket_labels)
    client, s = _pc_client(tmp_path, ClaudePostChecker(client=fc, model='claude-sonnet-5-5', votes=3, threshold=0.5))
    r, out = await run_chunks(client, _pc_body(mode=mode))
    assert r.headers['x-verification-post-check'] == 'on'
    text = ''.join(ch['delta'].get('content') or '' for c in out if c != 'DONE' for ch in c['choices'])
    assert text == BAD_2ND                                            # 본문 보존
    # 순서: ... 종료 청크 → post_check 청크 → [DONE]
    i_fin = next(i for i, c in enumerate(out) if c != 'DONE' and c['choices'] and c['choices'][0].get('finish_reason'))
    assert (out[i_fin + 1].get('verification') or {}).get('kind') == 'post_check' and out[i_fin + 2] == 'DONE'
    pc = _post(out)
    assert len(pc) == 1
    pc = pc[0]
    assert pc['verified'] and pc['flagged'] and pc['score'] == 1.0 and pc['votes'] == pc['valid_votes'] == 3
    assert pc['model'] == 'claude-sonnet-5-5' and pc['threshold'] == 0.5 and pc['check_ms'] >= 0
    flagged = [x for x in pc['sentences'] if x['flagged']]
    assert [x['sentence_idx'] for x in flagged] == [1] and flagged[0]['h_votes'] == 3 and flagged[0]['phrase'] == 'flew rockets'
    assert len(fc.calls) == 3 and all(c['model'] == 'claude-sonnet-5-5' for c in fc.calls)
    assert REF in fc.calls[0]['messages'][0]['content'] and '[2] Its chief executive flew rockets' in fc.calls[0]['messages'][0]['content']
    rec = json.loads(open(s.metrics_path).read().splitlines()[-1])
    assert rec['post_check']['flagged'] is True


@pytest.mark.asyncio
async def test_post_check_off_by_default(tmp_path):
    from hallugate.postcheck import ClaudePostChecker
    fc = FakeClaude(rocket_labels)
    client, _ = _pc_client(tmp_path, ClaudePostChecker(client=fc))
    r, out = await run_chunks(client, _pc_body(post_check=None))
    assert _post(out) == [] and fc.calls == [] and r.headers['x-verification-post-check'] == 'off'
    # 서버 기본값을 켜면 요청에 없어도 한다
    client, _ = _pc_client(tmp_path, ClaudePostChecker(client=fc), post_check_default=True)
    _, out = await run_chunks(client, _pc_body(post_check=None))
    assert len(_post(out)) == 1 and len(fc.calls) == 3


@pytest.mark.asyncio
async def test_post_check_votes_and_parse_failure(tmp_path):
    from hallugate.postcheck import ClaudePostChecker
    # 판정 3번: H, 파싱 실패, 모두 S → 응답 점수 1/3 < 0.5 → 표시 안 함. 문장은 유효 판정 2개 중 1개 H = 0.5 → 표시
    def fn(p, k):
        if k == 1:
            return 'not json'
        return rocket_labels(p, k) if k == 0 else rocket_labels(p, k).replace('"H"', '"S"')
    client, _ = _pc_client(tmp_path, ClaudePostChecker(client=FakeClaude(fn), votes=3, threshold=0.5))
    _, out = await run_chunks(client, _pc_body())
    pc = _post(out)[0]
    assert pc['verified'] and pc['valid_votes'] == 2 and abs(pc['score'] - 1 / 3) < 1e-9 and not pc['flagged']
    assert [x['sentence_idx'] for x in pc['sentences'] if x['flagged']] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['error', 'all_bad', 'timeout', 'unavailable', 'cut'])
async def test_post_check_fail_open(tmp_path, case):
    from hallugate.postcheck import ClaudePostChecker
    def boom(p, k):
        raise RuntimeError('api down')
    checker = {'error': ClaudePostChecker(client=FakeClaude(boom)),
               'all_bad': ClaudePostChecker(client=FakeClaude(lambda p, k: 'nope')),
               'timeout': ClaudePostChecker(client=FakeClaude(rocket_labels, delay=0.5)),
               'unavailable': None, 'cut': ClaudePostChecker(client=FakeClaude(rocket_labels))}[case]
    client, _ = _pc_client(tmp_path, checker, post_check_timeout_ms=100.0)
    _, out = await run_chunks(client, _pc_body(action='cut' if case == 'cut' else 'flag'))
    text = ''.join(ch['delta'].get('content') or '' for c in out if c != 'DONE' for ch in c['choices'])
    assert text == ('The company was founded in 1998 in Seoul. ' if case == 'cut' else BAD_2ND)
    pc = _post(out)
    want = {'error': 'error', 'all_bad': 'error', 'timeout': 'timeout', 'unavailable': 'unavailable', 'cut': 'cut'}[case]
    assert len(pc) == 1 and pc[0]['verified'] is False and pc[0]['reason'] == want, pc
    assert out[-1] == 'DONE'


@pytest.mark.asyncio
async def test_post_check_non_stream_and_validation(tmp_path):
    from hallugate.postcheck import ClaudePostChecker
    client, _ = _pc_client(tmp_path, ClaudePostChecker(client=FakeClaude(rocket_labels)))
    r = await client.post('/v1/chat/completions', json=_pc_body(stream=False))
    d = r.json()
    assert d['choices'][0]['message']['content'] == BAD_2ND
    assert d['verification']['post_check']['flagged'] is True
    bad = _pc_body(); bad['verification']['post_check'] = 'yes'
    assert (await client.post('/v1/chat/completions', json=bad)).status_code == 400
    h = (await client.get('/health')).json()['post_check']
    assert h == {'available': True, 'default': False, 'model': 'claude-sonnet-5-5', 'votes': 3, 'threshold': 0.5}


def test_post_check_prompt_matches_j2():
    # 2단 검증 지시문 = J2 노트북 sonnet_H 후보 지시문 (J2 결과를 그대로 옮겨 쓰기 위해)
    from hallugate.postcheck import SYSTEM
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'experiments', 'builders', 'build_j2.py')
    if not os.path.exists(path):
        pytest.skip('실험 builder가 없는 환경(S3 노트북)')
    src = open(path, encoding='utf-8').read()
    i = src.index("SYS_SENT = '''") + len("SYS_SENT = '''")
    assert src[i:src.index("'''", i)] == SYSTEM


# ---------------- 데모 화면 (SPEC §2.8) ----------------
@pytest.mark.asyncio
async def test_demo_page(tmp_path):
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path='', collection_db='')
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(s, judge=LexicalJudge(), http_client=up)),
                               base_url='http://gw')
    r = await client.get('/demo')
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/html')
    assert '/v1/chat/completions' in r.text and 'post_check' in r.text and 'mock_response' in r.text
    # 근거를 LLM에도 보내기: <context> 태그 + 시스템 지시, 운영 모드 전용 예제 (SPEC §2.8)
    assert 'id="ragmsg"' in r.text and '<context>' in r.text and 'LIVE_PRESETS' in r.text
    assert 'http://' not in r.text.replace('http://www.w3.org', '') and 'https://' not in r.text   # 외부 리소스 없음
    s_off = Settings(upstream_base_url='http://mock/v1', metrics_path='', collection_db='', demo=False)
    off = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(s_off, judge=LexicalJudge(), http_client=up)),
                            base_url='http://gw')
    assert (await off.get('/demo')).status_code == 404


# ---------------- 운영 모드: 인증·사용자 분리 (SPEC §10.1·10.2, H1·H2) ----------------
def test_keystore_create_verify_revoke(tmp_path):
    import sqlite3
    from hallugate.keys import KeyStore
    ks = KeyStore(str(tmp_path / 'k.sqlite3'))
    kid, secret = ks.create('alice')
    assert secret.startswith(f'hg_{kid}_') and len(secret) > 30
    t = ks.verify(secret)
    assert t is not None and t.key_id == kid and t.name == 'alice'
    assert ks.verify(None) is None and ks.verify('') is None and ks.verify('hg_nope_x') is None
    assert ks.verify(secret[:-1] + ('a' if secret[-1] != 'a' else 'b')) is None      # 비밀 한 글자만 달라도 거부
    assert ks.verify(f'hg_{kid}') is None and ks.verify('Bearer ' + secret) is None
    rows = ks.list()
    assert [r['key_id'] for r in rows] == [kid] and 'hash' not in rows[0] and rows[0]['revoked'] is None
    dump = '\n'.join(sqlite3.connect(str(tmp_path / 'k.sqlite3')).iterdump())
    assert secret.split('_', 2)[2] not in dump                           # 비밀 원문은 저장하지 않음 (해시만)
    assert ks.revoke(kid) and not ks.revoke('nope')
    assert ks.verify(secret) is None and ks.list()[0]['revoked'] is not None


def _hosted_gw(tmp_path, **kw):
    from hallugate.keys import KeyStore
    sent = []

    async def hook(req):
        sent.append({**json.loads(req.content), '_host': req.url.host, '_auth': req.headers.get('authorization')})
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock',
                           event_hooks={'request': [hook]})
    limiter, checker = kw.pop('limiter', None), kw.pop('post_checker', None)
    kw.setdefault('upstream_allowlist', 'openai=http://up-openai/v1,anthropic=http://up-anth/v1')
    s = Settings(upstream_base_url='http://mock/v1', metrics_path=str(tmp_path / 'm.jsonl'), stride_tokenizer='words',
                 collection_top_k=2, hosted=True, keys_db=str(tmp_path / 'k.sqlite3'), admin_token='adm', **kw)
    ks, st = KeyStore(s.keys_db), _store(tmp_path)
    app = create_app(s, judge=CapJudge(threshold=0.5), http_client=up, store=st, keys=ks, limiter=limiter,
                     post_checker=checker)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gw'), s, st, ks, sent


def _K(secret):
    return {'x-hallugate-key': secret}


@pytest.mark.asyncio
async def test_hosted_auth_required(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path)
    kid, key = ks.create('alice')
    b = body(GOOD, 'hold')
    for h in [{}, _K('hg_bad_bad'), {'Authorization': f'Bearer {key}'}, {'Authorization': 'Bearer adm'}]:
        r = await client.post('/v1/chat/completions', json=b, headers=h)
        assert r.status_code == 401 and r.json()['error']['type'] == 'authentication_error', h
        assert (await client.get('/v1/collections', headers=h)).status_code == 401
        assert (await client.post('/v1/collections/kb/documents', json={'text': DOC_A}, headers=h)).status_code == 401
    assert sent == []                                                  # 인증 실패는 업스트림을 부르지 않음
    text, ev, _ = await run(client.__class__(transport=client._transport, base_url='http://gw', headers=_K(key)), b)
    assert text == GOOD
    ks.revoke(kid)
    assert (await client.post('/v1/chat/completions', json=b, headers=_K(key))).status_code == 401
    h = (await client.get('/health')).json()                           # 운영 모드 health는 내부 값 숨김
    assert h['ok'] and h['hosted'] and 'threshold' not in h and 'judge' not in h


@pytest.mark.asyncio
async def test_hosted_tenant_isolation(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path)
    _, ka = ks.create('alice'); _, kb = ks.create('bob')
    A, B = _K(ka), _K(kb)
    url = '/v1/collections/kb/documents'
    assert (await client.post(url, json={'text': DOC_A, 'doc_id': 'a'}, headers=A)).status_code == 200
    # B에게는 A의 컬렉션이 없는 것과 같다
    assert (await client.get('/v1/collections', headers=B)).json()['data'] == []
    assert (await client.get(url, headers=B)).status_code == 404
    assert (await client.delete(url + '/a', headers=B)).status_code == 404
    rb = rag_body(GOOD)
    assert (await client.post('/v1/chat/completions', json=rb, headers=B)).status_code == 404
    rb['verification']['collection'] = f'{ks.list()[0]["key_id"]}/kb'                # 내부 이름을 흉내 내도 못 봄
    assert (await client.post('/v1/chat/completions', json=rb, headers=B)).status_code == 404
    # B가 같은 이름으로 만들면 별개 컬렉션
    assert (await client.post(url, json={'text': DOC_B, 'doc_id': 'b'}, headers=B)).status_code == 200
    assert (await client.get('/v1/collections', headers=A)).json()['data'] == [{'collection': 'kb', 'n_docs': 1}]
    assert [d['doc_id'] for d in (await client.get(url, headers=A)).json()['data']] == ['a']
    assert [d['doc_id'] for d in (await client.get(url, headers=B)).json()['data']] == ['b']
    # 채팅 근거도 자기 컬렉션에서만
    cA = client.__class__(transport=client._transport, base_url='http://gw', headers=A)
    _, ev, _ = await run(cA, rag_body(GOOD))
    evi = next(e for k, e in ev if k == 'v' and e.get('kind') == 'evidence')
    assert evi['collection'] == 'kb' and {x['doc_id'] for x in evi['evidence']} == {'a'}
    assert 'penguin' not in sent[-1]['messages'][0]['content'].lower()
    # A가 지워도 B의 문서는 그대로
    assert (await client.delete(url + '/a', headers=A)).json()['deleted'] is True
    assert [d['doc_id'] for d in (await client.get(url, headers=B)).json()['data']] == ['b']


@pytest.mark.asyncio
async def test_hosted_document_limits(tmp_path):
    client, s, st, ks, _ = _hosted_gw(tmp_path, max_docs=2, max_doc_chars=50)
    _, k = ks.create('alice')
    url = '/v1/collections/kb/documents'
    assert (await client.post(url, json={'text': 'x' * 51}, headers=_K(k))).status_code == 413
    assert (await client.post(url, json={'documents': [{'text': 'one doc'}, {'text': 'two doc'}, {'text': 'three'}]},
                              headers=_K(k))).status_code == 413
    assert (await client.post(url, json={'documents': [{'text': 'one doc', 'doc_id': '1'}, {'text': 'two', 'doc_id': '2'}]},
                              headers=_K(k))).status_code == 200
    assert (await client.post('/v1/collections/other/documents', json={'text': 'three'}, headers=_K(k))).status_code == 413
    assert (await client.post(url, json={'text': 'replaced', 'doc_id': '1'}, headers=_K(k))).status_code == 200   # 교체는 개수 안 늘림


def test_keys_cli(tmp_path, capsys):
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location('keys_cli', os.path.join(root, 'scripts', 'keys.py'))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    db, cdb = str(tmp_path / 'k.sqlite3'), str(tmp_path / 'c.sqlite3')
    cli.main(['--db', db, 'create', '--name', 'alice'])
    out = capsys.readouterr().out
    key = next(w for w in out.split() if w.startswith('hg_'))
    from hallugate.keys import KeyStore
    kid = KeyStore(db).verify(key).key_id
    st = _store(tmp_path); st.add(f'{kid}/kb', DOC_A); st.add('other/kb', DOC_B); st.close()
    cli.main(['--db', db, 'list']); assert kid in capsys.readouterr().out
    cli.main(['--db', db, '--collections', cdb, 'revoke', kid, '--purge'])
    assert KeyStore(db).verify(key) is None
    from hallugate.store import SQLiteStore
    assert [c['collection'] for c in SQLiteStore(cdb, None).collections()] == ['other/kb']     # 그 사용자 것만 지움
    from hallugate.metrics import MetricsStore
    MetricsStore(db).add({'key_id': kid, 'status': 200}); MetricsStore(db).add({'key_id': 'zz', 'status': 200})
    cli.main(['--db', db, '--collections', cdb, 'revoke', kid, '--purge'])
    assert [r['key_id'] for r in MetricsStore(db).rows(0)] == ['zz']                         # 기록도 그 사용자 것만 지움


# ---------------- 운영 모드: 업스트림 허용 목록·사용량 제한 (SPEC §10.3·10.4, H3·H4) ----------------
def _hc(client, key, **h):
    return client.__class__(transport=client._transport, base_url='http://gw', headers={**_K(key), **h})


@pytest.mark.asyncio
async def test_hosted_upstream_allowlist(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path, upstream_api_key='SERVER-KEY')
    _, k = ks.create('alice')
    c = _hc(client, k, Authorization='Bearer user-llm-key')
    r = await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))
    assert r.status_code == 200 and r.headers['x-upstream'] == 'openai'
    assert sent[-1]['_host'] == 'up-openai' and sent[-1]['_auth'] == 'Bearer user-llm-key'   # 운영자 키 아님
    r = await _hc(client, k, Authorization='Bearer a', **{'x-upstream': 'anthropic'}).post('/v1/chat/completions', json=body(GOOD, 'hold'))
    assert r.status_code == 200 and r.headers['x-upstream'] == 'anthropic' and sent[-1]['_host'] == 'up-anth'
    n = len(sent)
    for bad in ['http://evil.example/v1', 'OPENAI ', 'mock']:
        r = await _hc(client, k, **{'x-upstream': bad}).post('/v1/chat/completions', json=body(GOOD, 'hold'))
        assert r.status_code == 400, bad
    assert len(sent) == n                                              # 임의 주소는 부르지 않음
    await _hc(client, k).post('/v1/chat/completions', json=body(GOOD, 'hold'))
    assert sent[-1]['_auth'] is None                                    # 사용자 키가 없어도 운영자 키를 넣지 않음


def test_limiter_window_and_concurrency():
    from hallugate.limits import Limiter
    now = [1000.0]
    L = Limiter(now=lambda: now[0])
    assert L.hit('a', 2) is None and L.hit('a', 2) is None
    ra = L.hit('a', 2)
    assert ra is not None and 59 <= ra <= 60 and L.hit('b', 2) is None     # 다른 키는 따로
    now[0] += 30; assert L.hit('a', 2) is not None                      # 거절은 세지 않음, 창 안이면 계속 거절
    now[0] += 31; assert L.hit('a', 2) is None                          # 60초 지나면 다시 허용
    assert L.acquire('a', 1) and not L.acquire('a', 1) and L.acquire('b', 1)
    L.release('a'); assert L.acquire('a', 1)


@pytest.mark.asyncio
async def test_hosted_rate_limit(tmp_path):
    from hallugate.limits import Limiter
    now = [0.0]
    client, s, st, ks, sent = _hosted_gw(tmp_path, rate_rpm=2, limiter=Limiter(now=lambda: now[0]))
    _, ka = ks.create('alice'); _, kb = ks.create('bob'); _, kc = ks.create('carol', limits={'rpm': 1})
    a = _hc(client, ka)
    assert [(await a.post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code for _ in range(2)] == [200, 200]
    r = await a.post('/v1/chat/completions', json=body(GOOD, 'hold'))
    assert r.status_code == 429 and r.json()['error']['type'] == 'rate_limit_error' and int(r.headers['retry-after']) >= 1
    assert len(sent) == 2                                              # 거절된 요청은 업스트림을 부르지 않음
    assert (await _hc(client, kb).post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code == 200
    c = _hc(client, kc)                                                # 키별 한도가 기본값보다 우선
    assert [(await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code for _ in range(2)] == [200, 429]
    now[0] += 61
    assert (await a.post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code == 200


@pytest.mark.asyncio
async def test_hosted_concurrency_and_size_limits(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path, max_concurrent=1, max_ref_chars=200)
    _, k = ks.create('alice')
    c = _hc(client, k)
    slow = body(GOOD, 'hold'); slow['mock_token_ms'] = 20
    r1, r2 = await asyncio.gather(c.post('/v1/chat/completions', json=slow), c.post('/v1/chat/completions', json=slow))
    assert sorted([r1.status_code, r2.status_code]) == [200, 429]
    assert (await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code == 200      # 끝나면 다시 허용
    assert (await c.post('/v1/chat/completions', json=rag_body(GOOD, coll='missing'))).status_code == 404
    assert (await c.post('/v1/chat/completions', json=body(GOOD, 'hold', stream=False))).status_code == 200   # 오류 뒤에도 풀림
    big = body(GOOD, 'hold'); big['verification']['context'] = 'word ' * 50
    n = len(sent)
    r = await c.post('/v1/chat/completions', json=big)
    assert r.status_code == 413 and len(sent) == n


@pytest.mark.asyncio
async def test_hosted_post_check_daily_quota(tmp_path):
    from hallugate.keys import KeyStore
    from hallugate.postcheck import ClaudePostChecker
    fc = FakeClaude(rocket_labels)
    client, s, st, ks, sent = _hosted_gw(tmp_path, post_check_daily=1, post_checker=ClaudePostChecker(client=fc, votes=3))
    _, k = ks.create('alice')
    c = _hc(client, k)
    _, out = await run_chunks(c, _pc_body())
    assert _post(out)[0]['verified'] is True and len(fc.calls) == 3
    _, out = await run_chunks(c, _pc_body())
    text = ''.join(ch['delta'].get('content') or '' for x in out if x != 'DONE' for ch in x['choices'])
    assert text == BAD_2ND and _post(out)[0] == {'kind': 'post_check', 'verified': False, 'reason': 'quota'}
    assert len(fc.calls) == 3                                          # 한도를 넘으면 Claude를 부르지 않음
    kid = ks.list()[0]['key_id']
    assert KeyStore(s.keys_db).post_checks_today(kid) == 1             # 컨테이너가 다시 떠도 유지


# ---------------- 운영 모드: 원문 없는 기록·관리자 통계 (SPEC §10.5·10.6, H5·H7) ----------------
def _rows(path):
    import sqlite3
    return [json.loads(r[0]) for r in sqlite3.connect(path).execute('SELECT data FROM requests ORDER BY ts')]


@pytest.mark.asyncio
async def test_hosted_records_have_no_raw_text(tmp_path, capsys):
    from hallugate.postcheck import ClaudePostChecker
    client, s, st, ks, sent = _hosted_gw(tmp_path, post_checker=ClaudePostChecker(client=FakeClaude(rocket_labels)))
    _, k = ks.create('alice')
    c = _hc(client, k, Authorization='Bearer user-llm-key-123')
    QUESTION = 'Which secret question zebra?'
    b = _pc_body(); b['messages'][0]['content'] = QUESTION
    _, out = await run_chunks(c, b)                                        # 스트리밍 + 2단
    assert _post(out)[0]['flagged']
    await c.post('/v1/chat/completions', json=body(BAD_2ND, 'hold', stream=False))          # 비스트리밍
    await c.post('/v1/collections/kb/documents', json={'text': DOC_A, 'doc_id': 'a'})
    await run(c, rag_body(BAD_2ND))                                        # 컬렉션
    await client.post('/v1/chat/completions', json=b)                      # 401 (키 없음)
    off = body(GOOD, 'off'); await c.post('/v1/chat/completions', json=off)   # 검증 끔 → 그대로 통과도 기록
    rows = _rows(s.keys_db)
    assert [r['status'] for r in rows] == [200, 200, 200, 401, 200]
    assert rows[0]['stream'] and rows[0]['post_check']['flagged'] and rows[0]['n_flagged'] == 1 and rows[0]['n_sentences'] == 3
    assert rows[2]['collection_used'] and rows[2]['n_evidence'] >= 1 and rows[3]['key_id'] == ''
    dump = json.dumps(rows) + capsys.readouterr().out
    for secret in ['rockets', 'founded in 1998', 'zebra', 'Penguins', 'user-llm-key-123', k, k.split('_', 2)[2],
                   'not in reference', 'flew rockets', 'Be brief']:
        assert secret not in dump, secret                                 # 원문·키가 기록에 없음
    assert dump.count('"key_id"') >= 2 and not os.path.exists(s.metrics_path)   # 운영 모드는 JSONL을 쓰지 않음


def test_metrics_store_retention_and_purge(tmp_path):
    import time as _t
    from hallugate.metrics import MetricsStore
    m = MetricsStore(str(tmp_path / 'k.sqlite3'))
    m.add({'key_id': 'a', 'status': 200}, ts=_t.time() - 40 * 86400)
    m.add({'key_id': 'a', 'status': 200}); m.add({'key_id': 'b', 'status': 200})
    assert m.prune(30) == 1 and len(m.rows(0)) == 2
    assert m.purge('a') == 1 and [r['key_id'] for r in m.rows(0)] == ['b']


@pytest.mark.asyncio
async def test_admin_stats(tmp_path):
    from hallugate.postcheck import ClaudePostChecker
    client, s, st, ks, sent = _hosted_gw(tmp_path, rate_rpm=3, post_checker=ClaudePostChecker(client=FakeClaude(rocket_labels)))
    _, ka = ks.create('alice'); kid = ks.list()[0]['key_id']
    a = _hc(client, ka)
    await run_chunks(a, _pc_body())
    await a.post('/v1/chat/completions', json=body(GOOD, 'hold'))
    await a.post('/v1/chat/completions', json=body(GOOD, 'hold'))
    assert (await a.post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code == 429
    await client.post('/v1/chat/completions', json=body(GOOD, 'hold'))     # 401
    assert (await client.get('/admin/stats')).status_code == 401
    assert (await client.get('/admin/stats', headers={'Authorization': 'Bearer nope'})).status_code == 401
    assert (await client.get('/admin/stats?window=abc', headers={'Authorization': 'Bearer adm'})).status_code == 400
    r = await client.get('/admin/stats?window=24h', headers={'Authorization': 'Bearer adm'})
    st_ = r.json()
    assert st_['window'] == '24h' and st_['requests']['total'] == 5
    assert st_['requests']['by_status'] == {'200': 3, '401': 1, '429': 1} and st_['rate_limited'] == 1
    assert st_['requests']['by_key'][kid] == 4 and st_['requests']['by_upstream']['openai'] == 4
    for k_ in ['ttft_ms', 'total_ms']:
        assert st_['latency'][k_]['p50'] is not None and st_['latency'][k_]['p95'] >= st_['latency'][k_]['p50']
    assert st_['latency']['judge_ms_p95'] is not None
    assert st_['unverified']['rate'] == 0.0 and st_['post_check']['runs'] == 1 and st_['post_check']['flagged'] == 1
    assert st_['post_check']['est_cost_usd'] > 0 and st_['cold_starts'] == []
    rows = _rows(s.keys_db)
    assert rows[0]['cold'] is True and all(not x.get('cold') for x in rows[1:])    # 컨테이너 시작 직후 첫 요청만 콜드
    m = client._transport.app if hasattr(client._transport, 'app') else None
    from hallugate.metrics import MetricsStore
    MetricsStore(s.keys_db).add_start(1234.5)
    r = await client.get('/admin/stats?window=30m', headers={'Authorization': 'Bearer adm'})
    cs = r.json()['cold_starts']
    assert len(cs) == 1 and cs[0]['warmup_ms'] == 1234.5


# ---------------- 운영 모드: 관리자 키 API (SPEC §10.1) ----------------
@pytest.mark.asyncio
async def test_admin_keys_api(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path)
    A = {'Authorization': 'Bearer adm'}
    assert (await client.post('/admin/keys', json={'name': 'x'})).status_code == 401
    assert (await client.post('/admin/keys', json={'name': 'x'}, headers={'Authorization': 'Bearer no'})).status_code == 401
    r = (await client.post('/admin/keys', json={'name': 'alice', 'limits': {'rpm': 5}}, headers=A)).json()
    kid, key = r['key_id'], r['key']
    assert key.startswith(f'hg_{kid}_') and ks.verify(key).limits == {'rpm': 5}
    lst = (await client.get('/admin/keys', headers=A)).json()['data']
    assert [x['key_id'] for x in lst] == [kid] and 'hash' not in lst[0] and key not in json.dumps(lst)
    assert (await client.post('/admin/keys', json={'limits': 'bad'}, headers=A)).status_code == 400
    # 발급한 키로 바로 사용 → 폐기(+purge) → 401, 문서·기록 삭제
    c = _hc(client, key)
    await c.post('/v1/collections/kb/documents', json={'text': DOC_A})
    assert (await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code == 200
    r = (await client.post(f'/admin/keys/{kid}/revoke', json={'purge': True}, headers=A)).json()
    assert r['revoked'] and r['purged_docs'] == 1 and r['purged_records'] >= 1
    assert (await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))).status_code == 401
    assert (await client.post('/admin/keys/nope/revoke', json={}, headers=A)).json()['revoked'] is False


def test_keys_cli_remote(tmp_path, capsys, monkeypatch):
    # scripts/keys.py --url: 관리자 API 호출 (ASGI 앱에 바로 연결)
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location('keys_cli2', os.path.join(root, 'scripts', 'keys.py'))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    client, s, st, ks, sent = _hosted_gw(tmp_path)
    monkeypatch.setenv('ADMIN_TOKEN', 'adm')

    class Bridge(httpx.BaseTransport):                                  # 동기 httpx → 비동기 ASGI 앱
        def handle_request(self, request):
            async def go():
                r = await client._transport.handle_async_request(request)
                body_ = b''.join([c async for c in r.stream])
                return httpx.Response(r.status_code, headers=r.headers, content=body_)
            return asyncio.new_event_loop().run_until_complete(go())
    monkeypatch.setattr(cli, 'TRANSPORT', Bridge())
    cli.main(['--url', 'http://gw', 'create', '--name', 'bob', '--limits', '{"rpm": 3}'])
    key = next(w for w in capsys.readouterr().out.split() if w.startswith('hg_'))
    t = ks.verify(key); assert t.name == 'bob' and t.limits == {'rpm': 3}
    cli.main(['--url', 'http://gw', 'list']); assert t.key_id in capsys.readouterr().out
    cli.main(['--url', 'http://gw', 'revoke', t.key_id, '--purge']); assert ks.verify(key) is None


# ---------------- 운영 모드: 사용자 LLM 사용량·예산 (SPEC §10.10, H9) ----------------
def test_prices_lookup():
    from hallugate.prices import cost_usd, price_for
    assert price_for('gpt-4o-mini-2024-07-18') == price_for('gpt-4o-mini')          # 날짜 붙은 이름은 접두어로
    assert price_for('gpt-4o-2024-08-06') != price_for('gpt-4o-mini')               # 가장 긴 접두어
    assert price_for('unknown-model') is None and cost_usd('unknown-model', 10, 10) is None
    pin, pout = price_for('gpt-4o-mini')
    assert abs(cost_usd('gpt-4o-mini', 1_000_000, 2_000_000) - (pin + 2 * pout)) < 1e-9


@pytest.mark.asyncio
async def test_hosted_usage_collected_without_changing_output(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path)
    _, k = ks.create('alice')
    c = _hc(client, k)
    b = body(GOOD, 'hold'); b['model'] = 'gpt-4o-mini'
    _, out = await run_chunks(c, b)                                     # 사용자는 usage를 요청하지 않음
    assert sent[-1]['stream_options'] == {'include_usage': True}       # 업스트림에는 요청
    assert not any(x != 'DONE' and x.get('usage') for x in out)        # 사용자에게는 보내지 않음
    b2 = dict(b, stream_options={'include_usage': True})
    _, out2 = await run_chunks(c, b2)
    assert any(x != 'DONE' and x.get('usage') for x in out2)           # 요청하면 그대로 보냄
    await c.post('/v1/chat/completions', json=body(GOOD, 'hold', stream=False) | {'model': 'gpt-4o-mini'})
    r, out3 = await run_chunks(c, body(GOOD, 'off') | {'model': 'gpt-4o-mini'})      # 검증 끔
    assert not any(x != 'DONE' and x.get('usage') for x in out3)
    rows = _rows(s.keys_db)
    assert [(x['tokens_in'], x['tokens_out']) for x in rows] == [(3, 7)] * 4
    assert all(x['model'] == 'gpt-4o-mini' and x['llm_cost_usd'] > 0 for x in rows)
    u = (await c.get('/v1/usage?window=30d')).json()
    assert u['requests'] == 4 and u['tokens_in'] == 12 and u['tokens_out'] == 28
    assert u['by_model']['gpt-4o-mini']['requests'] == 4 and u['est_cost_usd'] > 0
    assert (await client.get('/v1/usage')).status_code == 401
    st_ = (await client.get('/admin/stats', headers={'Authorization': 'Bearer adm'})).json()
    assert st_['llm']['tokens_in'] == 12 and st_['llm']['by_model']['gpt-4o-mini']['tokens_out'] == 28


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['warn', 'block'])
async def test_hosted_budget(tmp_path, action):
    client, s, st, ks, sent = _hosted_gw(tmp_path)
    _, k = ks.create('alice', limits={'monthly_tokens': 15, 'budget_action': action})
    c = _hc(client, k)
    r1 = await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))           # 요청마다 10토큰
    assert r1.headers['x-budget-used'] == '0.000' and 'x-budget-warning' not in r1.headers
    r2 = await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))
    assert r2.headers['x-budget-used'] == '0.667' and 'x-budget-warning' not in r2.headers
    n = len(sent)
    r3 = await c.post('/v1/chat/completions', json=body(GOOD, 'hold'))
    if action == 'warn':
        assert r3.status_code == 200 and r3.headers['x-budget-warning'] == 'exceeded' and len(sent) == n + 1
    else:
        assert r3.status_code == 429 and r3.json()['error']['type'] == 'insufficient_quota' and len(sent) == n
    u = (await c.get('/v1/usage')).json()['budget']
    assert u['monthly_tokens'] == 15 and u['tokens_used'] == (30 if action == 'warn' else 20) and u['action'] == action
    # 예산 없는 키는 헤더만 없음
    _, k2 = ks.create('bob')
    r = await _hc(client, k2).post('/v1/chat/completions', json=body(GOOD, 'hold'))
    assert r.status_code == 200 and 'x-budget-used' not in r.headers


@pytest.mark.asyncio
async def test_admin_key_limits_accept_budget(tmp_path):
    client, *_ = _hosted_gw(tmp_path)
    A = {'Authorization': 'Bearer adm'}
    ok = await client.post('/admin/keys', json={'limits': {'monthly_usd': 5.0, 'budget_action': 'block', 'rpm': 10}}, headers=A)
    assert ok.status_code == 200
    bad = await client.post('/admin/keys', json={'limits': {'budget_action': 'nope'}}, headers=A)
    assert bad.status_code == 400


# ---------------- 근거 다이어트 (SPEC §2.9, A15) ----------------
FILLER = [f'Unrelated note {i}: the weather station recorded rain and wind on day {i} near the old harbor bridge.'
          for i in range(12)]
LONG_REF = ' '.join(FILLER[:6]) + ' ' + DOC_A + ' ' + ' '.join(FILLER[6:])


def test_trimmer_selects_relevant_in_order():
    from hallugate.trim import Trimmer
    t = Trimmer(HashEmb(), chunk_words=12, overlap=2)
    kept, total = t.select(LONG_REF, 'When was the company founded and what was its revenue?', 2)
    assert total > 2 and len(kept) == 2
    assert any('founded in 1998' in c for c in kept) or any('revenue' in c for c in kept)
    pos = [LONG_REF.find(c[:30]) for c in kept]
    assert pos == sorted(pos)                                          # 원문 순서
    kept2, total2 = t.select('short text only', 'q', 4)
    assert kept2 == ['short text only'] and total2 == 1


def _trim_body(resp, src='field', trim=True, stream=True):
    q = 'When was the company founded and what was its revenue?'
    if src == 'field':
        msgs = [{'role': 'user', 'content': q}]
        ver = {'context': LONG_REF, 'mode': 'hold', 'trim': trim}
    else:
        msgs = [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': f'<context>{LONG_REF}</context>\n{q}'}]
        ver = {'mode': 'hold', 'trim': trim}
    return {'model': 'mock', 'stream': stream, 'messages': msgs, 'verification': ver,
            'mock_response': resp, 'mock_ttft_ms': 0, 'mock_token_ms': 0}


@pytest.mark.asyncio
@pytest.mark.parametrize('src', ['field', 'tag'])
async def test_trim_sends_only_selected_chunks(tmp_path, src):
    j = CapJudge(threshold=0.5)
    client, s, st, sent = _rag_gw(tmp_path, judge=j, collection_chunk_words=12, collection_overlap=2)
    r = await client.post('/v1/chat/completions', json=_trim_body(GOOD, src, {'top_k': 2}))
    assert r.status_code == 200 and r.headers['x-verification-trim'].startswith('2/')
    chunks = [json.loads(l[5:]) for l in r.text.splitlines() if l.startswith('data:') and not l.endswith('[DONE]')]
    text = ''.join(ch['delta'].get('content') or '' for c in chunks for ch in c['choices'])
    assert text == GOOD                                                # 본문 보존
    info = next(c['verification'] for c in chunks if (c.get('verification') or {}).get('kind') == 'trim')
    assert info['chunks_kept'] == 2 and info['chunks_total'] > 2 and 0 < info['chars_after'] < info['chars_before']
    assert 0 < info['saved_ratio'] < 1 and info['top_k'] == 2
    msgs = sent[-1]['messages']
    sent_text = json.dumps(msgs)
    assert 'Unrelated note 0:' not in sent_text or 'Unrelated note 11:' not in sent_text   # 다 보내지 않음
    ref = j.refs[-1]
    assert len(ref) == info['chars_after']                             # 판정 근거 = 고른 조각
    if src == 'field':
        assert msgs[0]['role'] == 'system' and all(p_ in msgs[0]['content'] for p_ in ref.split('\n\n'))
        assert msgs[1:] == _trim_body(GOOD, src)['messages']
    else:
        assert msgs[0] == {'role': 'system', 'content': 'Be brief.'}
        assert msgs[1]['content'] == f'<context>{ref}</context>\nWhen was the company founded and what was its revenue?'


@pytest.mark.asyncio
async def test_trim_noop_cases_and_validation(tmp_path):
    j = CapJudge(threshold=0.5)
    client, s, st, sent = _rag_gw(tmp_path, judge=j, collection_chunk_words=12, collection_overlap=2)
    b = _trim_body(GOOD, 'tag', trim=False)                             # 끄면 그대로
    await client.post('/v1/chat/completions', json=b)
    assert sent[-1]['messages'] == b['messages'] and j.refs[-1] == LONG_REF
    b = _trim_body(GOOD, 'tag', {'top_k': 32})                          # 조각 수 이하면 그대로
    r = await client.post('/v1/chat/completions', json=b)
    assert sent[-1]['messages'] == b['messages'] and r.headers['x-verification-trim'].split('/')[0] == r.headers['x-verification-trim'].split('/')[1]
    nb = _trim_body(GOOD, 'field', stream=False)                       # 비스트리밍
    v = (await client.post('/v1/chat/completions', json=nb)).json()['verification']
    assert v['trim']['top_k'] == s.trim_top_k == 8 and Settings().trim_top_k == 8   # 기본 조각 수 = TRIM_TOP_K (T1c 채택값)
    assert v['trim']['chunks_kept'] == min(8, v['trim']['chunks_total'])
    st.add('kb', DOC_A, doc_id='a')                                     # 컬렉션 요청에는 적용 안 함
    rb = rag_body(GOOD); rb['verification']['trim'] = True
    r = await client.post('/v1/chat/completions', json=rb)
    assert r.status_code == 200 and 'x-verification-trim' not in r.headers
    for bad in ['yes', {'top_k': 0}, {'top_k': 99}, {'k': 2}]:
        b = _trim_body(GOOD); b['verification']['trim'] = bad
        assert (await client.post('/v1/chat/completions', json=b)).status_code == 400, bad


@pytest.mark.asyncio
async def test_hosted_trim_recorded_without_text(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path, collection_chunk_words=12, collection_overlap=2)
    _, k = ks.create('alice')
    await _hc(client, k).post('/v1/chat/completions', json=_trim_body(GOOD, 'field', {'top_k': 2}))
    row = _rows(s.keys_db)[-1]
    assert row['trim_chunks_kept'] == 2 and row['trim_chars_after'] < row['trim_chars_before'] == len(LONG_REF)
    assert 'harbor' not in json.dumps(row)


# ---------------- 쓸 때만 켜기 (SPEC §10.7) ----------------
class _FakeWarm:
    """scripts/warm.py의 Modal 연결을 대신한다: 최소 컨테이너 수와 Dict를 메모리에 둔다."""
    def __init__(self):
        self.min, self.state, self.health = 0, {}, []

    def set_min(self, n):
        self.min = n

    def get(self, k):
        return self.state.get(k)

    def put(self, k, v):
        self.state[k] = v

    def pop(self, k):
        self.state.pop(k, None)

    def ping(self, url):
        self.health.append(url)
        return True


def _warm_cli():
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location('warm_cli', os.path.join(root, 'scripts', 'warm.py'))
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    return cli


def test_warm_up_down_status(capsys):
    cli = _warm_cli(); b = _FakeWarm(); cli.BACKEND = b; cli.NOW = lambda: 1000.0
    cli.main(['up'])
    assert b.min == 1 and b.state['until'] == 1000.0 + 2 * 3600            # 기본 2시간
    cli.main(['up', '--hours', '0.5'])
    assert b.min == 1 and b.state['until'] == 1000.0 + 1800                 # 다시 켜면 끌 시각만 새로
    capsys.readouterr(); cli.main(['status']); assert '30' in capsys.readouterr().out
    cli.main(['down'])
    assert b.min == 0 and 'until' not in b.state
    for bad in ['0', '-1', '13']:                                           # 0 < h ≤ 12
        with pytest.raises(SystemExit):
            cli.main(['up', '--hours', bad])
    assert b.min == 0
    cli.main(['up', '--wait', '--url', 'https://gw.example'])
    assert b.health == ['https://gw.example']


def test_warm_auto_cooldown():
    cli = _warm_cli(); b = _FakeWarm()
    b.min, b.state = 1, {'until': 5000.0}
    assert cli.cooldown(b, now=4999.0) is False and b.min == 1              # 아직
    assert cli.cooldown(b, now=5001.0) is True and b.min == 0 and 'until' not in b.state
    assert cli.cooldown(b, now=9999.0) is False                              # 켠 적 없으면 그대로


# ---------------- 운영자·사용자 화면 (SPEC §10.11) ----------------
@pytest.mark.asyncio
async def test_verification_trim_totals(tmp_path):
    client, s, st, ks, sent = _hosted_gw(tmp_path, collection_chunk_words=12, collection_overlap=2)
    _, ka = ks.create('alice'); _, kb = ks.create('bob')
    a, b = _hc(client, ka), _hc(client, kb)
    await a.post('/v1/chat/completions', json=body(GOOD, 'hold'))                  # 2문장, 환각 0
    await a.post('/v1/chat/completions', json=body(BAD_2ND, 'hold'))               # 3문장, 환각 1
    await a.post('/v1/chat/completions', json=body(GOOD, 'off'))                   # 검증 끔 → 집계 제외
    await a.post('/v1/chat/completions', json=_trim_body(GOOD, 'field', {'top_k': 2}))
    await b.post('/v1/chat/completions', json=body(BAD_2ND, 'hold'))
    A = {'Authorization': 'Bearer adm'}
    v = (await client.get('/admin/stats?window=1h', headers=A)).json()['verification']
    assert v['responses'] == 4 and v['responses_flagged'] == 2 and v['fully_verified'] == 4
    assert v['sentences_flagged'] == 2 and v['sentences_judged'] >= 10
    assert v['flag_rate'] == pytest.approx(v['sentences_flagged'] / v['sentences_judged'])
    assert v['unverified_by_reason'] == {}
    tr = (await client.get('/admin/stats?window=1h', headers=A)).json()['trim']
    assert tr['requests'] == 1 and tr['chars_after'] < tr['chars_before'] and 0 < tr['saved_ratio'] < 1
    u = (await a.get('/v1/usage?window=1h')).json()                                # 사용자 본인 것만
    assert u['verification']['responses'] == 3 and u['verification']['responses_flagged'] == 1
    assert u['trim']['requests'] == 1
    ub = (await b.get('/v1/usage?window=1h')).json()
    assert ub['verification']['responses'] == 1 and ub['trim'] == {'requests': 0, 'chars_before': 0, 'chars_after': 0,
                                                                     'saved_ratio': None}


@pytest.mark.asyncio
@pytest.mark.parametrize('path,needs', [('/admin', ['/admin/stats', '/admin/keys', 'revoke']),
                                        ('/console', ['/v1/usage', 'x-hallugate-key']),
                                        ('/start', ['x-hallugate-key', 'verification', 'base_url'])])
async def test_hosted_pages(tmp_path, path, needs):
    client, s, st, ks, sent = _hosted_gw(tmp_path)
    r = await client.get(path)
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/html')
    for n in needs:
        assert n in r.text
    assert 'https://' not in r.text and 'http://' not in r.text.replace('http://www.w3.org', '')   # 외부 리소스 없음
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_upstream.app), base_url='http://mock')
    plain = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(
        Settings(upstream_base_url='http://mock/v1', metrics_path='', collection_db=''), judge=LexicalJudge(), http_client=up)),
        base_url='http://gw')
    assert (await plain.get(path)).status_code == 404                              # 운영 모드 전용
