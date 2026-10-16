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
    s = Settings(upstream_base_url='http://mock/v1', metrics_path=str(tmp_path / 'm.jsonl'))
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
    s = Settings(upstream_base_url=up + '/v1', metrics_path=str(tmp_path / 'm.jsonl'))
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
