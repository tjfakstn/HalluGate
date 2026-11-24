"""API 키 없이 게이트웨이를 시험하기 위한 가짜 LLM (OpenAI 호환 + Anthropic Messages 형식).

요청 body에 다음 필드를 넣으면 그대로 재생한다 (실제 API로는 보내지 말 것):
  mock_response : 흘려보낼 답변 텍스트 (없으면 마지막 user 메시지를 되돌려줌)
  mock_ttft_ms  : 첫 토큰까지 지연 (기본 300ms)
  mock_token_ms : 토큰 간 간격 (기본 25ms ≈ 초당 40토큰, 상용 API와 비슷한 속도)
  mock_tool_call: 함수 이름. 본문 뒤에 tool_calls 청크 2개를 보내고 finish_reason=tool_calls
  stream_options.include_usage=true 이면 마지막에 usage 청크(choices: [])를 보낸다 (OpenAI와 같은 형식)

Anthropic 형식 POST /v1/messages (SPEC §2.10 시험용): 위와 같은 mock_response·mock_ttft_ms·mock_token_ms에 더해
  mock_response가 리스트면 text 블록 여러 개, mock_thinking: 본문 앞 thinking 블록, mock_tool_use: 본문 뒤 tool_use 블록,
  mock_citation: 첫 text 블록에 citations_delta 하나. usage는 입력 5(+캐시 읽기 2), 출력 9

토큰 단위: 기본은 단어. 환경 변수 MOCK_TOKENIZER=tiktoken 이면 GPT 토큰(cl100k_base) 단위.

실행: python mock_upstream.py --port 9000
"""
import argparse
import asyncio
import json
import os
import re
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title='mock-llm')


_ENC = None
if os.environ.get('MOCK_TOKENIZER', 'words') == 'tiktoken':
    # 상용 API처럼 GPT 토큰 단위로 흘려보낸다 (E1·E2의 토큰 단위와 일치). 이어 붙이면 원문과 같다.
    import tiktoken
    _ENC = tiktoken.get_encoding('cl100k_base')


def _tokens(text):
    if _ENC is not None:
        ids = _ENC.encode(text)
        dec, starts = _ENC.decode_with_offsets(ids)
        if dec == text:
            ends = starts[1:] + [len(text)]
            return [text[a:b] for a, b in zip(starts, ends) if b > a]
    return re.findall(r'\S+\s*|\s+', text)


@app.post('/v1/chat/completions')
async def chat(request: Request):
    b = await request.json()
    text = b.get('mock_response')
    if text is None:
        users = [m for m in b.get('messages', []) if m.get('role') == 'user']
        text = users[-1]['content'] if users else 'Hello.'
    ttft, gap = b.get('mock_ttft_ms', 300) / 1000, b.get('mock_token_ms', 25) / 1000
    cid, created, model = 'chatcmpl-mock' + uuid.uuid4().hex[:12], int(time.time()), b.get('model', 'mock')

    if not b.get('stream'):
        await asyncio.sleep(ttft + gap * len(_tokens(text)))
        return JSONResponse({'id': cid, 'object': 'chat.completion', 'created': created, 'model': model,
                             'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                                          'finish_reason': 'stop'}],
                             'usage': {'prompt_tokens': 3, 'completion_tokens': 7, 'total_tokens': 10}})

    def c(delta, finish=None):
        return 'data: ' + json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                                      'model': model, 'choices': [{'index': 0, 'delta': delta,
                                                                   'finish_reason': finish}]}) + '\n\n'

    async def gen():
        yield c({'role': 'assistant', 'content': ''})
        await asyncio.sleep(ttft)
        for tok in _tokens(text):
            yield c({'content': tok})
            await asyncio.sleep(gap)
        tool = b.get('mock_tool_call')                    # 시험용: 본문 뒤에 도구 호출 청크
        if tool:
            yield c({'tool_calls': [{'index': 0, 'id': 'call_mock', 'type': 'function',
                                     'function': {'name': tool, 'arguments': ''}}]})
            yield c({'tool_calls': [{'index': 0, 'function': {'arguments': '{"q": 1}'}}]})
        yield c({}, 'tool_calls' if tool else 'stop')
        if (b.get('stream_options') or {}).get('include_usage'):
            yield 'data: ' + json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model,
                                         'choices': [], 'usage': {'prompt_tokens': 3, 'completion_tokens': 7,
                                                                  'total_tokens': 10}}) + '\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


MSG_USAGE = {'input_tokens': 5, 'output_tokens': 9, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 2}


@app.post('/v1/messages')
async def messages(request: Request):
    """Anthropic Messages API 흉내 (SPEC §2.10 시험용)."""
    b = await request.json()
    texts = b.get('mock_response')
    if texts is None:
        users = [m for m in b.get('messages', []) if m.get('role') == 'user']
        c = users[-1]['content'] if users else 'Hello.'
        texts = c if isinstance(c, str) else ''.join(x.get('text', '') for x in c if x.get('type') == 'text')
    texts = texts if isinstance(texts, list) else [texts]
    ttft, gap = b.get('mock_ttft_ms', 300) / 1000, b.get('mock_token_ms', 25) / 1000
    mid, model = 'msg_mock' + uuid.uuid4().hex[:12], b.get('model', 'mock')
    thinking, tool = b.get('mock_thinking'), b.get('mock_tool_use')
    blocks = ([{'type': 'thinking', 'thinking': thinking, 'signature': 'sig-mock'}] if thinking else []) + \
             [{'type': 'text', 'text': t} for t in texts] + \
             ([{'type': 'tool_use', 'id': 'toolu_mock', 'name': tool, 'input': {'q': 1}}] if tool else [])
    stop = 'tool_use' if tool else 'end_turn'

    if not b.get('stream'):
        await asyncio.sleep(ttft + gap * sum(len(_tokens(t)) for t in texts))
        return JSONResponse({'id': mid, 'type': 'message', 'role': 'assistant', 'model': model, 'content': blocks,
                             'stop_reason': stop, 'stop_sequence': None, 'usage': MSG_USAGE})

    def ev(name, data):
        return f'event: {name}\ndata: {json.dumps(data)}\n\n'

    async def gen():
        yield ev('message_start', {'type': 'message_start', 'message': {
            'id': mid, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None,
            'stop_sequence': None, 'usage': {**MSG_USAGE, 'output_tokens': 1}}})
        yield ev('ping', {'type': 'ping'})
        await asyncio.sleep(ttft)
        first_text = True
        for i, bl in enumerate(blocks):
            if bl['type'] == 'thinking':
                yield ev('content_block_start', {'type': 'content_block_start', 'index': i,
                                                 'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''}})
                yield ev('content_block_delta', {'type': 'content_block_delta', 'index': i,
                                                 'delta': {'type': 'thinking_delta', 'thinking': bl['thinking']}})
                yield ev('content_block_delta', {'type': 'content_block_delta', 'index': i,
                                                 'delta': {'type': 'signature_delta', 'signature': bl['signature']}})
            elif bl['type'] == 'text':
                yield ev('content_block_start', {'type': 'content_block_start', 'index': i,
                                                 'content_block': {'type': 'text', 'text': ''}})
                for tok in _tokens(bl['text']):
                    yield ev('content_block_delta', {'type': 'content_block_delta', 'index': i,
                                                     'delta': {'type': 'text_delta', 'text': tok}})
                    await asyncio.sleep(gap)
                if first_text and b.get('mock_citation'):
                    yield ev('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': {
                        'type': 'citations_delta', 'citation': {'type': 'char_location', 'cited_text': 'x',
                                                                'document_index': 0, 'start_char_index': 0,
                                                                'end_char_index': 1}}})
                first_text = False
            else:
                yield ev('content_block_start', {'type': 'content_block_start', 'index': i, 'content_block':
                                                 {'type': 'tool_use', 'id': bl['id'], 'name': bl['name'], 'input': {}}})
                yield ev('content_block_delta', {'type': 'content_block_delta', 'index': i,
                                                 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(bl['input'])}})
            yield ev('content_block_stop', {'type': 'content_block_stop', 'index': i})
        yield ev('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': stop, 'stop_sequence': None},
                                   'usage': {'output_tokens': MSG_USAGE['output_tokens']}})
        yield ev('message_stop', {'type': 'message_stop'})

    return StreamingResponse(gen(), media_type='text/event-stream')


if __name__ == '__main__':
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=9000)
    a = p.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=a.port, log_level='warning')
