"""API 키 없이 게이트웨이를 시험하기 위한 가짜 LLM (OpenAI 호환).

요청 body에 다음 필드를 넣으면 그대로 재생한다 (실제 API로는 보내지 말 것):
  mock_response : 흘려보낼 답변 텍스트 (없으면 마지막 user 메시지를 되돌려줌)
  mock_ttft_ms  : 첫 토큰까지 지연 (기본 300ms)
  mock_token_ms : 토큰 간 간격 (기본 25ms ≈ 초당 40토큰, 상용 API와 비슷한 속도)

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
                                          'finish_reason': 'stop'}]})

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
        yield c({}, 'stop')
        yield 'data: [DONE]\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


if __name__ == '__main__':
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=9000)
    a = p.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=a.port, log_level='warning')
