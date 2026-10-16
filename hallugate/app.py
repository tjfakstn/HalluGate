"""OpenAI 호환 검증 게이트웨이.

서비스는 base_url만 이 게이트웨이로 바꾸면 된다.
    client = OpenAI(base_url="http://localhost:8000/v1", api_key=...)
    client.chat.completions.create(..., stream=True,
        extra_body={"verification": {"context": 근거문서, "mode": "hold", "action": "cut"}})

스트리밍 응답에는 일반 OpenAI 청크 사이에 판정 결과 청크가 섞여 나간다.
    {"object": "chat.completion.chunk", "choices": [], "verification": {...}}
choices가 비어 있으므로 일반 클라이언트는 본문에 영향 없이 무시하고, 필요한 클라이언트만
`chunk.verification` 으로 읽는다. 끊기면 finish_reason="content_filter".
"""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import context as ctxmod
from .config import Settings
from .pipeline import RequestRecord, VerifyPipeline
from .splitter import IncrementalSplitter


def _sse(obj):
    return f'data: {json.dumps(obj, ensure_ascii=False)}\n\n'


async def _upstream_deltas(resp, state):
    """업스트림 SSE → 본문 텍스트 조각. 마지막 finish_reason과 청크 메타데이터를 state에 남긴다."""
    async for line in resp.aiter_lines():
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        chunk = json.loads(data)
        state.setdefault('id', chunk.get('id'))
        state.setdefault('model', chunk.get('model'))
        for ch in chunk.get('choices') or []:
            if ch.get('finish_reason'):
                state['finish_reason'] = ch['finish_reason']
            text = (ch.get('delta') or {}).get('content')
            if text:
                yield text


def create_app(settings=None, judge=None, http_client=None):
    settings = settings or Settings()
    state = {'judge': judge, 'client': http_client}

    @asynccontextmanager
    async def lifespan(app):
        if state['judge'] is None:
            from .judge import StudentJudge
            state['judge'] = StudentJudge(settings.judge_model_dir, settings.judge_embedder, settings.judge_k,
                                          settings.judge_max_len, settings.judge_chunk_words,
                                          settings.judge_threshold, settings.judge_device)
        # 예열: 첫 요청에서 GPU 커널 준비로 판정이 수백 ms 걸리는 것을 막는다 (S3 측정에서 첫 요청 468ms 관측)
        warm = state['judge'].prepare('warm up reference text ' * 40)
        for _ in range(3):
            state['judge'].judge(warm, 'This is a warm up sentence.')
        own_client = state['client'] is None
        if own_client:
            state['client'] = httpx.AsyncClient(timeout=settings.upstream_timeout)
        yield
        if own_client:
            await state['client'].aclose()

    app = FastAPI(title='hallugate', lifespan=lifespan)

    def _log(rec):
        if settings.metrics_path:
            with open(settings.metrics_path, 'a') as f:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + '\n')

    def _headers(request):
        h = {'Content-Type': 'application/json'}
        auth = f'Bearer {settings.upstream_api_key}' if settings.upstream_api_key else request.headers.get('authorization')
        if auth:
            h['Authorization'] = auth
        return h

    @app.get('/health')
    async def health():
        return {'ok': True, 'judge': type(state['judge']).__name__,
                'mode': settings.default_mode, 'action': settings.default_action}

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        raw = await request.json()
        try:
            body, ref, mode, action, stride = ctxmod.extract(raw, settings.default_mode, settings.default_action,
                                                             settings.judge_stride, with_stride=True)
        except ValueError as e:
            return JSONResponse({'error': {'message': str(e), 'type': 'invalid_request_error'}}, status_code=400)

        client, judge = state['client'], state['judge']
        url = settings.upstream_base_url.rstrip('/') + '/chat/completions'
        headers = _headers(request)
        rec = RequestRecord(mode=mode, action=action, stride=stride, tag=request.headers.get('x-request-tag', ''))
        vh = {'x-verification-mode': mode, 'x-verification-action': action, 'x-verification-stride': str(stride)}

        # ---------- 비스트리밍 ----------
        if not body.get('stream'):
            r = await client.post(url, json=body, headers=headers)
            if r.status_code != 200 or mode == 'off':
                return Response(r.content, status_code=r.status_code, media_type='application/json', headers=vh)
            data = r.json()
            content = data['choices'][0]['message'].get('content') or ''
            ref_idx = await asyncio.to_thread(judge.prepare, ref)
            rec.prepare_ms = ref_idx.prepare_ms
            sp = IncrementalSplitter()
            sents = sp.feed(content) + sp.flush()
            results, kept = [], ''
            for s in sents:
                if not s.clean:
                    kept += s.text
                    continue
                v = await asyncio.to_thread(judge.judge, ref_idx, s.clean)
                results.append({'sentence_idx': s.idx, 'start': s.start, 'text': s.clean, 'score': v.score,
                                'hallucinated': v.hallucinated, 'judge_ms': v.judge_ms})
                rec.sentences.append(dict(results[-1]))
                if v.hallucinated and action == 'cut':
                    rec.cut = True
                    break
                kept += s.text
            if rec.cut and mode == 'hold':            # hold+cut: 환각 문장부터 잘라서 반환
                data['choices'][0]['message']['content'] = kept
                data['choices'][0]['finish_reason'] = 'content_filter'
            data['verification'] = {'mode': mode, 'action': action, 'prepare_ms': ref_idx.prepare_ms,
                                    'sentences': results, 'cut': rec.cut}
            rec.t_client_done = rec.now()
            _log(rec)
            return JSONResponse(data, headers=vh)

        # ---------- 스트리밍 ----------
        req = client.build_request('POST', url, json=body, headers=headers)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            err = await resp.aread()
            await resp.aclose()
            return Response(err, status_code=resp.status_code, media_type='application/json', headers=vh)

        if mode == 'off':
            async def passthrough():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
            return StreamingResponse(passthrough(), media_type='text/event-stream', headers=vh)

        async def gen():
            meta = {}
            cid = 'chatcmpl-' + uuid.uuid4().hex[:24]
            created = int(time.time())

            def chunk(delta=None, finish=None, verification=None):
                c = {'id': meta.get('id') or cid, 'object': 'chat.completion.chunk', 'created': created,
                     'model': meta.get('model') or body.get('model', ''),
                     'choices': [] if delta is None and finish is None else
                     [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]}
                if verification is not None:
                    c['verification'] = verification
                return _sse(c)

            try:
                ref_idx = await asyncio.to_thread(judge.prepare, ref)
                rec.prepare_ms = ref_idx.prepare_ms
                yield chunk(delta={'role': 'assistant', 'content': ''})
                pipe = VerifyPipeline(judge, ref_idx, mode, action, rec, stride=stride)
                async for ev in pipe.run(_upstream_deltas(resp, meta)):
                    if ev.kind == 'text':
                        if ev.text:
                            yield chunk(delta={'content': ev.text})
                    elif ev.kind == 'verdict':
                        yield chunk(verification=ev.info)
                    elif ev.kind == 'cut':
                        yield chunk(finish='content_filter', verification={'cut': True, **ev.info})
                if not rec.cut:
                    yield chunk(finish=meta.get('finish_reason', 'stop'))
                yield 'data: [DONE]\n\n'
            finally:
                await resp.aclose()
                _log(rec)

        return StreamingResponse(gen(), media_type='text/event-stream', headers=vh)

    return app
