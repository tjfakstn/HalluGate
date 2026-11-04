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
from .pipeline import RequestRecord, VerifyPipeline, is_english
from .splitter import IncrementalSplitter
from .tokens import make_tokens


def _sse(obj):
    return f'data: {json.dumps(obj, ensure_ascii=False)}\n\n'


PASS_KEYS = ('tool_calls', 'function_call', 'refusal')


async def _upstream_deltas(resp, state):
    """업스트림 SSE → 본문 텍스트 조각(str) 또는 그대로 통과시킬 청크(dict).
    마지막 finish_reason, usage, 청크 메타데이터를 state에 남긴다 (SPEC §2.7, A10)."""
    async for line in resp.aiter_lines():
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        chunk = json.loads(data)
        state.setdefault('id', chunk.get('id'))
        state.setdefault('model', chunk.get('model'))
        if chunk.get('usage'):
            state['usage'] = chunk['usage']                 # 종료 청크 뒤에 그대로 보낸다
        for ch in chunk.get('choices') or []:
            if ch.get('finish_reason'):
                state['finish_reason'] = ch['finish_reason']
            delta = ch.get('delta') or {}
            other = {k: delta[k] for k in PASS_KEYS if delta.get(k)}
            if other:                                       # 본문이 아닌 부분(tool_calls 등)은 판정 없이 통과
                yield {**chunk, 'choices': [{'index': ch.get('index', 0), 'delta': other, 'finish_reason': None}],
                       **({'usage': None} if 'usage' in chunk else {})}
            text = delta.get('content')
            if text:
                yield text


async def _prepare(judge, ref, rec):
    """근거 준비. 실패하면 None (그 요청은 검증 없이 전달, fail-open)."""
    try:
        ref_idx = await asyncio.to_thread(judge.prepare, ref)
    except Exception:                                    # noqa: BLE001
        rec.verified, rec.unverified_reason = False, 'judge_error'
        return None
    rec.prepare_ms = ref_idx.prepare_ms
    return ref_idx


def create_app(settings=None, judge=None, http_client=None):
    settings = settings or Settings()
    state = {'judge': judge, 'client': http_client}
    tokens = make_tokens(settings.stride_tokenizer)

    def default_stride():
        return settings.judge_stride if settings.judge_stride is not None else getattr(state['judge'], 'default_stride', 0)

    @asynccontextmanager
    async def lifespan(app):
        if state['judge'] is None:
            from .judge import load_judge
            state['judge'] = load_judge(settings)
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
        j, st = state['judge'], default_stride()
        return {'ok': True, 'judge': type(j).__name__, 'input_mode': getattr(j, 'mode', ''),
                'mode': settings.default_mode, 'action': settings.default_action, 'stride': st,
                'stride_unit': tokens.name, 'threshold': j.threshold_for(st)}

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        raw = await request.json()
        try:
            body, ref, mode, action, stride = ctxmod.extract(raw, settings.default_mode, settings.default_action,
                                                             default_stride(), with_stride=True)
        except ValueError as e:
            return JSONResponse({'error': {'message': str(e), 'type': 'invalid_request_error'}}, status_code=400)

        client, judge = state['client'], state['judge']
        url = settings.upstream_base_url.rstrip('/') + '/chat/completions'
        headers = _headers(request)
        rec = RequestRecord(mode=mode, action=action, stride=stride, tag=request.headers.get('x-request-tag', ''))
        thr = judge.threshold_for(stride)
        vh = {'x-verification-mode': mode, 'x-verification-action': action, 'x-verification-stride': str(stride)}

        # ---------- 비스트리밍 ----------
        if not body.get('stream'):
            r = await client.post(url, json=body, headers=headers)
            if r.status_code != 200 or mode == 'off':
                return Response(r.content, status_code=r.status_code, media_type='application/json', headers=vh)
            data = r.json()
            content = data['choices'][0]['message'].get('content') or ''
            thr = judge.threshold_for(0)                  # 비스트리밍은 문장 끝에서만 판정
            ref_idx = await _prepare(judge, ref, rec)
            rec.threshold = thr
            sp = IncrementalSplitter()
            sents = sp.feed(content) + sp.flush()
            results, kept = [], ''
            for s in sents:
                if not s.clean or not rec.verified:
                    kept += s.text
                    continue
                reason = None
                if not is_english(s.clean):
                    reason = 'non_english'
                else:
                    try:
                        v = await asyncio.wait_for(asyncio.to_thread(judge.judge, ref_idx, s.clean, content[:s.start], thr),
                                                   settings.judge_timeout_ms / 1000)
                    except asyncio.TimeoutError:
                        reason = 'judge_timeout'
                    except Exception:                                   # noqa: BLE001 — fail-open
                        reason = 'judge_error'
                if reason:
                    rec.verified, rec.unverified_reason = False, reason
                    kept += s.text
                    continue
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
            data['verification'] = {'mode': mode, 'action': action, 'stride': 0, 'threshold': thr,
                                    'prepare_ms': rec.prepare_ms, 'sentences': results, 'cut': rec.cut,
                                    'verified': rec.verified,
                                    **({'reason': rec.unverified_reason} if not rec.verified else {})}
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
                ref_idx = await _prepare(judge, ref, rec)
                yield chunk(delta={'role': 'assistant', 'content': ''})
                pipe = VerifyPipeline(judge, ref_idx, mode, action, rec, stride=stride, tokens=tokens,
                                      threshold=thr, timeout_ms=settings.judge_timeout_ms)
                async for ev in pipe.run(_upstream_deltas(resp, meta)):
                    if ev.kind == 'raw':
                        yield _sse(ev.info)
                    elif ev.kind == 'text':
                        if ev.text:
                            yield chunk(delta={'content': ev.text})
                    elif ev.kind == 'verdict':
                        yield chunk(verification=ev.info)
                    elif ev.kind == 'cut':
                        yield chunk(finish='content_filter', verification={'cut': True, **ev.info})
                if not rec.cut:
                    yield chunk(finish=meta.get('finish_reason', 'stop'))
                if meta.get('usage'):
                    yield _sse({**json.loads(chunk()[6:]), 'usage': meta['usage']})
                yield 'data: [DONE]\n\n'
            finally:
                await resp.aclose()
                _log(rec)

        return StreamingResponse(gen(), media_type='text/event-stream', headers=vh)

    return app
