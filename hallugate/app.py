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


def _err(msg, status=400, kind='invalid_request_error'):
    return JSONResponse({'error': {'message': msg, 'type': kind}}, status_code=status)


def create_app(settings=None, judge=None, http_client=None, store=None):
    settings = settings or Settings()
    state = {'judge': judge, 'client': http_client, 'store': store}
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
        if state['store'] is None and settings.collection_db:
            from .store import SQLiteStore
            emb = None if settings.collection_embedder else getattr(state['judge'], 'embedder', None)
            if emb is None or not hasattr(emb, 'encode'):
                from sentence_transformers import SentenceTransformer
                emb = SentenceTransformer(settings.collection_embedder or 'sentence-transformers/all-MiniLM-L6-v2')
            state['store'] = SQLiteStore(settings.collection_db, emb, settings.collection_chunk_words,
                                         settings.collection_overlap)
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
                'stride_unit': tokens.name, 'threshold': j.threshold_for(st),
                'collections': state['store'] is not None, 'collections_auth': bool(settings.admin_token)}

    # ---------- 컬렉션 문서 등록 API (SPEC §2.5) ----------
    def _admin(request):
        """문서 API 접근 확인. 문제가 있으면 오류 응답, 통과면 None."""
        if state['store'] is None:
            return _err('collections are disabled (COLLECTION_DB is empty)', 503, 'unavailable')
        if settings.admin_token and request.headers.get('authorization') != f'Bearer {settings.admin_token}':
            return _err('invalid or missing admin token', 401, 'authentication_error')
        return None

    @app.get('/v1/collections')
    async def list_collections(request: Request):
        if (e := _admin(request)):
            return e
        return {'object': 'list', 'data': await asyncio.to_thread(state['store'].collections)}

    @app.post('/v1/collections/{cid}/documents')
    async def add_documents(cid: str, request: Request):
        """{"text": ..., "title"?: ..., "doc_id"?: ...} 또는 {"documents": [그런 객체, ...]}. 같은 doc_id면 교체."""
        if (e := _admin(request)):
            return e
        b = await request.json()
        docs = b.get('documents') if isinstance(b.get('documents'), list) else [b]
        if not docs or not all(isinstance(d, dict) and isinstance(d.get('text'), str) and d['text'].strip() for d in docs):
            return _err('each document needs a non-empty "text"')
        out = []
        for d in docs:
            out.append(await asyncio.to_thread(state['store'].add, cid, d['text'], d.get('title', ''), d.get('doc_id')))
        return {'object': 'list', 'collection': cid, 'data': out}

    @app.get('/v1/collections/{cid}/documents')
    async def list_documents(cid: str, request: Request):
        if (e := _admin(request)):
            return e
        return {'object': 'list', 'collection': cid, 'data': await asyncio.to_thread(state['store'].list, cid)}

    @app.delete('/v1/collections/{cid}/documents/{doc_id}')
    async def delete_document(cid: str, doc_id: str, request: Request):
        if (e := _admin(request)):
            return e
        if not await asyncio.to_thread(state['store'].delete, cid, doc_id):
            return _err(f'document {doc_id} not found in {cid}', 404, 'not_found_error')
        return {'deleted': True, 'collection': cid, 'doc_id': doc_id}

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        raw = await request.json()
        try:
            body, ref, mode, action, stride, opts = ctxmod.extract(raw, settings.default_mode, settings.default_action,
                                                                   default_stride(), with_stride=True, with_opts=True)
        except ValueError as e:
            return _err(str(e))

        # 컬렉션 RAG (SPEC §2.6): 요청에 근거가 없고 collection이 있으면 검색 → 프롬프트 삽입 → 같은 근거로 판정
        evidence, coll = None, opts.get('collection') if not ref else None
        t_r = time.perf_counter()
        if coll:
            store = state['store']
            if store is None:
                return _err('collections are disabled (COLLECTION_DB is empty)', 503, 'unavailable')
            if not await asyncio.to_thread(store.exists, coll):
                return _err(f'collection {coll} not found', 404, 'not_found_error')
            evidence = await asyncio.to_thread(store.search, coll, ctxmod.last_user_text(body), settings.collection_top_k)
            if evidence:
                body['messages'] = ctxmod.rag_messages(body.get('messages', []), evidence)
                ref = ctxmod.rag_reference(evidence)
            else:
                mode = 'off'                               # 질문이 비어 근거를 못 찾음 → 검증 없이 전달
        retrieve_ms = (time.perf_counter() - t_r) * 1000

        client, judge = state['client'], state['judge']
        url = settings.upstream_base_url.rstrip('/') + '/chat/completions'
        headers = _headers(request)
        rec = RequestRecord(mode=mode, action=action, stride=stride, tag=request.headers.get('x-request-tag', ''),
                            collection=coll or '', retrieve_ms=retrieve_ms if coll else 0.0,
                            n_evidence=len(evidence or []))
        thr = judge.threshold_for(stride)
        vh = {'x-verification-mode': mode, 'x-verification-action': action, 'x-verification-stride': str(stride)}
        if coll:
            vh['x-verification-evidence'] = str(len(evidence or []))
        ev_info = ({'kind': 'evidence', 'collection': coll, 'retrieve_ms': retrieve_ms,
                    'evidence': [e.to_dict() for e in evidence]} if evidence else None)

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
                                    **({'collection': coll, 'evidence': ev_info['evidence']} if ev_info else {}),
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
                if ev_info:                                       # 사용한 근거 조각을 본문보다 먼저 한 번 보낸다
                    yield chunk(verification=ev_info)
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
