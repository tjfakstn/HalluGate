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
import os
import time
import uuid
from types import SimpleNamespace
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import context as ctxmod
from .config import Settings
from .pipeline import RequestRecord, VerifyPipeline, is_english
from .limits import Limiter
from .metrics import MetricsStore, parse_window, rejected, summarize
from .trim import Trimmer
from .postcheck import skipped
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


async def _post_check(checker, ref, text, rec, cut, timeout_ms, allow=None):
    """응답 끝 2단 검증 (SPEC §5.3). 본문은 바꾸지 않고 결과 dict만 돌려준다. 실패는 verified=False(fail-open)."""
    if cut:
        res = skipped('cut')
    elif checker is None:
        res = skipped('unavailable')
    else:
        sp = IncrementalSplitter()
        sents = [(s.idx, s.start, s.clean) for s in sp.feed(text) + sp.flush() if s.clean]
        if not sents:
            res = skipped('empty')
        elif allow is not None and not allow():          # 운영 모드 하루 한도 (SPEC §10.4)
            res = skipped('quota')
        else:
            try:
                res = await asyncio.wait_for(checker.check(ref, sents), timeout_ms / 1000)
            except asyncio.TimeoutError:
                res = skipped('timeout')
            except Exception:                                   # noqa: BLE001 — fail-open
                res = skipped('error')
    rec.post_check = res.to_dict()
    return rec.post_check


def _err(msg, status=400, kind='invalid_request_error'):
    return JSONResponse({'error': {'message': msg, 'type': kind}}, status_code=status)


def create_app(settings=None, judge=None, http_client=None, store=None, post_checker=None, keys=None, limiter=None):
    settings = settings or Settings()
    if settings.hosted and keys is None:                  # 운영 모드 API 키 저장소 (SPEC §10.1)
        from .keys import KeyStore
        keys = KeyStore(settings.keys_db)
    if post_checker is None and settings.anthropic_api_key:
        from .postcheck import ClaudePostChecker
        post_checker = ClaudePostChecker(api_key=settings.anthropic_api_key, model=settings.post_check_model,
                                         votes=settings.post_check_votes, threshold=settings.post_check_threshold)
    state = {'judge': judge, 'client': http_client, 'store': store, 'post_checker': post_checker, 'keys': keys,
             'limiter': limiter or Limiter(), 'cold': True,
             'metrics': MetricsStore(settings.keys_db) if settings.hosted else None}   # 운영 모드 기록 (SPEC §10.5)
    allow_up = dict(x.split('=', 1) for x in settings.upstream_allowlist.split(',') if '=' in x)   # SPEC §10.3
    tokens = make_tokens(settings.stride_tokenizer)

    def default_stride():
        return settings.judge_stride if settings.judge_stride is not None else getattr(state['judge'], 'default_stride', 0)

    @asynccontextmanager
    async def lifespan(app):
        if state['judge'] is None:
            from .judge import load_judge
            state['judge'] = load_judge(settings)
        # 예열: 첫 요청에서 GPU 커널 준비로 판정이 수백 ms 걸리는 것을 막는다 (S3 측정에서 첫 요청 468ms 관측)
        t_w = time.perf_counter()
        warm = state['judge'].prepare('warm up reference text ' * 40)
        for _ in range(3):
            state['judge'].judge(warm, 'This is a warm up sentence.')
        if state['metrics'] is not None:                 # 콜드 스타트 관측·보관 기간 정리 (SPEC §10.5·10.6)
            state['metrics'].add_start((time.perf_counter() - t_w) * 1000)
            state['metrics'].prune(settings.metrics_retention_days)
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

    def _emit(row):
        # 운영 모드: 원문 없는 요약을 SQLite와 표준 출력에 (SPEC §10.5)
        state['metrics'].add(row)
        print(json.dumps({'event': 'request', **row}, ensure_ascii=False), flush=True)

    def _log(rec):
        if settings.hosted:
            row = summarize(rec.to_dict(), rec.status, rec.stream, rec.cold)
            _emit(row)
            if rec.tenant and row.get('tokens_in') is not None:     # 예산용 월 누적 (SPEC §10.10)
                state['keys'].add_month_usage(rec.tenant, (row['tokens_in'] or 0) + (row['tokens_out'] or 0),
                                              row.get('llm_cost_usd') or 0.0)
            return
        if settings.metrics_path:
            with open(settings.metrics_path, 'a') as f:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + '\n')

    def _headers(request):
        h = {'Content-Type': 'application/json'}
        if settings.hosted:                              # 운영 모드: 사용자 LLM 키만, 운영자 키는 절대 안 씀 (SPEC §10.3)
            auth = request.headers.get('authorization')
        else:
            auth = f'Bearer {settings.upstream_api_key}' if settings.upstream_api_key else request.headers.get('authorization')
        if auth:
            h['Authorization'] = auth
        return h

    @app.get('/demo', include_in_schema=False)
    async def demo():
        # 발표·시연용 화면 (SPEC §2.8). 외부 리소스 없는 HTML 한 장
        path = os.path.join(os.path.dirname(__file__), 'static', 'demo.html')
        if not settings.demo or not os.path.isfile(path):
            return _err('demo page is disabled', 404, 'not_found_error')
        return HTMLResponse(open(path, encoding='utf-8').read())

    @app.get('/health')
    async def health():
        if settings.hosted:                              # 운영 모드: 판정기 내부 값은 숨긴다 (SPEC §10.1)
            return {'ok': True, 'hosted': True}
        j, st, pc = state['judge'], default_stride(), state['post_checker']
        return {'ok': True, 'judge': type(j).__name__, 'input_mode': getattr(j, 'mode', ''),
                'mode': settings.default_mode, 'action': settings.default_action, 'stride': st,
                'stride_unit': tokens.name, 'threshold': j.threshold_for(st),
                'collections': state['store'] is not None, 'collections_auth': bool(settings.admin_token),
                'post_check': {'available': pc is not None, 'default': settings.post_check_default,
                               'model': getattr(pc, 'model', settings.post_check_model),
                               'votes': getattr(pc, 'votes', settings.post_check_votes),
                               'threshold': getattr(pc, 'threshold', settings.post_check_threshold)}}

    # ---------- 컬렉션 문서 등록 API (SPEC §2.5) ----------
    def _tenant(request):
        """운영 모드 인증 (SPEC §10.1): (Tenant 또는 None, 오류 응답 또는 None). 셀프호스팅이면 (None, None)."""
        if not settings.hosted:
            return None, None
        t = state['keys'].verify(request.headers.get('x-hallugate-key'))
        if t is None:
            return None, _err('invalid or missing x-hallugate-key', 401, 'authentication_error')
        return t, None

    def _cname(t, cid):
        # 운영 모드에서는 컬렉션 이름 앞에 키 ID를 붙여 사용자별로 분리한다 (SPEC §10.2)
        return f'{t.key_id}/{cid}' if t is not None else cid

    def _docs_access(request):
        """문서 API 접근 확인: (Tenant 또는 None, 오류 응답 또는 None)."""
        if settings.hosted:
            t, e = _tenant(request)
            if e:
                return None, e
        else:
            t = None
            if settings.admin_token and request.headers.get('authorization') != f'Bearer {settings.admin_token}':
                return None, _err('invalid or missing admin token', 401, 'authentication_error')
        if state['store'] is None:
            return None, _err('collections are disabled (COLLECTION_DB is empty)', 503, 'unavailable')
        return t, None

    @app.get('/v1/collections')
    async def list_collections(request: Request):
        t, e = _docs_access(request)
        if e:
            return e
        rows = await asyncio.to_thread(state['store'].collections)
        if t is not None:
            pre = _cname(t, '')
            rows = [{**r, 'collection': r['collection'][len(pre):]} for r in rows if r['collection'].startswith(pre)]
        return {'object': 'list', 'data': rows}

    @app.post('/v1/collections/{cid}/documents')
    async def add_documents(cid: str, request: Request):
        """{"text": ..., "title"?: ..., "doc_id"?: ...} 또는 {"documents": [그런 객체, ...]}. 같은 doc_id면 교체."""
        t, e = _docs_access(request)
        if e:
            return e
        b = await request.json()
        docs = b.get('documents') if isinstance(b.get('documents'), list) else [b]
        if not docs or not all(isinstance(d, dict) and isinstance(d.get('text'), str) and d['text'].strip() for d in docs):
            return _err('each document needs a non-empty "text"')
        store, name = state['store'], _cname(t, cid)
        if t is not None:                                # 운영 모드 크기 한도 (SPEC §10.2)
            if any(len(d['text']) > settings.max_doc_chars for d in docs):
                return _err(f'document longer than {settings.max_doc_chars} characters', 413, 'request_too_large')
            new = sum(1 for d in docs if not d.get('doc_id') or not store.doc_exists(name, d['doc_id']))
            if await asyncio.to_thread(store.count_docs, _cname(t, '')) + new > settings.max_docs:
                return _err(f'document limit ({settings.max_docs}) reached', 413, 'request_too_large')
        out = []
        for d in docs:
            out.append(await asyncio.to_thread(store.add, name, d['text'], d.get('title', ''), d.get('doc_id')))
        return {'object': 'list', 'collection': cid, 'data': out}

    @app.get('/v1/collections/{cid}/documents')
    async def list_documents(cid: str, request: Request):
        t, e = _docs_access(request)
        if e:
            return e
        name = _cname(t, cid)
        if t is not None and not await asyncio.to_thread(state['store'].exists, name):
            return _err(f'collection {cid} not found', 404, 'not_found_error')
        return {'object': 'list', 'collection': cid, 'data': await asyncio.to_thread(state['store'].list, name)}

    @app.delete('/v1/collections/{cid}/documents/{doc_id}')
    async def delete_document(cid: str, doc_id: str, request: Request):
        t, e = _docs_access(request)
        if e:
            return e
        if not await asyncio.to_thread(state['store'].delete, _cname(t, cid), doc_id):
            return _err(f'document {doc_id} not found in {cid}', 404, 'not_found_error')
        return {'deleted': True, 'collection': cid, 'doc_id': doc_id}

    def _admin_only(request):
        # 관리자 API (SPEC §10.1·10.6): 운영 모드에서만, ADMIN_TOKEN 필요
        if not settings.hosted:
            return _err('admin API is available only in hosted mode', 404, 'not_found_error')
        if not settings.admin_token or request.headers.get('authorization') != f'Bearer {settings.admin_token}':
            return _err('invalid or missing admin token', 401, 'authentication_error')
        return None

    @app.post('/admin/keys', include_in_schema=False)
    async def admin_create_key(request: Request):
        if (e := _admin_only(request)):
            return e
        b = await request.json() if await request.body() else {}
        limits = b.get('limits') or {}
        if not _valid_limits(limits):
            return _err('limits: integer rpm / concurrent / post_check_daily / monthly_tokens, number monthly_usd, '
                        'budget_action warn | block')
        kid, key = await asyncio.to_thread(state['keys'].create, str(b.get('name', '')), limits)
        return {'key_id': kid, 'key': key}

    @app.get('/admin/keys', include_in_schema=False)
    async def admin_list_keys(request: Request):
        if (e := _admin_only(request)):
            return e
        return {'object': 'list', 'data': await asyncio.to_thread(state['keys'].list)}

    @app.post('/admin/keys/{key_id}/revoke', include_in_schema=False)
    async def admin_revoke_key(key_id: str, request: Request):
        if (e := _admin_only(request)):
            return e
        b = await request.json() if await request.body() else {}
        out = {'key_id': key_id, 'revoked': await asyncio.to_thread(state['keys'].revoke, key_id)}
        if b.get('purge'):                               # 그 사용자의 컬렉션·기록 삭제 (SPEC §10.5)
            out['purged_docs'] = (await asyncio.to_thread(state['store'].purge, key_id + '/')) if state['store'] else 0
            out['purged_records'] = await asyncio.to_thread(state['metrics'].purge, key_id)
        return out

    @app.get('/admin/stats', include_in_schema=False)
    async def admin_stats(request: Request, window: str = '24h'):
        # 관리자 통계 (SPEC §10.6)
        if (e := _admin_only(request)):
            return e
        try:
            w = parse_window(window)
        except ValueError as ex:
            return _err(str(ex))
        m = state['metrics']
        await asyncio.to_thread(m.prune, settings.metrics_retention_days)
        return await asyncio.to_thread(m.stats, w, window)

    def _valid_limits(lm):
        if not isinstance(lm, dict):
            return False
        for k, v in lm.items():
            if k in ('rpm', 'concurrent', 'post_check_daily', 'monthly_tokens'):
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    return False
            elif k == 'monthly_usd':
                if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                    return False
            elif k == 'budget_action':
                if v not in ('warn', 'block'):
                    return False
            else:
                return False
        return True

    @app.get('/v1/usage')
    async def usage(request: Request, window: str = '30d'):
        # 사용자 본인 LLM 사용량·예산 (SPEC §10.10)
        if not settings.hosted:
            return _err('usage is available only in hosted mode', 404, 'not_found_error')
        t, e = _tenant(request)
        if e:
            return e
        try:
            w = parse_window(window)
        except ValueError as ex:
            return _err(str(ex))
        out = await asyncio.to_thread(state['metrics'].usage, t.key_id, w)
        used, u = _budget(t)
        lm = t.limits or {}
        out['budget'] = None if used is None else {
            'month': u['month'], 'used': round(used, 4), 'tokens_used': u['tokens'], 'cost_used_usd': round(u['cost_usd'], 6),
            'monthly_tokens': lm.get('monthly_tokens'), 'monthly_usd': lm.get('monthly_usd'),
            'action': lm.get('budget_action', 'warn')}
        return {'window': window, **out}

    def _trim_embedder():
        # 근거 다이어트 임베더 = 컬렉션 검색과 같은 것 (SPEC §2.9)
        if state['store'] is not None:
            return state['store'].embedder
        emb = getattr(state['judge'], 'embedder', None)
        if hasattr(emb, 'encode') and not settings.collection_embedder:
            return emb
        if state.get('trim_emb') is None:
            from sentence_transformers import SentenceTransformer
            state['trim_emb'] = SentenceTransformer(settings.collection_embedder or 'sentence-transformers/all-MiniLM-L6-v2')
        return state['trim_emb']

    def _lim(t, name, default):
        return int((t.limits or {}).get(name, default))

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        tenant, e = _tenant(request)
        if e:
            if settings.hosted:
                _emit(rejected('', e.status_code))
            return e
        if tenant is None:
            return await _chat(request, None)
        resp = await _limited_chat(request, tenant)
        if not isinstance(resp, StreamingResponse) and not getattr(request.state, 'hg_logged', False):
            _emit(rejected(tenant.key_id, resp.status_code, request.headers.get('x-upstream', 'openai')))
        return resp

    def _budget(tenant):
        """이번 달 예산 사용 비율 (SPEC §10.10). 예산이 없으면 None."""
        lm = tenant.limits or {}
        if 'monthly_tokens' not in lm and 'monthly_usd' not in lm:
            return None, None
        u = state['keys'].month_usage(tenant.key_id)
        fr = [u['tokens'] / lm['monthly_tokens'] if lm.get('monthly_tokens') else 0.0,
              u['cost_usd'] / lm['monthly_usd'] if lm.get('monthly_usd') else 0.0]
        return max(fr), u

    async def _limited_chat(request, tenant):
        used, _ = _budget(tenant)
        resp = await _limited_chat_inner(request, tenant, used)
        if used is not None:
            resp.headers['x-budget-used'] = f'{used:.3f}'
            if used >= 1.0:
                resp.headers['x-budget-warning'] = 'exceeded'
            elif used >= 0.8:
                resp.headers['x-budget-warning'] = 'near'
        return resp

    async def _limited_chat_inner(request, tenant, used):
        if used is not None and used >= 1.0 and (tenant.limits or {}).get('budget_action') == 'block':
            return _err('monthly budget exceeded', 429, 'insufficient_quota')
        # 운영 모드 한도 (SPEC §10.4): 분당 → 동시. 거절된 요청은 업스트림을 부르지 않는다
        lim = state['limiter']
        ra = lim.hit(tenant.key_id, _lim(tenant, 'rpm', settings.rate_rpm))
        if ra is not None:
            r = _err('rate limit exceeded (requests per minute)', 429, 'rate_limit_error')
            r.headers['retry-after'] = str(ra)
            return r
        if not lim.acquire(tenant.key_id, _lim(tenant, 'concurrent', settings.max_concurrent)):
            r = _err('too many concurrent requests', 429, 'rate_limit_error')
            r.headers['retry-after'] = '1'
            return r
        try:
            resp = await _chat(request, tenant)
        except BaseException:
            lim.release(tenant.key_id)
            raise
        if isinstance(resp, StreamingResponse):          # 스트림이 끝나거나 끊길 때 풀어 준다
            inner = resp.body_iterator

            async def released():
                try:
                    async for x in inner:
                        yield x
                finally:
                    lim.release(tenant.key_id)
            resp.body_iterator = released()
        else:
            lim.release(tenant.key_id)
        return resp

    async def _chat(request, tenant):
        if tenant is not None:                           # 업스트림 허용 목록 (SPEC §10.3)
            up_name = request.headers.get('x-upstream', 'openai')
            if up_name not in allow_up:
                return _err(f'x-upstream must be one of {sorted(allow_up)}')
            up_base = allow_up[up_name]
        else:
            up_name, up_base = '', settings.upstream_base_url
        raw = await request.json()
        try:
            body, ref, mode, action, stride, opts = ctxmod.extract(raw, settings.default_mode, settings.default_action,
                                                                   default_stride(), with_stride=True, with_opts=True)
        except ValueError as e:
            return _err(str(e))
        if tenant is not None and ref and len(ref) > settings.max_ref_chars:
            return _err(f'reference longer than {settings.max_ref_chars} characters', 413, 'request_too_large')

        # 컬렉션 RAG (SPEC §2.6): 요청에 근거가 없고 collection이 있으면 검색 → 프롬프트 삽입 → 같은 근거로 판정
        evidence, coll = None, opts.get('collection') if not ref else None
        t_r = time.perf_counter()
        if coll:
            store = state['store']
            if store is None:
                return _err('collections are disabled (COLLECTION_DB is empty)', 503, 'unavailable')
            cname = _cname(tenant, coll)
            if not await asyncio.to_thread(store.exists, cname):
                return _err(f'collection {coll} not found', 404, 'not_found_error')
            evidence = await asyncio.to_thread(store.search, cname, ctxmod.last_user_text(body), settings.collection_top_k)
            if evidence:
                body['messages'] = ctxmod.rag_messages(body.get('messages', []), evidence)
                ref = ctxmod.rag_reference(evidence)
            else:
                mode = 'off'                               # 질문이 비어 근거를 못 찾음 → 검증 없이 전달
        retrieve_ms = (time.perf_counter() - t_r) * 1000

        # 근거 다이어트 (SPEC §2.9): 관련 조각만 LLM에 보내고 같은 조각으로 판정
        trim_info, tr = None, opts.get('trim')
        if tr and ref and not coll and mode != 'off':
            k = tr['top_k'] if isinstance(tr, dict) else settings.trim_top_k
            from_tag = not opts.get('context')
            query = ctxmod.strip_context_tags(ctxmod.last_user_text(body))
            trimmer = Trimmer(_trim_embedder(), settings.collection_chunk_words, settings.collection_overlap)
            kept, total = await asyncio.to_thread(trimmer.select, ref, query, k)
            new_ref = '\n\n'.join(kept) if total > k else ref
            if total > k:
                if from_tag:
                    body['messages'] = ctxmod.replace_context_tags(body.get('messages', []), new_ref)
                else:
                    body['messages'] = ctxmod.rag_messages(body.get('messages', []),
                                                           [SimpleNamespace(text=c) for c in kept])
            trim_info = {'kind': 'trim', 'chunks_total': total, 'chunks_kept': min(total, k) if total > k else total,
                         'chars_before': len(ref), 'chars_after': len(new_ref),
                         'saved_ratio': round(1 - len(new_ref) / max(len(ref), 1), 4), 'top_k': k}
            ref = new_ref

        client, judge = state['client'], state['judge']
        url = up_base.rstrip('/') + '/chat/completions'
        headers = _headers(request)
        rec = RequestRecord(mode=mode, action=action, stride=stride, tag=request.headers.get('x-request-tag', ''),
                            collection=coll or '', retrieve_ms=retrieve_ms if coll else 0.0,
                            n_evidence=len(evidence or []), tenant=tenant.key_id if tenant else '', upstream=up_name,
                            ref_chars=len(ref or ''), stream=bool(body.get('stream')), cold=state.pop('cold', False))
        thr = judge.threshold_for(stride)
        pc_on = bool(opts.get('post_check', settings.post_check_default)) and mode != 'off'
        vh = {'x-verification-mode': mode, 'x-verification-action': action, 'x-verification-stride': str(stride),
              'x-verification-post-check': 'on' if pc_on else 'off'}
        if coll:
            vh['x-verification-evidence'] = str(len(evidence or []))
        if tenant is not None:
            vh['x-upstream'] = up_name
        pc_allow = ((lambda: state['keys'].take_post_check(tenant.key_id, _lim(tenant, 'post_check_daily', settings.post_check_daily)))
                    if tenant is not None else None)
        ev_info = ({'kind': 'evidence', 'collection': coll, 'retrieve_ms': retrieve_ms,
                    'evidence': [e.to_dict() for e in evidence]} if evidence else None)
        if trim_info:
            vh['x-verification-trim'] = f"{trim_info['chunks_kept']}/{trim_info['chunks_total']}"
            rec.trim_chars_before, rec.trim_chars_after = trim_info['chars_before'], trim_info['chars_after']
            rec.trim_chunks_kept, rec.trim_chunks_total = trim_info['chunks_kept'], trim_info['chunks_total']

        # ---------- 비스트리밍 ----------
        if not body.get('stream'):
            r = await client.post(url, json=body, headers=headers)
            if r.status_code != 200 or mode == 'off':
                if r.status_code == 200:                 # 검증을 끈 요청도 기록 (오류는 바깥에서 상태 코드로 기록)
                    try:
                        _d = r.json(); rec.usage, rec.model = _d.get('usage'), _d.get('model', '')
                    except ValueError:
                        pass
                    rec.t_client_done = rec.now()
                    _log(rec)
                    request.state.hg_logged = True
                return Response(r.content, status_code=r.status_code, media_type='application/json', headers=vh)
            data = r.json()
            rec.usage, rec.model = data.get('usage'), data.get('model', '')
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
                                    **({'trim': trim_info} if trim_info else {}),
                                    **({'reason': rec.unverified_reason} if not rec.verified else {})}
            if pc_on:
                data['verification']['post_check'] = await _post_check(state['post_checker'], ref, content, rec, rec.cut,
                                                                       settings.post_check_timeout_ms, pc_allow)
            rec.t_client_done = rec.now()
            _log(rec)
            request.state.hg_logged = True
            return JSONResponse(data, headers=vh)

        # ---------- 스트리밍 ----------
        wants_usage = bool((body.get('stream_options') or {}).get('include_usage'))
        if tenant is not None:                           # 사용량 수집 (SPEC §10.10): 업스트림에는 항상 요청
            body['stream_options'] = {**(body.get('stream_options') or {}), 'include_usage': True}
        else:
            wants_usage = True                           # 셀프호스팅: 업스트림이 보낸 그대로
        req = client.build_request('POST', url, json=body, headers=headers)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            err = await resp.aread()
            await resp.aclose()
            return Response(err, status_code=resp.status_code, media_type='application/json', headers=vh)

        if mode == 'off':
            async def passthrough():
                try:
                    if tenant is None:
                        async for chunk in resp.aiter_raw():
                            yield chunk
                    else:                                # 운영 모드: usage만 가로채고 나머지는 그대로 (SPEC §10.10)
                        async for line in resp.aiter_lines():
                            if line.startswith('data:') and line[5:].strip() != '[DONE]':
                                try:
                                    c = json.loads(line[5:])
                                except ValueError:
                                    c = {}
                                rec.model = c.get('model') or rec.model
                                if c.get('usage'):
                                    rec.usage = c['usage']
                                    if not c.get('choices') and not wants_usage:
                                        continue
                            if line:
                                yield line + '\n\n'
                finally:
                    await resp.aclose()
                    rec.t_client_done = rec.now()
                    _log(rec)
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

            delivered = ''                                        # 사용자에게 나간 본문 (2단 검증 대상)
            try:
                ref_idx = await _prepare(judge, ref, rec)
                yield chunk(delta={'role': 'assistant', 'content': ''})
                if ev_info:                                       # 사용한 근거 조각을 본문보다 먼저 한 번 보낸다
                    yield chunk(verification=ev_info)
                if trim_info:                                     # 근거 다이어트 결과 (SPEC §2.9)
                    yield chunk(verification=trim_info)
                pipe = VerifyPipeline(judge, ref_idx, mode, action, rec, stride=stride, tokens=tokens,
                                      threshold=thr, timeout_ms=settings.judge_timeout_ms)
                async for ev in pipe.run(_upstream_deltas(resp, meta)):
                    if ev.kind == 'raw':
                        yield _sse(ev.info)
                    elif ev.kind == 'text':
                        if ev.text:
                            delivered += ev.text
                            yield chunk(delta={'content': ev.text})
                    elif ev.kind == 'verdict':
                        yield chunk(verification=ev.info)
                    elif ev.kind == 'cut':
                        yield chunk(finish='content_filter', verification={'cut': True, **ev.info})
                if not rec.cut:
                    yield chunk(finish=meta.get('finish_reason', 'stop'))
                if pc_on:                                         # 종료 청크 뒤, usage·[DONE] 앞에 1회 (SPEC §5.3)
                    yield chunk(verification=await _post_check(state['post_checker'], ref, delivered, rec, rec.cut,
                                                               settings.post_check_timeout_ms, pc_allow))
                rec.usage, rec.model = meta.get('usage'), meta.get('model') or ''
                if meta.get('usage') and wants_usage:
                    yield _sse({**json.loads(chunk()[6:]), 'usage': meta['usage']})
                yield 'data: [DONE]\n\n'
            finally:
                await resp.aclose()
                _log(rec)

        return StreamingResponse(gen(), media_type='text/event-stream', headers=vh)

    return app
