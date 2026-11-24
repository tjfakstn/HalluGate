"""Anthropic Messages 형식 입출력 (SPEC §2.10).

업스트림 SSE(`event:` + `data:`)를 읽어 본문 텍스트는 판정 파이프라인으로, 나머지 이벤트는 순서를 지켜
그대로 또는 본문 뒤로 미뤄 보낸다. 게이트웨이가 내보내는 블록 index는 보내는 순서대로 다시 매긴다.
"""
import json

PASS_HEADERS = ('x-api-key', 'authorization', 'anthropic-version', 'anthropic-beta')


def usage_to_openai(u):
    """Anthropic usage → 기록용 OpenAI 형식 usage (§10.10). 입력에는 캐시 생성·읽기 토큰을 더한다."""
    if not isinstance(u, dict) or u.get('input_tokens') is None:
        return None
    cr = u.get('cache_read_input_tokens') or 0
    tin = u['input_tokens'] + (u.get('cache_creation_input_tokens') or 0) + cr
    return {'prompt_tokens': tin, 'completion_tokens': u.get('output_tokens'),
            'prompt_tokens_details': {'cached_tokens': cr}}


def text_of(content):
    """응답 content 블록 중 text 블록만 순서대로 이어 붙인 본문 (판정 대상)."""
    return ''.join(b.get('text', '') for b in content or [] if isinstance(b, dict) and b.get('type') == 'text')


def ev(name, data):
    return f'event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


async def sse_events(resp):
    """업스트림 응답 → (이벤트 이름, data dict)."""
    name = None
    async for line in resp.aiter_lines():
        if line.startswith('event:'):
            name = line[6:].strip()
        elif line.startswith('data:'):
            try:
                data = json.loads(line[5:])
            except ValueError:
                continue
            yield name or data.get('type'), data
            name = None


async def deltas(events, st):
    """판정 파이프라인 입력: 본문 텍스트(str) 또는 지금 바로 보낼 이벤트(dict).

    st에 남기는 것: usage(message_start + message_delta), message_delta, deferred(본문이 시작된 뒤 온 다른 블록 이벤트).
    text 블록의 시작·끝은 게이트웨이가 다시 만들므로 버린다. ping·citations_delta는 보내지 않는다."""
    text_blocks, started = set(), False
    st.setdefault('deferred', [])
    async for name, d in events:
        t = d.get('type') or name
        if t == 'ping':
            continue
        if t == 'message_delta':
            st['message_delta'] = d
            if d.get('usage'):
                st['usage'] = {**st.get('usage', {}), **{k: v for k, v in d['usage'].items() if v is not None}}
            continue
        if t == 'message_stop':
            break
        if t == 'error':
            st['error'] = True
            yield d
            break
        if t == 'content_block_start' and (d.get('content_block') or {}).get('type') == 'text':
            text_blocks.add(d.get('index'))
            continue
        if t in ('content_block_delta', 'content_block_stop') and d.get('index') in text_blocks:
            if t == 'content_block_delta' and (d.get('delta') or {}).get('type') == 'text_delta':
                txt = d['delta'].get('text') or ''
                if txt:
                    started = True
                    yield txt
            continue
        if started:                       # 본문 뒤에 온 블록(tool_use 등): 본문을 다 보낸 뒤에
            st['deferred'].append(d)
        else:
            yield d


class Out:
    """게이트웨이가 내보내는 Anthropic SSE. 블록 index를 보내는 순서대로 다시 매긴다."""

    def __init__(self):
        self.next, self.map, self.text_idx, self.text_closed = 0, {}, None, False

    def raw(self, d):
        t = d.get('type')
        if t == 'content_block_start':
            self.map[d.get('index')] = self.next
            d = {**d, 'index': self.next}
            self.next += 1
        elif t in ('content_block_delta', 'content_block_stop'):
            d = {**d, 'index': self.map.get(d.get('index'), d.get('index'))}
        return ev(t, d)

    def text(self, s):
        out = ''
        if self.text_idx is None:
            self.text_idx, self.next = self.next, self.next + 1
            out += ev('content_block_start', {'type': 'content_block_start', 'index': self.text_idx,
                                              'content_block': {'type': 'text', 'text': ''}})
        return out + ev('content_block_delta', {'type': 'content_block_delta', 'index': self.text_idx,
                                                'delta': {'type': 'text_delta', 'text': s}})

    def close_text(self):
        if self.text_idx is None or self.text_closed:
            return ''
        self.text_closed = True
        return ev('content_block_stop', {'type': 'content_block_stop', 'index': self.text_idx})

    @staticmethod
    def verification(info):
        return ev('verification', {'type': 'verification', **info})
