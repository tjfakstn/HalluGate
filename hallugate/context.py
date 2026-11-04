"""요청에서 근거 문서(RAG 컨텍스트)와 검증 옵션을 꺼낸다.

우선순위
1. 최상위 필드 `verification.context` (문자열 또는 문자열 리스트)
   - OpenAI SDK에서는 `extra_body={"verification": {...}}` 로 보낸다.
2. 메시지 본문 안의 `<context>...</context>` 태그 (RAG 서비스가 흔히 쓰는 형식)
3. `verification.collection` — 게이트웨이에 등록된 컬렉션에서 검색 (app이 처리, SPEC §2.6)
4. 모두 없으면 검증하지 않는다 (mode=off).

`verification` 필드는 실제 LLM API가 모르는 필드이므로 전달 전에 제거한다.
"""
import copy
import re

CTX_TAG = re.compile(r'<context>(.*?)</context>', re.S | re.I)
MODES = {'pass', 'hold', 'off'}
ACTIONS = {'flag', 'cut'}


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):          # [{"type":"text","text":...}, ...]
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
    return ''


RAG_INSTRUCTION = ('Answer using only the reference documents below. '
                   'If they do not contain the answer, say that you do not know.')


def last_user_text(body):
    users = [m for m in body.get('messages', []) if m.get('role') == 'user']
    return _content_text(users[-1].get('content')) if users else ''


def rag_messages(messages, evidence):
    """근거 조각을 담은 시스템 메시지를 맨 앞에 더한다. 서비스가 보낸 메시지는 바꾸지 않는다 (SPEC §2.6)."""
    block = '\n\n'.join(f'[{i + 1}] {e.text}' for i, e in enumerate(evidence))
    return [{'role': 'system', 'content': f'{RAG_INSTRUCTION}\n\n<documents>\n{block}\n</documents>'}] + list(messages)


def rag_reference(evidence):
    """판정 근거 = LLM에 넣은 조각과 같은 텍스트."""
    return '\n\n'.join(e.text for e in evidence)


def extract(body, default_mode='pass', default_action='flag', default_stride=0, with_stride=False, with_opts=False):
    """반환: (upstream으로 보낼 body, reference 문자열 또는 None, mode, action[, stride][, opts])
    with_stride=True면 앞부분 재판정 간격(stride)도 함께 돌려준다.
    with_opts=True면 원래 verification 옵션도 돌려준다(collection 등). collection이 있고 근거가 없으면 mode를 off로 바꾸지 않는다."""
    body = copy.deepcopy(body)
    opts = body.pop('verification', None) or {}

    ref = opts.get('context')
    if isinstance(ref, list):
        ref = '\n\n'.join(str(x) for x in ref)
    if not ref:
        found = []
        for m in body.get('messages', []):
            found += CTX_TAG.findall(_content_text(m.get('content')))
        ref = '\n\n'.join(s.strip() for s in found) or None

    mode = opts.get('mode', default_mode)
    action = opts.get('action', default_action)
    if mode not in MODES:
        raise ValueError(f'verification.mode must be one of {sorted(MODES)}')
    if action not in ACTIONS:
        raise ValueError(f'verification.action must be one of {sorted(ACTIONS)}')
    stride = opts.get('stride', default_stride)
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 0:
        raise ValueError('verification.stride must be a non-negative integer')
    coll = opts.get('collection')
    if coll is not None and (not isinstance(coll, str) or not coll):
        raise ValueError('verification.collection must be a non-empty string')
    if not ref and not coll:
        mode = 'off'
    out = (body, ref, mode, action) + ((stride,) if with_stride else ()) + ((opts,) if with_opts else ())
    return out
