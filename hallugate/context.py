"""요청에서 근거 문서(RAG 컨텍스트)와 검증 옵션을 꺼낸다.

우선순위
1. 최상위 필드 `verification.context` (문자열 또는 문자열 리스트)
   - OpenAI SDK에서는 `extra_body={"verification": {...}}` 로 보낸다.
2. 메시지 본문 안의 `<context>...</context>` 태그 (RAG 서비스가 흔히 쓰는 형식)
3. 둘 다 없으면 검증하지 않는다 (mode=off).

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


def extract(body, default_mode='pass', default_action='flag'):
    """반환: (upstream으로 보낼 body, reference 문자열 또는 None, mode, action)"""
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
    if not ref:
        mode = 'off'
    return body, ref, mode, action
