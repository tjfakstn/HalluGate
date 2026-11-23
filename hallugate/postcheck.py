"""응답 끝 2단 검증 (SPEC §5.3, 🧪 실험 기능, 기본 꺼짐).

응답이 끝난 뒤 강한 검증기(Claude)가 응답 전체를 근거와 다시 대조한다. 1단(경량 판정기)이 놓친 환각을 잡기 위한 것이다.
지시문·파싱은 J2 노트북의 `sonnet_H` 후보와 같다(experiments/builders/build_j2.py SYS_SENT, 테스트로 일치 확인).
검증기·임계값은 J2가 채택되면 바꾼다. 본문은 바꾸지 않는다. 오류·시간 초과는 verified=False로 알린다(fail-open).
"""
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field

SYSTEM = '''You check whether a RESPONSE is fully supported by a REFERENCE document.
You receive the REFERENCE and the RESPONSE split into numbered sentences. Judge EVERY sentence against the REFERENCE only (not world knowledge).

A sentence contains a hallucination if any part of it is:
- Evident conflict: clearly contradicts the reference (wrong fact, number, name, date, negation).
- Subtle conflict: deviates from the reference in a way that changes meaning (altered term, scope, time, attribution, certainty).
- Evident baseless info: states information that is not in the reference and cannot be directly inferred from it, even if it may be true in the real world.
- Subtle baseless info: adds unsupported inference, opinion, or embellishment that the reference does not support.

Not hallucinations: faithful paraphrase; reasonable summarization or generalization of what the reference says; statements that the reference lacks the information.

Labels: "S" = fully supported. "H" = contains a hallucination. "A" = ambiguous, careful annotators could reasonably disagree.
For every H or A sentence also give "type" (evident_conflict, subtle_conflict, evident_baseless, subtle_baseless), "phrase" (shortest hallucinated span copied exactly) and "reason" (at most 20 words).

Return ONLY a JSON object, no other text:
{"sentences": [{"i": 1, "label": "S"}, {"i": 2, "label": "H", "type": "evident_baseless", "phrase": "...", "reason": "..."}]}
with exactly one entry per numbered sentence, in order.'''


def user_message(reference, sentences):
    lines = '\n'.join(f'[{i + 1}] {t}' for i, t in enumerate(sentences))
    return f"REFERENCE:\n<<<\n{reference}\n>>>\n\nRESPONSE ({len(sentences)} numbered sentences):\n{lines}"


def parse(text, n):
    """R4·J2와 같은 파싱: 문장마다 S/H/A. 모든 문장이 있어야 성공."""
    try:
        d = json.loads(text[text.find('{'): text.rfind('}') + 1])
        labs = {}
        for e in d.get('sentences', []):
            i, lab = int(e.get('i', 0)), str(e.get('label', '')).strip().upper()[:1]
            if 1 <= i <= n and lab and lab in 'SHA':
                labs[i] = {'label': lab, 'phrase': e.get('phrase', '') or '', 'reason': e.get('reason', '') or ''}
        return labs, len(labs) == n
    except Exception:                                        # noqa: BLE001
        return {}, False


@dataclass
class PostCheckResult:
    verified: bool
    reason: str = ''
    model: str = ''
    votes: int = 0
    valid_votes: int = 0
    score: float = 0.0
    threshold: float = 0.0
    flagged: bool = False
    check_ms: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    sentences: list = field(default_factory=list)

    def to_dict(self):
        if not self.verified:                                # 실패·건너뜀은 이유만
            return {'kind': 'post_check', 'verified': False, 'reason': self.reason}
        d = {'kind': 'post_check', **asdict(self)}
        d.pop('reason')
        return d


def skipped(reason):
    return PostCheckResult(False, reason)


class ClaudePostChecker:
    """Claude에 문장별 S/H/A 판정을 votes번 동시에 요청하고 표를 모은다.
    응답 점수 = H 문장이 하나라도 있다고 본 판정의 비율(파싱 실패 판정은 H 없음으로 셈)."""

    def __init__(self, client=None, api_key='', model='claude-sonnet-5-5', votes=3, threshold=0.5, max_tokens=2048):
        if client is None:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=2)
        self.client, self.model, self.votes, self.threshold, self.max_tokens = client, model, votes, threshold, max_tokens

    def params(self, reference, sentences):
        return {'model': self.model, 'max_tokens': self.max_tokens, 'system': SYSTEM,
                'messages': [{'role': 'user', 'content': user_message(reference, sentences)}]}

    async def _one(self, p):
        m = await self.client.messages.create(**p)
        u = getattr(m, 'usage', None)
        text = ''.join(getattr(b, 'text', '') for b in m.content if getattr(b, 'type', '') == 'text')
        return text, getattr(u, 'input_tokens', 0) or 0, getattr(u, 'output_tokens', 0) or 0

    async def check(self, reference, sents):
        """sents: [(sentence_idx, start, 문장 텍스트)] — 1단과 같은 문장 분할.
        모든 호출이 예외면 그 예외를 올린다(호출한 쪽이 fail-open으로 처리)."""
        t0 = time.perf_counter()
        p = self.params(reference, [t for _, _, t in sents])
        got = await asyncio.gather(*[self._one(p) for _ in range(self.votes)], return_exceptions=True)
        if all(isinstance(o, BaseException) for o in got):
            raise got[0]
        ok_calls = [o for o in got if not isinstance(o, BaseException)]
        tok_in, tok_out = sum(o[1] for o in ok_calls), sum(o[2] for o in ok_calls)   # 비용 추정용 (SPEC §10.6)
        outs = [o if isinstance(o, BaseException) else o[0] for o in got]
        n = len(sents)
        valid = [labs for labs, ok in (parse(o, n) if isinstance(o, str) else ({}, False) for o in outs) if ok]
        if not valid:
            return PostCheckResult(False, 'error')
        score = sum(any(v['label'] == 'H' for v in labs.values()) for labs in valid) / self.votes
        rows = []
        for k, (idx, start, _) in enumerate(sents, 1):
            hv = [labs[k] for labs in valid if labs[k]['label'] == 'H']
            row = {'sentence_idx': idx, 'start': start, 'h_votes': len(hv),
                   'flagged': len(hv) / len(valid) >= self.threshold}
            if hv:
                row.update(phrase=hv[0]['phrase'], reason=hv[0]['reason'])
            rows.append(row)
        return PostCheckResult(True, '', self.model, self.votes, len(valid), score, self.threshold,
                               score >= self.threshold, (time.perf_counter() - t0) * 1000, tok_in, tok_out, rows)
