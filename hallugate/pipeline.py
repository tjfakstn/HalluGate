"""스트리밍 검증 파이프라인 (게이트웨이의 핵심).

입력: 실제 LLM이 흘려보내는 텍스트 조각(str) 또는 판정 없이 통과시킬 청크(dict, tool_calls 등)
출력: 사용자에게 보낼 이벤트 — text(본문), verdict(판정 결과), cut(응답 중단), raw(그대로 통과)

정책 (SPEC §5)
  pass : 본문을 받는 즉시 내보내고, 판정은 백그라운드로 실행해 결과를 뒤따라 보낸다.
         추가 지연은 0이지만, 판정이 끝나기 전에 틀린 내용이 사용자에게 이미 나간다(누출).
  hold : 판정이 끝날 때까지 본문을 붙잡았다가 내보낸다. 누출은 0이지만 사용자는 기다려야 한다.
  action=cut 이면 환각 판정 시 그 자리에서 응답을 끊는다.

판정 시점 (SPEC §5.1)
  stride=0 : 문장이 완성되면 그 문장을 판정한다.
  stride=s : 위에 더해, 현재 문장의 토큰(cl100k)이 s개 확정될 때마다 '지금까지의 문장 앞부분'을 판정한다.
             마지막 토큰은 다음 글자가 와야 확정되므로, 토큰 s개 다음의 글자가 도착한 시점에 판정한다.
             pass는 문장 도중에도 경보를 낼 수 있고, hold는 안전 판정된 앞부분까지 먼저 내보낸다.

검증 중단 (SPEC §4, fail-open)
  판정기 오류·시간 초과·영어가 아닌 텍스트 → 판정 청크 {"kind": "unverified", "verified": false, "reason": ...}를
  한 번 보내고, 그 요청의 나머지 본문은 판정 없이 그대로 내보낸다.

측정 (RequestRecord)
  sentences : 완성 문장 판정 — 완성 시각, 판정 시각, 전달 시각, 점수, 판정 시간
  checks    : 앞부분 판정 — 판정 구간, 시각, 점수, 판정 시점까지 사용자에게 나간 글자 수
"""
import asyncio
import time
from dataclasses import asdict, dataclass, field

from .splitter import IncrementalSplitter
from .tokens import WordTokens


@dataclass
class Event:
    kind: str                 # 'text' | 'verdict' | 'cut' | 'raw'
    text: str = ''
    info: dict = None


@dataclass
class RequestRecord:
    mode: str
    action: str
    stride: int = 0
    tag: str = ''                      # 요청 헤더 x-request-tag (측정용 식별자)
    t0: float = field(default_factory=time.perf_counter)
    prepare_ms: float = 0.0
    t_upstream_first: float = None     # 실제 LLM의 첫 토큰 도착 (ms, 요청 기준)
    t_client_first: float = None       # 사용자에게 첫 본문이 나간 시각
    t_upstream_done: float = None
    t_client_done: float = None
    released_chars: int = 0
    cut: bool = False
    verified: bool = True              # 검증을 끝까지 했는지 (fail-open이면 False)
    unverified_reason: str = ''
    threshold: float = None
    sentences: list = field(default_factory=list)
    checks: list = field(default_factory=list)

    def now(self):
        return (time.perf_counter() - self.t0) * 1000

    def to_dict(self):
        d = asdict(self)
        d.pop('t0')
        return d


def is_english(text, min_letters=3, min_ratio=0.8):
    """글자 중 라틴 문자 비율로 영어 여부를 본다. 판정기는 영어 RAGTruth로만 학습·평가됐다(SSOT §4)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < min_letters:
        return True
    return sum(c.isascii() for c in letters) / len(letters) >= min_ratio


class VerifyPipeline:
    def __init__(self, judge, ref_index, mode, action, record, stride=0, tokens=None, threshold=None,
                 timeout_ms=None):
        assert mode in ('pass', 'hold')
        self.judge, self.ref, self.mode, self.action, self.rec = judge, ref_index, mode, action, record
        self.stride = stride or 0
        self.tokens = tokens or WordTokens()
        self.threshold = threshold
        self.timeout = timeout_ms / 1000 if timeout_ms else None
        self.rec.threshold = threshold
        self.splitter = IncrementalSplitter()
        self.text = ''            # 지금까지 받은 전체 본문
        self.released_upto = 0    # hold: 사용자에게 내보낸 위치
        self.cur_sidx = 0         # 앞부분 재판정: 현재 문장 번호
        self.last_mult = 0        #   현재 문장에서 마지막으로 판정한 s의 배수
        self.disabled = None      # 검증 중단 사유
        self.notified = False
        if ref_index is None:     # 근거 준비 실패
            self._disable(record.unverified_reason or 'judge_error')

    # ---- 내부 도우미 ----
    def _release_text(self, text):
        if text:
            if self.rec.t_client_first is None:
                self.rec.t_client_first = self.rec.now()
            self.rec.released_chars += len(text)
        return Event('text', text)

    def _release_to(self, end):
        """hold: 본문을 end 위치까지 내보낸다."""
        piece = self.text[self.released_upto:end]
        self.released_upto = max(self.released_upto, end)
        return self._release_text(piece)

    def _disable(self, reason):
        if self.disabled is None:
            self.disabled = reason
            self.rec.verified, self.rec.unverified_reason = False, reason

    async def _score(self, text, prior):
        """판정 1회. 검증을 멈춰야 하면 None."""
        if self.disabled:
            return None
        if not is_english(text):
            self._disable('non_english')
            return None
        try:
            call = asyncio.to_thread(self.judge.judge, self.ref, text, prior, self.threshold)
            return await (asyncio.wait_for(call, self.timeout) if self.timeout else call)
        except asyncio.TimeoutError:
            self._disable('judge_timeout')
        except Exception:                                # noqa: BLE001 — 판정기 장애는 본문 전달을 막지 않는다
            self._disable('judge_error')
        return None

    async def _judge_sentence(self, s, srec):
        if not s.clean:                                   # 공백뿐인 꼬리
            srec.update(skipped=True, t_verdict=self.rec.now(), hallucinated=False)
            return None
        v = await self._score(s.clean, self.text[:s.start])
        if v is None:
            srec.update(skipped=True, t_verdict=self.rec.now(), hallucinated=False)
            return None
        srec.update(score=v.score, hallucinated=v.hallucinated, judge_ms=v.judge_ms,
                    t_verdict=self.rec.now(),
                    leak_chars=max(0, self.rec.released_chars - s.start) if v.hallucinated else 0)
        return {'kind': 'sentence', 'sentence_idx': s.idx, 'start': s.start, 'upto': s.start + len(s.text),
                'text': s.clean, 'score': v.score, 'hallucinated': v.hallucinated, 'judge_ms': v.judge_ms}

    async def _judge_prefix(self, sidx, start, raw):
        clean = raw.strip()
        if not clean:
            return None
        t_check = self.rec.now()
        v = await self._score(clean, self.text[:start])
        if v is None:
            return None
        info = {'kind': 'prefix', 'sentence_idx': sidx, 'start': start, 'upto': start + len(raw), 'text': clean,
                'score': v.score, 'hallucinated': v.hallucinated, 'judge_ms': v.judge_ms}
        self.rec.checks.append({k: info[k] for k in ('sentence_idx', 'start', 'upto', 'score', 'hallucinated', 'judge_ms')}
                               | {'t_check': t_check, 't_verdict': self.rec.now()})
        return info

    def _verdict(self, info):
        info = dict(info, released_at_verdict=self.rec.released_chars)
        if info['kind'] == 'prefix':
            self.rec.checks[-1]['released_at_verdict'] = info['released_at_verdict']
        return Event('verdict', info=info)

    def _new_srec(self, s):
        srec = {'sentence_idx': s.idx, 'start': s.start, 'n_chars': len(s.text), 't_complete': self.rec.now()}
        self.rec.sentences.append(srec)
        return srec

    def _prefix_point(self, d):
        """이번 조각으로 앞부분 재판정 시점이 됐으면 (문장 시작 위치, 판정할 앞부분 원문), 아니면 None.
        문장 시작부터 확정된 토큰 수가 s의 새 배수에 이르면 그 배수 번째 토큰 끝까지를 판정한다
        (한 조각이 여러 배수를 넘으면 마지막 배수 하나만)."""
        if not self.stride or d is None or self.disabled:
            return None
        off, buf = self.splitter.pending()
        if self.splitter._n != self.cur_sidx:
            self.cur_sidx, self.last_mult = self.splitter._n, 0
        toks = self.tokens.offsets(buf)
        m = (len(toks) - 1) // self.stride               # 마지막 토큰은 아직 이어질 수 있어 확정 전
        if m <= self.last_mult:
            return None
        self.last_mult = m
        return off, buf[:toks[m * self.stride - 1][1]]

    def _cut(self, info):
        self.rec.cut = True
        return Event('cut', info={'sentence_idx': info['sentence_idx'], 'kind': info['kind']})

    def _unverified_notice(self):
        """검증이 멈췄으면 한 번만 알린다."""
        if self.disabled and not self.notified:
            self.notified = True
            return Event('verdict', info={'kind': 'unverified', 'verified': False, 'reason': self.disabled,
                                          'sentence_idx': self.splitter._n, 'released_at_verdict': self.rec.released_chars})
        return None

    # ---- 본체 ----
    async def run(self, deltas):
        q = asyncio.Queue()

        async def producer():
            try:
                async for d in deltas:
                    if self.rec.t_upstream_first is None:
                        self.rec.t_upstream_first = self.rec.now()
                    await q.put(d)
            finally:
                self.rec.t_upstream_done = self.rec.now()
                await q.put(None)

        prod = asyncio.create_task(producer())
        pending = []                                     # pass 모드: 판정 task들 (발생 순서)
        get = None
        try:
            while True:
                if get is None:
                    get = asyncio.create_task(q.get())
                # pass: 새 조각 도착 또는 가장 앞선 판정 완료 중 먼저 일어나는 쪽을 처리
                waits = {get} | ({pending[0]} if pending else set())
                done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)

                while pending and pending[0].done():         # 끝난 판정은 순서대로 즉시 전달
                    info = pending.pop(0).result()
                    if info is None:
                        continue
                    yield self._verdict(info)
                    if info['hallucinated'] and self.action == 'cut':
                        yield self._cut(info)
                        return
                if get not in done:
                    if (n := self._unverified_notice()):
                        yield n
                    continue
                d, get = get.result(), None
                if isinstance(d, dict):                       # tool_calls 등: 판정 없이 그대로 통과
                    yield Event('raw', info=d)
                    continue
                if d:
                    self.text += d
                sents = self.splitter.flush() if d is None else self.splitter.feed(d)
                point = self._prefix_point(d)

                if self.mode == 'pass':
                    if d:
                        yield self._release_text(d)
                    for s in sents:
                        srec = self._new_srec(s)
                        srec['t_release'] = srec['t_complete']   # 이미 나갔음
                        pending.append(asyncio.create_task(self._judge_sentence(s, srec)))
                    if point:
                        pending.append(asyncio.create_task(self._judge_prefix(self.splitter._n, *point)))
                    if d is None:                                 # 스트림 종료: 남은 판정을 순서대로 기다림
                        for t in pending:
                            info = await t
                            if info is None:
                                continue
                            yield self._verdict(info)
                            if info['hallucinated'] and self.action == 'cut':
                                yield self._cut(info)
                                return
                        pending = []
                    if (n := self._unverified_notice()):
                        yield n
                else:  # hold
                    for s in sents:
                        srec = self._new_srec(s)
                        info = await self._judge_sentence(s, srec)
                        if info is not None:
                            yield self._verdict(info)
                            if info['hallucinated'] and self.action == 'cut':
                                yield self._cut(info)
                                return
                        srec['t_release'] = self.rec.now()
                        yield self._release_to(s.start + len(s.text))
                    if point:
                        info = await self._judge_prefix(self.splitter._n, *point)
                        if info is not None:
                            yield self._verdict(info)
                            if info['hallucinated'] and self.action == 'cut':
                                yield self._cut(info)
                                return
                            # 안전하거나(flag 모드에서는 표시 후) → 판정한 앞부분까지 내보낸다
                            yield self._release_to(point[0] + len(point[1]))
                    if self.disabled:                             # 검증 중단: 알리고 보류 중인 본문을 모두 내보낸다
                        if (n := self._unverified_notice()):
                            yield n
                        yield self._release_to(len(self.text))

                if d is None:
                    break
        finally:
            self.rec.t_client_done = self.rec.now()
            for t in pending:
                t.cancel()
            if get is not None:
                get.cancel()
            prod.cancel()
            try:
                await prod
            except (asyncio.CancelledError, Exception):
                pass
