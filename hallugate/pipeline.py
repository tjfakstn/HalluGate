"""스트리밍 검증 파이프라인 (게이트웨이의 핵심).

입력: 실제 LLM이 흘려보내는 텍스트 조각(delta)
출력: 사용자에게 보낼 이벤트 — text(본문), verdict(판정 결과), cut(응답 중단)

정책 (SPEC §5)
  pass : 본문을 받는 즉시 내보내고, 판정은 백그라운드로 실행해 결과를 뒤따라 보낸다.
         추가 지연은 0이지만, 판정이 끝나기 전에 틀린 내용이 사용자에게 이미 나간다(누출).
  hold : 판정이 끝날 때까지 본문을 붙잡았다가 내보낸다. 누출은 0이지만 사용자는 기다려야 한다.
  action=cut 이면 환각 판정 시 그 자리에서 응답을 끊는다.

판정 시점 (SPEC §5.1)
  stride=0 : 문장이 완성되면 그 문장을 판정한다.
  stride=s : 위에 더해, 현재 문장에 조각이 s개 쌓일 때마다 '지금까지의 문장 앞부분'을 판정한다.
             pass는 문장 도중에도 경보를 낼 수 있고, hold는 안전 판정된 앞부분까지 먼저 내보낸다.

측정 (RequestRecord)
  sentences : 완성 문장 판정 — 완성 시각, 판정 시각, 전달 시각, 점수, 판정 시간
  checks    : 앞부분 판정 — 판정 구간, 시각, 점수, 판정 시점까지 사용자에게 나간 글자 수
"""
import asyncio
import time
from dataclasses import asdict, dataclass, field

from .splitter import IncrementalSplitter


@dataclass
class Event:
    kind: str                 # 'text' | 'verdict' | 'cut'
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
    sentences: list = field(default_factory=list)
    checks: list = field(default_factory=list)

    def now(self):
        return (time.perf_counter() - self.t0) * 1000

    def to_dict(self):
        d = asdict(self)
        d.pop('t0')
        return d


class VerifyPipeline:
    def __init__(self, judge, ref_index, mode, action, record, stride=0):
        assert mode in ('pass', 'hold')
        self.judge, self.ref, self.mode, self.action, self.rec = judge, ref_index, mode, action, record
        self.stride = stride or 0
        self.splitter = IncrementalSplitter()
        self.text = ''            # 지금까지 받은 전체 본문
        self.released_upto = 0    # hold: 사용자에게 내보낸 위치
        self.since = 0            # 현재 문장에 쌓인 조각 수

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

    async def _judge_sentence(self, s, srec):
        if not s.clean:                                   # 공백뿐인 꼬리
            srec.update(skipped=True, t_verdict=self.rec.now(), hallucinated=False)
            return None
        v = await asyncio.to_thread(self.judge.judge, self.ref, s.clean)
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
        v = await asyncio.to_thread(self.judge.judge, self.ref, clean)
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

    def _prefix_due(self, d, sents):
        """이번 조각으로 앞부분 재판정 시점이 됐는지. 문장이 끝났으면 새 문장 기준으로 다시 센다."""
        if d is None:
            return False
        if sents:
            _, buf = self.splitter.pending()
            self.since = 1 if buf.strip() else 0
        else:
            self.since += 1
        return bool(self.stride) and self.since > 0 and self.since % self.stride == 0

    def _cut(self, info):
        self.rec.cut = True
        return Event('cut', info={'sentence_idx': info['sentence_idx'], 'kind': info['kind']})

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
                    continue
                d, get = get.result(), None
                if d:
                    self.text += d
                sents = self.splitter.flush() if d is None else self.splitter.feed(d)
                due = self._prefix_due(d, sents)

                if self.mode == 'pass':
                    if d:
                        yield self._release_text(d)
                    for s in sents:
                        srec = self._new_srec(s)
                        srec['t_release'] = srec['t_complete']   # 이미 나갔음
                        pending.append(asyncio.create_task(self._judge_sentence(s, srec)))
                    if due:
                        off, buf = self.splitter.pending()
                        pending.append(asyncio.create_task(self._judge_prefix(self.splitter._n, off, buf)))
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
                    if due:
                        off, buf = self.splitter.pending()
                        info = await self._judge_prefix(self.splitter._n, off, buf)
                        if info is not None:
                            yield self._verdict(info)
                            if info['hallucinated'] and self.action == 'cut':
                                yield self._cut(info)
                                return
                            # 안전하거나(flag 모드에서는 표시 후) → 판정한 앞부분까지 내보낸다
                            yield self._release_to(off + len(buf))

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
