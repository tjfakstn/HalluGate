"""스트리밍 검증 파이프라인 (게이트웨이의 핵심).

입력: 실제 LLM이 흘려보내는 텍스트 조각(delta)
출력: 사용자에게 보낼 이벤트 — text(본문), verdict(문장 판정 결과), cut(응답 중단)

정책
  pass : 본문을 받는 즉시 내보내고, 문장이 완성되면 백그라운드로 판정해 결과를 뒤따라 보낸다.
         추가 지연은 0이지만, 판정이 끝나기 전에 틀린 문장이 사용자에게 이미 나간다(누출).
  hold : 문장이 완성될 때까지 붙잡고, 판정한 뒤에 그 문장을 내보낸다.
         누출은 0이지만, 문장이 생성되는 동안 사용자는 기다려야 한다.
  action=cut 이면 환각 판정 시 그 자리에서 응답을 끊는다.

측정 (RequestRecord)
  문장마다: 완성 시각, 판정 완료 시각, 사용자에게 내보낸 시각, 판정 소요 시간, 점수
  누출량: 환각 문장의 판정이 나온 순간까지 그 문장 시작부터 사용자에게 이미 나간 문자 수
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

    def now(self):
        return (time.perf_counter() - self.t0) * 1000

    def to_dict(self):
        d = asdict(self)
        d.pop('t0')
        return d


class VerifyPipeline:
    def __init__(self, judge, ref_index, mode, action, record):
        assert mode in ('pass', 'hold')
        self.judge, self.ref, self.mode, self.action, self.rec = judge, ref_index, mode, action, record
        self.splitter = IncrementalSplitter()

    # ---- 내부 도우미 ----
    def _release(self, text):
        if text:
            if self.rec.t_client_first is None:
                self.rec.t_client_first = self.rec.now()
            self.rec.released_chars += len(text)
        return Event('text', text)

    async def _judge(self, s, srec):
        if not s.clean:                                   # 공백뿐인 꼬리
            srec.update(skipped=True, t_verdict=self.rec.now(), hallucinated=False)
            return None
        v = await asyncio.to_thread(self.judge.judge, self.ref, s.clean)
        srec.update(score=v.score, hallucinated=v.hallucinated, judge_ms=v.judge_ms,
                    t_verdict=self.rec.now(),
                    leak_chars=max(0, self.rec.released_chars - s.start) if v.hallucinated else 0)
        return v

    def _verdict_event(self, s, srec):
        info = {k: srec[k] for k in ('sentence_idx', 'start', 'score', 'hallucinated', 'judge_ms') if k in srec}
        info['text'] = s.clean
        return Event('verdict', info=info)

    def _new_srec(self, s):
        srec = {'sentence_idx': s.idx, 'start': s.start, 'n_chars': len(s.text), 't_complete': self.rec.now()}
        self.rec.sentences.append(srec)
        return srec

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
        pending = []                                     # pass 모드: (문장, 기록, 판정 task)
        get = None
        try:
            while True:
                if get is None:
                    get = asyncio.create_task(q.get())
                # pass 모드: 새 토큰 도착 또는 가장 앞선 판정 완료 중 먼저 일어나는 쪽을 처리
                waits = {get} | ({pending[0][2]} if pending else set())
                done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)

                while pending and pending[0][2].done():      # 끝난 판정은 문장 순서대로 즉시 전달
                    s, srec, t = pending.pop(0)
                    v = t.result()
                    if v is None:
                        continue
                    yield self._verdict_event(s, srec)
                    if v.hallucinated and self.action == 'cut':
                        self.rec.cut = True
                        yield Event('cut', info={'sentence_idx': s.idx})
                        return
                if get not in done:
                    continue
                d, get = get.result(), None
                sents = self.splitter.flush() if d is None else self.splitter.feed(d)

                if self.mode == 'pass':
                    if d:
                        yield self._release(d)
                    for s in sents:
                        srec = self._new_srec(s)
                        srec['t_release'] = srec['t_complete']   # 이미 나갔음
                        pending.append((s, srec, asyncio.create_task(self._judge(s, srec))))
                    if d is None:                                 # 스트림 종료: 남은 판정을 순서대로 기다림
                        for s, srec, t in pending:
                            v = await t
                            if v is None:
                                continue
                            yield self._verdict_event(s, srec)
                            if v.hallucinated and self.action == 'cut':
                                self.rec.cut = True
                                yield Event('cut', info={'sentence_idx': s.idx})
                                return
                        pending = []
                else:  # hold
                    for s in sents:
                        srec = self._new_srec(s)
                        v = await self._judge(s, srec)
                        if v is not None:
                            yield self._verdict_event(s, srec)
                            if v.hallucinated and self.action == 'cut':
                                self.rec.cut = True
                                yield Event('cut', info={'sentence_idx': s.idx})
                                return
                        srec['t_release'] = self.rec.now()
                        yield self._release(s.text)

                if d is None:
                    break
        finally:
            self.rec.t_client_done = self.rec.now()
            for _, _, t in pending:
                t.cancel()
            if get is not None:
                get.cancel()
            prod.cancel()
            try:
                await prod
            except (asyncio.CancelledError, Exception):
                pass
