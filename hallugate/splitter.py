"""스트리밍 텍스트를 문장 단위로 끊는다.

토큰이 조금씩 들어오므로, 버퍼에 쌓다가 '다음 문장이 시작된 것이 확인되면'
앞 문장을 완성된 것으로 보고 내보낸다. 마지막 문장은 스트림이 끝날 때 flush로 내보낸다.

내보낸 문장들을 이어 붙이면 원문과 정확히 같다(공백 포함). hold 모드에서 문장을
그대로 사용자에게 내보내야 하므로 이 성질이 중요하다.
"""
from dataclasses import dataclass

import pysbd


@dataclass
class Sentence:
    idx: int
    text: str        # 원문 그대로 (뒤 공백 포함)
    start: int       # 응답 전체에서의 시작 위치 (문자)

    @property
    def clean(self):
        return self.text.strip()


class IncrementalSplitter:
    def __init__(self, language='en'):
        self._seg = pysbd.Segmenter(language=language, clean=False, char_span=True)
        self._buf = ''
        self._offset = 0      # 버퍼 시작이 응답 전체에서 몇 번째 문자인지
        self._n = 0

    def _emit(self, text):
        s = Sentence(self._n, text, self._offset)
        self._n += 1
        self._offset += len(text)
        return s

    def feed(self, delta):
        """새 조각을 넣고, 완성된 문장 리스트를 돌려준다."""
        self._buf += delta
        if not self._buf.strip():
            return []
        starts = [sp.start for sp in self._seg.segment(self._buf) if sp.sent.strip()]
        if len(starts) < 2:
            return []
        out = []
        cuts = starts[1:]                 # 각 문장의 끝 = 다음 문장의 시작
        prev = 0
        for c in cuts:
            out.append(self._emit(self._buf[prev:c]))
            prev = c
        self._buf = self._buf[prev:]
        return out

    def pending(self):
        """아직 끝나지 않은 현재 문장: (응답 전체에서의 시작 위치, 지금까지의 텍스트)."""
        return self._offset, self._buf

    def flush(self):
        """스트림 종료 시 남은 텍스트를 마지막 문장으로 내보낸다 (공백뿐이면 판정은 건너뛰게 됨)."""
        if not self._buf:
            return []
        s = self._emit(self._buf)
        self._buf = ''
        return [s]
