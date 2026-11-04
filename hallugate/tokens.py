"""재판정 간격을 세는 토큰 단위 (SPEC §5.1).

판정기는 tiktoken cl100k_base 토큰으로 '문장 시작부터 s토큰마다' 판정 지점을 잡아 학습·보정됐다(SSOT §4).
게이트웨이도 같은 단위로 세야 학습·임계값과 맞는다. 업스트림 조각 수로 세면 Claude(조각당 약 5토큰)에서 어긋난다.

  offsets(text) -> [(start, end), ...]  각 토큰의 글자 구간. 이어 붙이면 text 전체를 덮는다.

'words'는 네트워크 없이 테스트하기 위한 대체 단위다(앞 공백 + 단어 = 토큰 1개, cl100k와 비슷한 모양).
"""
import re


class WordTokens:
    name = 'words'
    _pat = re.compile(r'\s*\S+|\s+')

    def offsets(self, text):
        return [m.span() for m in self._pat.finditer(text)]


class TiktokenTokens:
    def __init__(self, encoding='cl100k_base'):
        import tiktoken
        self.name = encoding
        self.enc = tiktoken.get_encoding(encoding)

    def offsets(self, text):
        ids = self.enc.encode(text, disallowed_special=())
        dec, starts = self.enc.decode_with_offsets(ids)
        if dec != text:                       # 드문 경우: 멀티바이트 문자 경계 — 누적 디코드로 대체 (E1과 같은 처리)
            starts, pos = [], 0
            for i in range(len(ids)):
                starts.append(pos)
                pos = len(self.enc.decode(ids[:i + 1]))
        ends = starts[1:] + [len(text)]
        return [(a, b) for a, b in zip(starts, ends) if b > a]


def make_tokens(name):
    return WordTokens() if name == 'words' else TiktokenTokens(name)
