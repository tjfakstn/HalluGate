"""요청 추적 (SPEC §10.12): 사용자가 `x-hallugate-trace: 1`로 켠 요청만 원문 상세를 남긴다.

측정 기록(metrics.py, 원문 없음)과 다른 표(`traces`)에 두고, 그 키 본인만 `/v1/traces`로 본다.
요청 헤더(사용자 LLM 키·게이트웨이 키)는 받지도 않는다 — 넘겨받는 것은 본문·근거·판정·요약뿐이다.
"""
import json
import sqlite3
import threading
import time
import uuid

SCHEMA = '''
CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, ts REAL NOT NULL, key_id TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS traces_key_ts ON traces (key_id, ts);
'''

LIST_KEYS = ('status', 'model', 'mode', 'stream', 'n_sentences', 'n_flagged', 'verified', 'preview')


def new_id():
    return 'tr_' + uuid.uuid4().hex[:20]


def _cut(v, limit, flag):
    if isinstance(v, str) and len(v) > limit:
        flag[0] = True
        return v[:limit]
    return v


def build(*, request_body, verification, reference, evidence, trim, response, sentences, summary, status,
          question='', error=None, post_check=None, max_chars=60000):
    """저장할 추적 한 건. 글자 수가 많은 값은 max_chars에서 자르고 truncated를 단다."""
    flag = [False]
    req = {k: v for k, v in (request_body or {}).items() if k not in ('messages', 'stream_options')}
    msgs = []
    for m in (request_body or {}).get('messages') or []:
        m = dict(m)
        if isinstance(m.get('content'), str):
            m['content'] = _cut(m['content'], max_chars, flag)
        msgs.append(m)
    req['messages'] = msgs
    ver = {k: _cut(v, max_chars, flag) for k, v in (verification or {}).items()} if isinstance(verification, dict) else None
    s = summary or {}
    d = {
        'status': status, 'stream': bool(s.get('stream')), 'mode': s.get('mode'), 'model': s.get('model') or req.get('model', ''),
        'request': req, 'verification': ver, 'reference': _cut(reference, max_chars, flag),
        'evidence': [{**e, 'text': _cut(e.get('text'), max_chars, flag)} for e in (evidence or [])],
        'trim': trim, 'response': _cut(response, max_chars, flag),
        'sentences': [{**x, 'text': _cut(x.get('text'), max_chars, flag)} for x in (sentences or [])],
        'post_check': post_check, 'summary': s,
        'n_sentences': len(sentences or []), 'n_flagged': sum(1 for x in sentences or [] if x.get('hallucinated')),
        'verified': s.get('verified'), 'preview': _strip_context(question or ''),
    }
    if error is not None:
        d['error'] = _cut(error, max_chars, flag)
    d['truncated'] = flag[0]
    return d


def _strip_context(q):
    """미리보기에서 <context>…</context> 근거 태그를 뺀다 (질문만 보이게)."""
    import re
    return re.sub(r'<context>.*?(</context>|$)', '', q, flags=re.S).strip()[:120]


class TraceStore:
    def __init__(self, path, max_per_key=200):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.max_per_key = max_per_key

    def add(self, key_id, data, tid=None, ts=None):
        tid, ts = tid or new_id(), time.time() if ts is None else ts
        with self._lock, self._db:
            self._db.execute('INSERT INTO traces VALUES (?,?,?,?)', (tid, ts, key_id, json.dumps(data, ensure_ascii=False)))
            self._db.execute('DELETE FROM traces WHERE key_id=? AND id NOT IN '
                             '(SELECT id FROM traces WHERE key_id=? ORDER BY ts DESC LIMIT ?)',
                             (key_id, key_id, self.max_per_key))
        return tid

    def list(self, key_id, limit):
        with self._lock:
            rows = self._db.execute('SELECT id, ts, data FROM traces WHERE key_id=? ORDER BY ts DESC LIMIT ?',
                                    (key_id, limit)).fetchall()
        out = []
        for tid, ts, data in rows:
            d = json.loads(data)
            out.append({'id': tid, 'ts': ts, **{k: d.get(k) for k in LIST_KEYS}})
        return out

    def get(self, key_id, tid):
        with self._lock:
            r = self._db.execute('SELECT ts, data FROM traces WHERE key_id=? AND id=?', (key_id, tid)).fetchone()
        return None if r is None else {'id': tid, 'ts': r[0], **json.loads(r[1])}

    def delete(self, key_id, tid=None):
        with self._lock, self._db:
            if tid is None:
                return self._db.execute('DELETE FROM traces WHERE key_id=?', (key_id,)).rowcount
            return self._db.execute('DELETE FROM traces WHERE key_id=? AND id=?', (key_id, tid)).rowcount

    def count(self, since=0):
        with self._lock:
            return self._db.execute('SELECT COUNT(*) FROM traces WHERE ts>=?', (since,)).fetchone()[0]

    def prune(self, days):
        cut = time.time() - days * 86400
        with self._lock, self._db:
            return self._db.execute('DELETE FROM traces WHERE ts<?', (cut,)).rowcount
