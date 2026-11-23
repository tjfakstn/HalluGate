"""운영 모드 API 키 저장소 (SPEC §10.1).

키 형식: hg_<키 ID>_<비밀>. 저장소에는 비밀의 SHA-256 해시만 둔다(원문은 발급할 때 한 번만 보여 줌).
키 ID는 사용자(테넌트) 식별자로도 쓴다 — 컬렉션 이름 앞에 붙여 사용자별로 분리한다(SPEC §10.2).
"""
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field

SCHEMA = '''
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY, hash TEXT NOT NULL, name TEXT, created REAL NOT NULL, revoked REAL, limits TEXT);
'''


@dataclass
class Tenant:
    key_id: str
    name: str = ''
    limits: dict = field(default_factory=dict)       # 키별 한도 (SPEC §10.4). 비어 있으면 서버 기본값


def _hash(secret):
    return hashlib.sha256(secret.encode()).hexdigest()


def parse_key(full):
    """'hg_<id>_<비밀>' → (id, 비밀). 형식이 아니면 None."""
    if not isinstance(full, str) or not full.startswith('hg_'):
        return None
    parts = full.split('_', 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


class KeyStore:
    def __init__(self, path):
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def create(self, name='', limits=None):
        """새 키 발급. (키 ID, 전체 키 문자열) — 전체 키는 이때만 알 수 있다."""
        kid, secret = secrets.token_hex(4), secrets.token_urlsafe(24).replace('_', 'x')
        with self._lock, self._db:
            while self._db.execute('SELECT 1 FROM api_keys WHERE key_id=?', (kid,)).fetchone():
                kid = secrets.token_hex(4)
            self._db.execute('INSERT INTO api_keys VALUES (?,?,?,?,?,?)',
                             (kid, _hash(secret), name, time.time(), None, json.dumps(limits or {})))
        return kid, f'hg_{kid}_{secret}'

    def verify(self, full):
        """유효한 키면 Tenant, 아니면 None (없음·형식 오류·비밀 불일치·폐기)."""
        p = parse_key(full)
        if p is None:
            return None
        kid, secret = p
        with self._lock:
            row = self._db.execute('SELECT hash, name, revoked, limits FROM api_keys WHERE key_id=?', (kid,)).fetchone()
        if row is None or row[2] is not None or not hmac.compare_digest(row[0], _hash(secret)):
            return None
        return Tenant(kid, row[1] or '', json.loads(row[3] or '{}'))

    def list(self):
        with self._lock:
            rows = self._db.execute('SELECT key_id, name, created, revoked, limits FROM api_keys ORDER BY created').fetchall()
        return [{'key_id': r[0], 'name': r[1], 'created': r[2], 'revoked': r[3], 'limits': json.loads(r[4] or '{}')}
                for r in rows]

    def revoke(self, key_id):
        with self._lock, self._db:
            n = self._db.execute('UPDATE api_keys SET revoked=? WHERE key_id=? AND revoked IS NULL',
                                 (time.time(), key_id)).rowcount
        return n > 0

    def close(self):
        self._db.close()
