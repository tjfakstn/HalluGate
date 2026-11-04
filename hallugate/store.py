"""컬렉션 저장소 (SPEC §2.5, §2.6).

관리자·개발자가 등록한 문서를 조각으로 나눠 임베딩과 함께 저장하고, 질문으로 검색한다.
게이트웨이는 찾은 조각을 LLM 프롬프트에 넣고, **같은 조각을 판정 근거로** 쓴다(LLM이 본 근거 = 판정 근거).

  Store 인터페이스
    add(collection, text, title='', doc_id=None) -> dict(doc_id, title, n_chunks, n_chars)
    list(collection) -> [dict(doc_id, title, n_chunks, n_chars, created)]
    delete(collection, doc_id) -> bool
    collections() -> [dict(collection, n_docs)]
    search(collection, query, k) -> [Evidence]       # 점수 내림차순

처음 구현은 SQLite 파일 하나 + 임베딩 배열(코사인 유사도 전수 검색)이다. 규모가 커지면 같은 인터페이스로 벡터 DB로 바꾼다.
임베더는 encode(texts) -> 정규화된 벡터 배열을 돌려주는 객체(sentence-transformers 호환)면 된다.
"""
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass

import numpy as np

SCHEMA = '''
CREATE TABLE IF NOT EXISTS documents (
    collection TEXT NOT NULL, doc_id TEXT NOT NULL, title TEXT, text TEXT NOT NULL, created REAL NOT NULL,
    PRIMARY KEY (collection, doc_id));
CREATE TABLE IF NOT EXISTS chunks (
    collection TEXT NOT NULL, doc_id TEXT NOT NULL, idx INTEGER NOT NULL, text TEXT NOT NULL, emb BLOB NOT NULL,
    PRIMARY KEY (collection, doc_id, idx));
'''


@dataclass
class Evidence:
    doc_id: str
    title: str
    chunk_idx: int
    text: str
    score: float

    def to_dict(self):
        return asdict(self)


def split_words(text, n, overlap=0):
    """n단어 조각(겹침 overlap단어). 빈 문서는 조각 없음."""
    w = text.split()
    if not w:
        return []
    step = max(1, n - overlap)
    return [' '.join(w[i:i + n]) for i in range(0, max(len(w) - overlap, 1), step)]


class SQLiteStore:
    def __init__(self, path, embedder, chunk_words=120, overlap=20):
        self.path, self.embedder = path, embedder
        self.chunk_words, self.overlap = chunk_words, overlap
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._cache = {}          # collection -> (meta 리스트, 임베딩 행렬)

    def _embed(self, texts):
        v = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32).reshape(len(texts), -1)

    def add(self, collection, text, title='', doc_id=None):
        if not text or not text.strip():
            raise ValueError('document text is empty')
        doc_id = doc_id or uuid.uuid4().hex[:16]
        chunks = split_words(text, self.chunk_words, self.overlap)
        emb = self._embed(chunks)
        with self._lock, self._db:
            self._db.execute('DELETE FROM chunks WHERE collection=? AND doc_id=?', (collection, doc_id))   # 같은 id면 교체
            self._db.execute('INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?)',
                             (collection, doc_id, title or '', text, time.time()))
            self._db.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)',
                                 [(collection, doc_id, i, c, e.tobytes()) for i, (c, e) in enumerate(zip(chunks, emb))])
            self._cache.pop(collection, None)
        return {'doc_id': doc_id, 'title': title or '', 'n_chunks': len(chunks), 'n_chars': len(text)}

    def list(self, collection):
        with self._lock:
            rows = self._db.execute(
                'SELECT d.doc_id, d.title, length(d.text), d.created, '
                '(SELECT count(*) FROM chunks c WHERE c.collection=d.collection AND c.doc_id=d.doc_id) '
                'FROM documents d WHERE d.collection=? ORDER BY d.created', (collection,)).fetchall()
        return [{'doc_id': r[0], 'title': r[1], 'n_chars': r[2], 'created': r[3], 'n_chunks': r[4]} for r in rows]

    def delete(self, collection, doc_id):
        with self._lock, self._db:
            n = self._db.execute('DELETE FROM documents WHERE collection=? AND doc_id=?', (collection, doc_id)).rowcount
            self._db.execute('DELETE FROM chunks WHERE collection=? AND doc_id=?', (collection, doc_id))
            self._cache.pop(collection, None)
        return n > 0

    def collections(self):
        with self._lock:
            rows = self._db.execute('SELECT collection, count(*) FROM documents GROUP BY collection ORDER BY collection').fetchall()
        return [{'collection': c, 'n_docs': n} for c, n in rows]

    def exists(self, collection):
        with self._lock:
            return self._db.execute('SELECT 1 FROM documents WHERE collection=? LIMIT 1', (collection,)).fetchone() is not None

    def _matrix(self, collection):
        with self._lock:
            if collection not in self._cache:
                rows = self._db.execute(
                    'SELECT c.doc_id, d.title, c.idx, c.text, c.emb FROM chunks c JOIN documents d '
                    'ON c.collection=d.collection AND c.doc_id=d.doc_id WHERE c.collection=? ORDER BY d.created, c.idx',
                    (collection,)).fetchall()
                meta = [r[:4] for r in rows]
                mat = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), np.float32)
                self._cache[collection] = (meta, mat)
            return self._cache[collection]

    def search(self, collection, query, k):
        meta, mat = self._matrix(collection)
        if not meta or not query.strip():
            return []
        sims = mat @ self._embed([query])[0]
        top = np.argsort(-sims)[:k]
        return [Evidence(meta[i][0], meta[i][1], meta[i][2], meta[i][3], float(sims[i])) for i in top]

    def close(self):
        self._db.close()
