"""운영 모드 측정 기록·통계 (SPEC §10.5·10.6).

요청마다 **원문 없는** 요약 한 줄을 SQLite `requests` 표(KEYS_DB와 같은 파일)에 남기고 표준 출력으로도 낸다.
남기는 항목은 허용 목록(summarize)으로만 고른다 — 기록 객체에 새 필드가 생겨도 원문이 새지 않게.
컨테이너 시작(예열 시간)은 `starts` 표에 남겨 콜드 스타트를 본다.
"""
import json
import re
import sqlite3
import threading
import time

import numpy as np

from .prices import cost_usd

SCHEMA = '''
CREATE TABLE IF NOT EXISTS requests (ts REAL NOT NULL, key_id TEXT, status INTEGER, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
CREATE TABLE IF NOT EXISTS starts (ts REAL NOT NULL, warmup_ms REAL);
'''

# 2단 검증 실시간 단가 추정($/백만 토큰): 배치 단가(SSOT, M1)의 2배
PRICE_RT = {'claude-haiku-5-5': (0.10, 0.50), 'claude-sonnet-5-5': (2.0, 10.0), 'claude-opus-5-5': (4.0, 20.0)}

PC_KEYS = ('verified', 'reason', 'model', 'votes', 'valid_votes', 'score', 'threshold', 'flagged', 'check_ms',
           'in_tokens', 'out_tokens')


def _pct(xs, q):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def summarize(rec, status=200, stream=True, cold=False):
    """RequestRecord.to_dict() → 원문 없는 요약 (허용 목록)."""
    sents, checks = rec.get('sentences') or [], rec.get('checks') or []
    judged = [x for x in sents if 'score' in x] + list(checks)
    jms = [x.get('judge_ms') for x in judged if x.get('judge_ms') is not None]
    pc = rec.get('post_check')
    return {
        'key_id': rec.get('tenant', ''), 'upstream': rec.get('upstream', ''), 'status': status, 'stream': bool(stream),
        'mode': rec.get('mode'), 'action': rec.get('action'), 'stride': rec.get('stride'),
        'ttft_ms': rec.get('t_client_first'), 'total_ms': rec.get('t_client_done'),
        'upstream_first_ms': rec.get('t_upstream_first'), 'upstream_done_ms': rec.get('t_upstream_done'),
        'prepare_ms': rec.get('prepare_ms'), 'retrieve_ms': rec.get('retrieve_ms'), 'ref_chars': rec.get('ref_chars', 0),
        'released_chars': rec.get('released_chars'), 'n_sentences': len(sents), 'n_judgments': len(judged),
        'n_flagged': sum(1 for x in sents if x.get('hallucinated')), 'judge_ms_max': max(jms) if jms else None,
        'judge_ms_p95': _pct(jms, 95), 'verified': rec.get('verified'), 'unverified_reason': rec.get('unverified_reason') or '',
        'cut': rec.get('cut'), 'collection_used': bool(rec.get('collection')), 'n_evidence': rec.get('n_evidence', 0),
        'cold': bool(cold),
        'trim_chars_before': rec.get('trim_chars_before', 0), 'trim_chars_after': rec.get('trim_chars_after', 0),
        'trim_chunks_kept': rec.get('trim_chunks_kept', 0), 'trim_chunks_total': rec.get('trim_chunks_total', 0),
        'post_check': {k: pc[k] for k in PC_KEYS if k in pc} if isinstance(pc, dict) else None,
        **llm_usage(rec.get('model'), rec.get('usage')),
    }


def llm_usage(model, usage):
    """업스트림 usage → 토큰 수·추정 비용 (SPEC §10.10). 사용량이 없으면 None."""
    u = usage if isinstance(usage, dict) else {}
    tin, tout = u.get('prompt_tokens'), u.get('completion_tokens')
    cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
    return {'model': model or '', 'tokens_in': tin, 'tokens_out': tout, 'tokens_cached': cached,
            'llm_cost_usd': cost_usd(model, tin, tout)}


def _verify_totals(rows):
    """검증·환각 집계 (SPEC §10.11). rows는 200 응답, 검증을 켠(mode≠off) 것만 센다."""
    vr = [r for r in rows if r.get('mode') and r.get('mode') != 'off' and r.get('verified') is not None]
    judged = sum(r.get('n_sentences') or 0 for r in vr)
    flagged = sum(r.get('n_flagged') or 0 for r in vr)
    reasons = {}
    for r in vr:
        if not r.get('verified'):
            k = r.get('unverified_reason') or 'unknown'
            reasons[k] = reasons.get(k, 0) + 1
    return {'responses': len(vr), 'fully_verified': sum(1 for r in vr if r.get('verified')),
            'responses_flagged': sum(1 for r in vr if (r.get('n_flagged') or 0) > 0),
            'sentences_judged': judged, 'sentences_flagged': flagged,
            'flag_rate': (flagged / judged) if judged else None, 'unverified_by_reason': reasons}


def _trim_totals(rows):
    """근거 다이어트 절감 집계 (SPEC §10.11)."""
    tr = [r for r in rows if (r.get('trim_chars_before') or 0) > 0]
    before = sum(r['trim_chars_before'] for r in tr)
    after = sum(r.get('trim_chars_after') or 0 for r in tr)
    return {'requests': len(tr), 'chars_before': before, 'chars_after': after,
            'saved_ratio': (1 - after / before) if before else None}


def _llm_totals(rows):
    out = {'tokens_in': 0, 'tokens_out': 0, 'tokens_cached': 0, 'est_cost_usd': 0.0, 'by_model': {}}
    for r in rows:
        if r.get('tokens_in') is None:
            continue
        m = out['by_model'].setdefault(r.get('model') or '?', {'requests': 0, 'tokens_in': 0, 'tokens_out': 0, 'est_cost_usd': 0.0})
        for d in (out, m):
            d['tokens_in'] += r.get('tokens_in') or 0
            d['tokens_out'] += r.get('tokens_out') or 0
            d['est_cost_usd'] += r.get('llm_cost_usd') or 0.0
        out['tokens_cached'] += r.get('tokens_cached') or 0
        m['requests'] += 1
    out['est_cost_usd'] = round(out['est_cost_usd'], 6)
    for m in out['by_model'].values():
        m['est_cost_usd'] = round(m['est_cost_usd'], 6)
    return out


def rejected(key_id, status, upstream='', stream=False, cold=False):
    """기록 객체가 없는 요청(인증 실패·한도·잘못된 요청·업스트림 오류)의 요약."""
    return {'key_id': key_id or '', 'upstream': upstream, 'status': status, 'stream': bool(stream), 'cold': bool(cold)}


def parse_window(w):
    m = re.fullmatch(r'(\d+)([mhd])', w or '')
    if not m:
        raise ValueError('window must look like 30m, 24h or 7d')
    return int(m.group(1)) * {'m': 60, 'h': 3600, 'd': 86400}[m.group(2)]


class MetricsStore:
    def __init__(self, path):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def add(self, row, ts=None):
        ts = time.time() if ts is None else ts
        with self._lock, self._db:
            self._db.execute('INSERT INTO requests VALUES (?,?,?,?)',
                             (ts, row.get('key_id', ''), row.get('status'), json.dumps({**row, 'ts': ts})))

    def add_start(self, warmup_ms, ts=None):
        with self._lock, self._db:
            self._db.execute('INSERT INTO starts VALUES (?,?)', (time.time() if ts is None else ts, warmup_ms))

    def rows(self, since):
        with self._lock:
            return [json.loads(r[0]) for r in self._db.execute('SELECT data FROM requests WHERE ts>=? ORDER BY ts', (since,))]

    def starts(self, since):
        with self._lock:
            return [{'ts': r[0], 'warmup_ms': r[1]} for r in
                    self._db.execute('SELECT ts, warmup_ms FROM starts WHERE ts>=? ORDER BY ts', (since,))]

    def prune(self, days):
        """보관 기간이 지난 기록을 지운다 (SPEC §10.5). 지운 요청 수."""
        cut = time.time() - days * 86400
        with self._lock, self._db:
            n = self._db.execute('DELETE FROM requests WHERE ts<?', (cut,)).rowcount
            self._db.execute('DELETE FROM starts WHERE ts<?', (cut,))
        return n

    def usage(self, key_id, window_s):
        """사용자 본인 사용량 (GET /v1/usage, SPEC §10.10)."""
        rows = [r for r in self.rows(time.time() - window_s) if r.get('key_id') == key_id and r.get('status') == 200]
        return {'requests': len(rows), **_llm_totals(rows), 'verification': _verify_totals(rows), 'trim': _trim_totals(rows)}

    def purge(self, key_id):
        with self._lock, self._db:
            return self._db.execute('DELETE FROM requests WHERE key_id=?', (key_id,)).rowcount

    def stats(self, window_s, label=''):
        """SPEC §10.6 관리자 통계."""
        since = time.time() - window_s
        rows = self.rows(since)
        ok = [r for r in rows if r.get('status') == 200]
        verified_rows = [r for r in ok if r.get('verified') is not None and r.get('mode') != 'off']
        count = lambda key: {str(k): sum(1 for r in rows if str(r.get(key)) == str(k))   # noqa: E731
                             for k in sorted({r.get(key) for r in rows if r.get(key) is not None}, key=str)}
        pcs = [r['post_check'] for r in ok if r.get('post_check')]
        ran = [p for p in pcs if p.get('verified')]
        cost = sum(p.get('in_tokens', 0) / 1e6 * PRICE_RT.get(p.get('model'), (2.0, 10.0))[0] +
                   p.get('out_tokens', 0) / 1e6 * PRICE_RT.get(p.get('model'), (2.0, 10.0))[1] for p in ran)
        reasons = {}
        for r in verified_rows:
            if not r.get('verified'):
                reasons[r.get('unverified_reason') or 'unknown'] = reasons.get(r.get('unverified_reason') or 'unknown', 0) + 1
        starts = self.starts(since)
        return {
            'window': label, 'since': since,
            'requests': {'total': len(rows), 'by_status': count('status'),
                         'by_key': {k: v for k, v in count('key_id').items() if k},
                         'by_upstream': {k: v for k, v in count('upstream').items() if k}},
            'latency': {'ttft_ms': {'p50': _pct([r.get('ttft_ms') for r in ok], 50), 'p95': _pct([r.get('ttft_ms') for r in ok], 95)},
                        'total_ms': {'p50': _pct([r.get('total_ms') for r in ok], 50), 'p95': _pct([r.get('total_ms') for r in ok], 95)},
                        'judge_ms_p95': _pct([r.get('judge_ms_p95') for r in ok], 95)},
            'unverified': {'rate': (sum(reasons.values()) / len(verified_rows)) if verified_rows else None, 'by_reason': reasons},
            'post_check': {'requested': len(pcs), 'runs': len(ran), 'flagged': sum(1 for p in ran if p.get('flagged')),
                           'quota': sum(1 for p in pcs if p.get('reason') == 'quota'), 'est_cost_usd': round(cost, 6)},
            'rate_limited': sum(1 for r in rows if r.get('status') == 429),
            'llm': _llm_totals(ok),
            'verification': _verify_totals(ok),
            'trim': _trim_totals(ok),
            'cold_starts': [{**st, 'first_ttft_ms': next((r.get('ttft_ms') for r in ok if r.get('cold') and r['ts'] >= st['ts']), None)}
                            for st in starts],
        }
