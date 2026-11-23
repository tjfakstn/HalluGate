"""운영 모드 사용량 제한 (SPEC §10.4). 컨테이너 1대 전제라 메모리에서 센다.

  hit(key, rpm)        최근 60초 요청 수가 rpm 미만이면 기록하고 None, 아니면 다시 시도까지 남은 초(거절은 세지 않음)
  acquire(key, n)      처리 중인 요청이 n개 미만이면 하나 늘리고 True
  release(key)         처리 끝
"""
import math
import threading
import time
from collections import defaultdict, deque

WINDOW = 60.0


class Limiter:
    def __init__(self, now=time.monotonic):
        self.now = now
        self._hits = defaultdict(deque)
        self._active = defaultdict(int)
        self._lock = threading.Lock()

    def hit(self, key, rpm):
        t = self.now()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= t - WINDOW:
                q.popleft()
            if len(q) >= rpm:
                return max(1, math.ceil(q[0] + WINDOW - t))
            q.append(t)
            return None

    def acquire(self, key, limit):
        with self._lock:
            if self._active[key] >= limit:
                return False
            self._active[key] += 1
            return True

    def release(self, key):
        with self._lock:
            self._active[key] = max(0, self._active[key] - 1)
