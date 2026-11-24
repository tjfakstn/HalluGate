"""관리형 베타: 쓸 때만 컨테이너 켜기 (SPEC §10.7).

평소에는 최소 0대(요청이 없으면 꺼짐, A10G 콜드 스타트 약 20초). 시연처럼 첫 요청도 바로 응답해야 할 때 1대를 켜 두고, 끝나면 끈다.
재배포 없이 Modal 자동 확장 설정만 바꾸므로 몇 초면 된다. 끄는 것을 잊어도 배포의 auto_cooldown이
끌 시각이 지나면 10분 안에 끈다.

  python scripts/warm.py up [--hours 2] [--wait --url https://<게이트웨이>]
  python scripts/warm.py status
  python scripts/warm.py down
"""
import argparse
import datetime
import time

APP, FUNCTION, DICT = 'hallugate', 'web', 'hallugate-warm'
MAX_HOURS = 12

BACKEND = None            # 테스트에서 바꿔 끼우는 Modal 연결
NOW = time.time


class ModalBackend:
    """web 함수의 최소 컨테이너 수와 끌 시각(Modal Dict)을 다룬다."""
    def __init__(self):
        import modal
        self.fn = modal.Function.from_name(APP, FUNCTION)
        self.d = modal.Dict.from_name(DICT, create_if_missing=True)

    def set_min(self, n):
        self.fn.autoscaler.update(min_containers=n)

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d.put(k, v)

    def pop(self, k):
        try:
            self.d.pop(k)
        except KeyError:
            pass

    def ping(self, url, limit=1200):
        import httpx
        t0 = time.time()
        while time.time() - t0 < limit:
            try:
                r = httpx.get(url.rstrip('/') + '/health', timeout=200, follow_redirects=True)
                if r.status_code == 200:
                    print(f'준비됨: {time.time() - t0:.0f}초 걸림')
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(5)
        print(f'{limit}초 안에 응답이 없습니다. modal app logs hallugate로 확인하세요')
        return False


def cooldown(b, now):
    """끌 시각이 지났으면 최소 0으로 되돌린다. 바꿨으면 True (배포의 auto_cooldown이 10분마다 부름)."""
    until = b.get('until')
    if until is None or now < until:
        return False
    b.set_min(0)
    b.pop('until')
    return True


def _hhmm(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')


def _hours(x):
    h = float(x)
    if not 0 < h <= MAX_HOURS:
        raise argparse.ArgumentTypeError(f'0보다 크고 {MAX_HOURS} 이하')
    return h


def main(argv=None):
    ap = argparse.ArgumentParser(description='관리형 베타 컨테이너 켜기·끄기 (SPEC §10.7)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    up = sub.add_parser('up', help='최소 1대로 켜고 끌 시각을 적는다')
    up.add_argument('--hours', type=_hours, default=2.0, help=f'몇 시간 뒤 자동으로 끌지 (기본 2, 최대 {MAX_HOURS})')
    up.add_argument('--wait', action='store_true', help='/health가 응답할 때까지 기다림')
    up.add_argument('--url', default='', help='게이트웨이 주소 (--wait에 필요)')
    sub.add_parser('down', help='최소 0대로 되돌린다')
    sub.add_parser('status', help='끌 시각')
    a = ap.parse_args(argv)
    if a.cmd == 'up' and a.wait and not a.url:
        ap.error('--wait에는 --url이 필요합니다')

    b = BACKEND or ModalBackend()
    now = NOW()
    if a.cmd == 'up':
        until = now + a.hours * 3600
        b.set_min(1)
        b.put('until', until)
        print(f'켬: 최소 1대, {_hhmm(until)}에 자동으로 끕니다 (끝나면 python scripts/warm.py down)')
        if a.wait:
            b.ping(a.url)
    elif a.cmd == 'down':
        b.set_min(0)
        b.pop('until')
        print('끔: 최소 0대 (남은 컨테이너는 유휴 5분 뒤 내려갑니다)')
    else:
        until = b.get('until')
        if until is None:
            print('꺼져 있음 (최소 0대)')
        else:
            print(f'켜져 있음: {_hhmm(until)}에 자동으로 끔 (남은 {max(0, (until - now) / 60):.0f}분)')


if __name__ == '__main__':
    main()
