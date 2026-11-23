"""운영 모드 API 키 관리 (SPEC §10.1). 셀프 가입이 없으므로 관리자만 발급·폐기한다.

배포된 게이트웨이(관리자 API, 관리자 토큰은 ADMIN_TOKEN 환경 변수):
  ADMIN_TOKEN=... python scripts/keys.py --url https://<게이트웨이> create --name alice [--limits '{"rpm": 60}']
  ADMIN_TOKEN=... python scripts/keys.py --url https://<게이트웨이> list
  ADMIN_TOKEN=... python scripts/keys.py --url https://<게이트웨이> revoke <키 ID> [--purge]

로컬 저장소 직접(셀프호스팅·테스트):
  python scripts/keys.py --db hallugate.sqlite3 [--collections collections.sqlite3] create|list|revoke ...

--purge: 그 사용자의 컬렉션(문서·조각)과 측정 기록을 모두 지운다 (SPEC §10.5).
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRANSPORT = None          # 테스트에서 바꿔 끼우는 httpx 전송 계층


def _ts(x):
    return datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M') if x else '-'


def _print_list(rows):
    for row in rows:
        lim = json.dumps(row.get('limits') or {}) if row.get('limits') else ''
        print(f"{row['key_id']}  {row['name'] or '-':20}  만든 시각 {_ts(row['created'])}  폐기 {_ts(row['revoked'])}  {lim}")


def _created(kid, name, key):
    print(f'키 ID {kid} ({name})\n{key}\n이 키는 다시 볼 수 없습니다. 사용자에게 안전하게 전달하세요.')


def remote(a, limits):
    import httpx
    tok = os.environ.get('ADMIN_TOKEN', '')
    if not tok:
        sys.exit('ADMIN_TOKEN 환경 변수가 필요합니다')
    with httpx.Client(base_url=a.url.rstrip('/'), headers={'Authorization': f'Bearer {tok}'}, timeout=60,
                      **({'transport': TRANSPORT} if TRANSPORT else {})) as c:
        if a.cmd == 'create':
            r = c.post('/admin/keys', json={'name': a.name, 'limits': limits}); r.raise_for_status()
            _created(r.json()['key_id'], a.name, r.json()['key'])
        elif a.cmd == 'list':
            r = c.get('/admin/keys'); r.raise_for_status(); _print_list(r.json()['data'])
        else:
            r = c.post(f'/admin/keys/{a.key_id}/revoke', json={'purge': a.purge}); r.raise_for_status()
            d = r.json()
            print('폐기함' if d['revoked'] else '이미 폐기됐거나 없는 키')
            if a.purge:
                print(f"컬렉션 문서 {d.get('purged_docs', 0)}개, 측정 기록 {d.get('purged_records', 0)}줄 삭제")


def local(a, limits):
    from hallugate.keys import KeyStore
    ks = KeyStore(a.db)
    if a.cmd == 'create':
        kid, key = ks.create(a.name, limits)
        _created(kid, a.name, key)
    elif a.cmd == 'list':
        _print_list(ks.list())
    else:
        print('폐기함' if ks.revoke(a.key_id) else '이미 폐기됐거나 없는 키')
        if a.purge:
            from hallugate.metrics import MetricsStore
            from hallugate.store import SQLiteStore
            st = SQLiteStore(a.collections, None)
            print(f'컬렉션 문서 {st.purge(a.key_id + "/")}개 삭제')
            st.close()
            print(f'측정 기록 {MetricsStore(a.db).purge(a.key_id)}줄 삭제')
    ks.close()


def main(argv=None):
    p = argparse.ArgumentParser(description='hallugate API 키 관리')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--url', help='배포된 게이트웨이 주소 (관리자 API 사용)')
    g.add_argument('--db', help='로컬 키 저장소 파일 (직접 수정)')
    p.add_argument('--collections', default=os.environ.get('COLLECTION_DB', 'collections.sqlite3'))
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('create'); c.add_argument('--name', default=''); c.add_argument('--limits', default='{}')
    sub.add_parser('list')
    r = sub.add_parser('revoke'); r.add_argument('key_id'); r.add_argument('--purge', action='store_true')
    a = p.parse_args(argv)
    limits = json.loads(a.limits) if a.cmd == 'create' else None
    if a.url:
        remote(a, limits)
    else:
        a.db = a.db or os.environ.get('KEYS_DB', 'hallugate.sqlite3')
        local(a, limits)


if __name__ == '__main__':
    main()
