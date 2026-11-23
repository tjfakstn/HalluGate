"""운영 모드 API 키 관리 (SPEC §10.1). 셀프 가입이 없으므로 관리자만 이 스크립트로 발급·폐기한다.

  python scripts/keys.py --db /data/hallugate.sqlite3 create --name alice
  python scripts/keys.py --db ... list
  python scripts/keys.py --db ... --collections /data/collections.sqlite3 revoke <키 ID> [--purge]

--purge: 그 사용자의 컬렉션(문서·조각)을 모두 지운다. 측정 기록 삭제는 SPEC §10.5(4단계)에서 함께 처리한다.
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallugate.keys import KeyStore       # noqa: E402


def _ts(x):
    return datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M') if x else '-'


def main(argv=None):
    p = argparse.ArgumentParser(description='hallugate API 키 관리')
    p.add_argument('--db', default=os.environ.get('KEYS_DB', '/data/hallugate.sqlite3'))
    p.add_argument('--collections', default=os.environ.get('COLLECTION_DB', 'collections.sqlite3'))
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('create'); c.add_argument('--name', default='')
    sub.add_parser('list')
    r = sub.add_parser('revoke'); r.add_argument('key_id'); r.add_argument('--purge', action='store_true')
    a = p.parse_args(argv)
    ks = KeyStore(a.db)
    if a.cmd == 'create':
        kid, key = ks.create(a.name)
        print(f'키 ID {kid} ({a.name})\n{key}\n이 키는 다시 볼 수 없습니다. 사용자에게 안전하게 전달하세요.')
    elif a.cmd == 'list':
        for row in ks.list():
            print(f"{row['key_id']}  {row['name'] or '-':20}  만든 시각 {_ts(row['created'])}  폐기 {_ts(row['revoked'])}")
    elif a.cmd == 'revoke':
        print('폐기함' if ks.revoke(a.key_id) else '이미 폐기됐거나 없는 키')
        if a.purge:
            from hallugate.store import SQLiteStore
            st = SQLiteStore(a.collections, None)
            print(f'컬렉션 문서 {st.purge(a.key_id + "/")}개 삭제')
            st.close()
    ks.close()


if __name__ == '__main__':
    main()
