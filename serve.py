"""게이트웨이 실행.

  JUDGE_MODEL_DIR=student_vark UPSTREAM_BASE_URL=http://localhost:9000/v1 python serve.py --port 8000
  (판정기 없이 파이프라인만 시험: --lexical)
"""
import argparse

import uvicorn

from hallugate import Settings, create_app
from hallugate.judge import LexicalJudge

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=8000)
p.add_argument('--host', default='127.0.0.1')
p.add_argument('--lexical', action='store_true', help='모델 없이 어휘 기반 대체 판정기 사용 (시험용)')
a = p.parse_args()

uvicorn.run(create_app(Settings(), judge=LexicalJudge() if a.lexical else None),
            host=a.host, port=a.port, log_level='warning')
