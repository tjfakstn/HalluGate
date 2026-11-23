"""hallugate 관리형 베타 배포 — Modal 서버리스 GPU (SPEC §10.7, 운영 문서 docs/OPERATIONS.md).

게이트웨이 전체(FastAPI + 조합 판정기)를 GPU 컨테이너 하나에 올린다. 판정기를 따로 떼면 8토큰마다 네트워크 왕복이 생겨
실시간 조건을 맞추기 어렵기 때문이다. 요청이 없으면 0대로 줄고(초당 과금), 최대 1대(SQLite 쓰는 곳 하나, 메모리 한도 계산).

배포 (저장소 루트에서):
  modal deploy deploy/modal_app.py
  HG_MIN_CONTAINERS=1 modal deploy deploy/modal_app.py     # 시연 날: 1대를 켜 둬 콜드 스타트 없앰
배포 때만 읽는 값: HG_GPU(기본 L4), HG_MIN_CONTAINERS(0), SCALEDOWN_SECONDS(300), HG_MAX_INPUTS(32)
"""
import os

import modal

APP_NAME = 'hallugate'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(os.path.join(ROOT, 'requirements.txt'))
    # 임베더·토크나이저 내려받기 캐시를 Volume에 둬 콜드 스타트 때 다시 받지 않게 한다
    .env({'HF_HOME': '/models/hf_cache', 'TIKTOKEN_CACHE_DIR': '/models/tiktoken_cache', 'PYTHONUNBUFFERED': '1'})
    # ignore=[]: 기본값은 .py가 아닌 파일을 빼므로, 데모 화면(hallugate/static/demo.html)을 넣으려면 비워야 한다
    .add_local_python_source('hallugate', ignore=['**/__pycache__/**'])
)

data = modal.Volume.from_name('hallugate-data', create_if_missing=True)       # /data: 키·컬렉션·기록 SQLite
models = modal.Volume.from_name('hallugate-models', create_if_missing=True)   # /models: 판정기 가중치·캐시

app = modal.App(APP_NAME, image=image)

# 컨테이너 안 운영 설정 (SPEC §10.7). 비밀 값(ADMIN_TOKEN, ANTHROPIC_API_KEY)은 Modal Secret 'hallugate'에서 온다
HOSTED_ENV = {
    'HOSTED': '1',
    'KEYS_DB': '/data/hallugate.sqlite3',
    'COLLECTION_DB': '/data/collections.sqlite3',
    'JUDGE_MODEL_DIR': '/models/judge_combo',
    'METRICS_PATH': '',
}


@app.function(
    gpu=os.environ.get('HG_GPU', 'L4'),
    volumes={'/data': data, '/models': models},
    secrets=[modal.Secret.from_name('hallugate')],
    min_containers=int(os.environ.get('HG_MIN_CONTAINERS', '0')),
    max_containers=1,
    scaledown_window=int(os.environ.get('SCALEDOWN_SECONDS', '300')),
    timeout=600,
)
@modal.concurrent(max_inputs=int(os.environ.get('HG_MAX_INPUTS', '32')))
@modal.asgi_app()
def web():
    for k, v in HOSTED_ENV.items():
        os.environ.setdefault(k, v)
    from hallugate import Settings, create_app
    return create_app(Settings())      # 판정기 로드·예열·콜드 스타트 기록은 FastAPI lifespan에서 (컨테이너 시작 때)


@app.function(volumes={'/models': models}, timeout=600)
def check_models():
    """판정기 폴더가 제대로 올라갔는지 확인: modal run deploy/modal_app.py::check_models"""
    import json
    base = '/models/judge_combo'
    cfg = json.load(open(os.path.join(base, 'judge_config.json')))
    print('type', cfg.get('type'), '| members', cfg.get('members'))
    for name, sub in cfg.get('members', {}).items():
        d = os.path.join(base, sub)
        print(name, d, sorted(os.listdir(d)) if os.path.isdir(d) else '없음')
