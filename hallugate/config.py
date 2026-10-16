"""게이트웨이 설정. 환경 변수로 덮어쓸 수 있다."""
import os
from dataclasses import dataclass, field


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None else cast(v)


@dataclass
class Settings:
    # 실제 LLM API (OpenAI 호환). 로컬 테스트 때는 mock_upstream 주소.
    upstream_base_url: str = field(default_factory=lambda: _env('UPSTREAM_BASE_URL', 'https://api.openai.com/v1'))
    upstream_api_key: str = field(default_factory=lambda: _env('UPSTREAM_API_KEY', ''))
    upstream_timeout: float = field(default_factory=lambda: _env('UPSTREAM_TIMEOUT', 120.0, float))

    # 판정기 (S2-b에서 학습한 student_vark)
    judge_model_dir: str = field(default_factory=lambda: _env('JUDGE_MODEL_DIR', 'student_vark'))
    judge_embedder: str = field(default_factory=lambda: _env('JUDGE_EMBEDDER', 'sentence-transformers/all-MiniLM-L6-v2'))
    judge_k: int = field(default_factory=lambda: _env('JUDGE_K', 8, int))
    judge_max_len: int = field(default_factory=lambda: _env('JUDGE_MAX_LEN', 512, int))
    judge_chunk_words: int = field(default_factory=lambda: _env('JUDGE_CHUNK_WORDS', 40, int))
    # 판정기 출력(logit)이 이 값 이상이면 환각으로 본다. 0.0 = 확률 0.5. S4에서 보정 예정.
    judge_threshold: float = field(default_factory=lambda: _env('JUDGE_THRESHOLD', 0.0, float))
    # 문장 앞부분 재판정 간격(업스트림 조각 수). 0이면 문장 끝에서만 판정. 앞부분 학습 판정기(student_prefix)와 함께 쓴다.
    judge_stride: int = field(default_factory=lambda: _env('JUDGE_STRIDE', 0, int))
    judge_device: str = field(default_factory=lambda: _env('JUDGE_DEVICE', ''))

    # 기본 정책 (요청마다 verification 필드로 바꿀 수 있음)
    #   pass : 토큰을 즉시 내보내고 판정 결과를 뒤따라 보냄 (지연 0, 틀린 문장이 먼저 나갈 수 있음)
    #   hold : 문장 단위로 붙잡았다가 판정 후 내보냄 (틀린 문장 차단 가능, 문장 길이만큼 지연)
    #   off  : 검증 없이 그대로 전달
    default_mode: str = field(default_factory=lambda: _env('VERIFY_MODE', 'pass'))
    #   flag : 환각 판정 문장에 표시만 함
    #   cut  : 환각 판정 시 응답을 끊음 (finish_reason=content_filter)
    default_action: str = field(default_factory=lambda: _env('VERIFY_ACTION', 'flag'))

    metrics_path: str = field(default_factory=lambda: _env('METRICS_PATH', 'gateway_metrics.jsonl'))
