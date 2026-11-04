"""게이트웨이 설정. 환경 변수로 덮어쓸 수 있다.

판정기 관련 값(임베더·k·입력 길이·임계값·재판정 간격)은 비워 두면 판정기 폴더의 judge_config.json 값을 쓴다 (SPEC §4, §7).
"""
import os
from dataclasses import dataclass, field


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == '' else cast(v)


@dataclass
class Settings:
    # 실제 LLM API (OpenAI 호환). 로컬 테스트 때는 mock_upstream 주소.
    upstream_base_url: str = field(default_factory=lambda: _env('UPSTREAM_BASE_URL', 'https://api.openai.com/v1'))
    upstream_api_key: str = field(default_factory=lambda: _env('UPSTREAM_API_KEY', ''))
    upstream_timeout: float = field(default_factory=lambda: _env('UPSTREAM_TIMEOUT', 120.0, float))

    # 판정기 폴더 (모델 v1.0 = F1 base). judge_config.json이 있으면 아래 None 값들을 거기서 채운다.
    judge_model_dir: str = field(default_factory=lambda: _env('JUDGE_MODEL_DIR', 'judge_base'))
    judge_embedder: str = field(default_factory=lambda: _env('JUDGE_EMBEDDER', None))
    judge_k: int = field(default_factory=lambda: _env('JUDGE_K', None, int))
    judge_max_len: int = field(default_factory=lambda: _env('JUDGE_MAX_LEN', None, int))
    judge_chunk_words: int = field(default_factory=lambda: _env('JUDGE_CHUNK_WORDS', None, int))
    # 판정기 출력(logit)이 이 값 이상이면 환각. 지정하면 설정 파일의 간격별 임계값 대신 이 값 하나를 쓴다.
    judge_threshold: float = field(default_factory=lambda: _env('JUDGE_THRESHOLD', None, float))
    # 문장 앞부분 재판정 간격(토큰). 0이면 문장 끝에서만 판정. None이면 설정 파일 기본값(F1 base는 8).
    judge_stride: int = field(default_factory=lambda: _env('JUDGE_STRIDE', None, int))
    # 재판정 간격을 세는 토큰 단위. 학습과 같은 cl100k_base. 'words'는 테스트용 대체 단위
    stride_tokenizer: str = field(default_factory=lambda: _env('STRIDE_TOKENIZER', 'cl100k_base'))
    judge_device: str = field(default_factory=lambda: _env('JUDGE_DEVICE', ''))
    # 판정 1회 시간 제한(ms). 넘거나 판정기가 오류를 내면 그 요청은 검증을 멈추고 본문을 그대로 전달한다(fail-open).
    judge_timeout_ms: float = field(default_factory=lambda: _env('JUDGE_TIMEOUT_MS', 2000.0, float))

    # 기본 정책 (요청마다 verification 필드로 바꿀 수 있음). SSOT §7 10/28 A6: hold + 8토큰 + flag
    #   pass : 토큰을 즉시 내보내고 판정 결과를 뒤따라 보냄 (지연 0, 틀린 문장이 먼저 나갈 수 있음)
    #   hold : 판정이 끝날 때까지 붙잡았다가 내보냄 (틀린 문장 차단 가능, 재판정 간격만큼 지연)
    #   off  : 검증 없이 그대로 전달
    default_mode: str = field(default_factory=lambda: _env('VERIFY_MODE', 'hold'))
    #   flag : 환각 판정 문장에 표시만 함
    #   cut  : 환각 판정 시 응답을 끊음 (finish_reason=content_filter)
    default_action: str = field(default_factory=lambda: _env('VERIFY_ACTION', 'flag'))

    # 빈 값이면 기록하지 않는다
    metrics_path: str = field(default_factory=lambda: os.environ.get('METRICS_PATH', 'gateway_metrics.jsonl'))
