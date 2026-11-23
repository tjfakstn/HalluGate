# 운영 문서 — hallugate 관리형 베타 (Modal)

게이트웨이를 Modal 서버리스 GPU에 올려 초대한 사용자에게 운영하는 절차다. 동작 명세는 [SPEC.md §10](../SPEC.md), 결정 이력은 [SSOT.md §7](../SSOT.md).

- **대상**: 영어 RAG 응답만 검증한다(판정기가 영어로만 학습됨). 영어가 아니면 `verified: false, reason: "non_english"`로 본문은 그대로 전달
- **구성**: GPU 컨테이너 1개(최소 0·최대 1)에 게이트웨이 + 조합 판정기. 요청이 없으면 5분 뒤 0대로 줄고 초당 과금
- **명령은 모두 저장소 루트에서 실행**한다

## 1. 처음 한 번 준비

### 1.1 Modal 계정과 CLI
```bash
pip install modal
```
```bash
modal token new
```
브라우저에서 로그인하면 맥에 토큰이 저장된다. 결제 수단 등록과 요금제 선택은 계정 주인이 직접 한다(무료 크레딧 조건은 [modal.com/pricing](https://modal.com/pricing)에서 확인).

### 1.2 판정기 가중치 올리기
드라이브 `final/`에서 받은 두 판정기를 이 구조로 맞춘다. 가중치는 저장소에 넣지 않는다.
```
judge_combo/
├─ judge_config.json   ← configs/judge_combo_v1.json 복사
├─ f1_base/            ← F1 base 가중치·토크나이저 + 자체 judge_config.json
└─ i2/                 ← I2 가중치·토크나이저 + 자체 judge_config.json
```
```bash
modal volume create hallugate-models
```
```bash
modal volume create hallugate-data
```
```bash
modal volume put hallugate-models judge_combo /judge_combo
```
```bash
modal run deploy/modal_app.py::check_models
```
마지막 명령이 `type combo`와 두 폴더의 파일 목록을 출력하면 된다.

### 1.3 비밀 값
관리자 토큰은 길고 무작위로 만든다. 2단 검증(`post_check`)을 쓰려면 운영자의 Anthropic 키를 함께 넣는다(사용자 요청의 2단 검증 비용이 이 키로 나간다, 하루 한도 `POST_CHECK_DAILY`).
```bash
modal secret create hallugate ADMIN_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))") ANTHROPIC_API_KEY=<운영자 키>
```
관리자 토큰은 비밀번호 관리자에 따로 적어 둔다. 키를 바꿀 때는 Modal 대시보드의 Secrets에서 고치고 다시 배포한다.

## 2. 배포

```bash
modal deploy deploy/modal_app.py
```
출력의 주소(`https://<워크스페이스>--hallugate-web.modal.run`)가 게이트웨이 주소다. 아래에서는 `$HG`로 쓴다.

```bash
curl $HG/health
```
`{"ok": true, "hosted": true}`가 나오면 된다. 첫 호출은 콜드 스타트(컨테이너 시작 + 판정기 로딩 + 예열)라 오래 걸린다.

| 배포 때 바꿀 값 | 기본값 | 쓰임 |
|---|---|---|
| `HG_GPU` | `L4` | GPU 종류 |
| `HG_MIN_CONTAINERS` | `0` | 항상 켜 둘 컨테이너 수. 시연 날만 `1` |
| `SCALEDOWN_SECONDS` | `300` | 요청이 없을 때 내리기까지(초) |
| `HG_MAX_INPUTS` | `32` | 컨테이너 하나가 동시에 받는 요청 수 |

예: `HG_MIN_CONTAINERS=1 modal deploy deploy/modal_app.py` (시연이 끝나면 기본값으로 다시 배포해 비용을 막는다)

**되돌리기**: `modal app history hallugate`로 버전을 보고 `modal app rollback hallugate <버전>`.

## 3. 사용자 관리

셀프 가입은 없다. 관리자가 키를 발급해 사용자에게 안전한 경로로 전달한다. 배포된 게이트웨이는 관리자 API로 관리한다(실행 중인 컨테이너가 저장소를 열고 있기 때문, SPEC §10.1).
```bash
export ADMIN_TOKEN=<관리자 토큰>
```
```bash
python scripts/keys.py --url $HG create --name alice
```
```bash
python scripts/keys.py --url $HG create --name heavy-user --limits '{"rpm": 60, "post_check_daily": 200}'
```
```bash
python scripts/keys.py --url $HG list
```
```bash
python scripts/keys.py --url $HG revoke <키 ID> --purge
```
`--purge`는 그 사용자의 문서(컬렉션)와 측정 기록을 모두 지운다. 사용자가 데이터 삭제를 요청하면 이것을 쓴다.

**사용자에게 줄 안내** (OpenAI SDK):
```python
from openai import OpenAI
client = OpenAI(base_url="<게이트웨이 주소>/v1", api_key="<사용자의 OpenAI 또는 Anthropic 키>",
                default_headers={"x-hallugate-key": "hg_...", "x-upstream": "openai"})   # Claude면 "anthropic"
stream = client.chat.completions.create(model="gpt-4o-mini", stream=True, messages=[...],
                                        extra_body={"verification": {"context": "<영어 근거 문서>"}})
```
- 기본 한도: 분당 30요청, 동시 4개, 하루 2단 검증 50회, 근거 5만 자, 문서 200개(문서당 20만 자). 넘으면 429(`retry-after`) 또는 413
- 사용자의 LLM 키는 고른 제공자에게만 넘기고 기록하지 않는다. 원문(근거·응답·질문)도 기록하지 않는다
- 데모 화면: `$HG/demo` (게이트웨이 키·LLM 키는 그 브라우저 탭에만 보관)
- `anthropic`은 Anthropic의 OpenAI SDK 호환 주소를 쓴다. Anthropic은 이 계층을 주로 시험·비교용으로 안내한다

## 4. 관측

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$HG/admin/stats?window=24h"
```
```bash
modal app logs hallugate -f
```

| 볼 것 (`/admin/stats`) | 정상 범위 (베타 목표, 실측 후 갱신) | 이상하면 |
|---|---|---|
| `latency.ttft_ms.p95` (첫 글자) | 업스트림 첫 토큰 + 수백 ms 이내 | 판정 시간, GPU 종류, 콜드 스타트 비율 확인 |
| `latency.judge_ms_p95` | ≤ 68ms (8토큰 생성 시간, A100 실측 61ms) | GPU가 L4라 느린지 확인 → `HG_GPU` |
| `unverified.rate` / `by_reason` | `judge_error`·`judge_timeout` ≈ 0 | 로그에서 판정기 오류 확인 (§5) |
| `requests.by_status` | 5xx 없음, 429는 일부 | 4xx·5xx 급증 → §5 |
| `post_check.est_cost_usd` | 예산 안 | 키별 `post_check_daily` 낮추기, 필요하면 키 폐기 |
| `cold_starts[].warmup_ms`, `first_ttft_ms` | 기록용 (H8에서 측정) | 시연 날은 `HG_MIN_CONTAINERS=1` |

로그 한 줄은 `{"event": "request", ...}` 형식이고 원문·키가 없다. 비용은 Modal 대시보드의 Usage에서 실제 청구 금액을 본다(`est_cost_usd`는 2단 검증 API 비용 추정만).

## 5. 장애 대응

| 증상 | 확인 | 조치 |
|---|---|---|
| `/health`가 안 뜸, 모든 요청 5xx | `modal app logs hallugate` — 판정기 로딩 오류(가중치 경로·설정), 패키지 설치 오류 | `check_models`로 Volume 확인 → 고친 뒤 다시 배포, 급하면 `modal app rollback` |
| `unverified`가 `judge_error`·`judge_timeout`으로 늘어남 | 로그의 판정기 예외, `judge_ms_p95` | 본문은 fail-open으로 그대로 나가고 있음(사용자 영향은 "검증 안 됨"). GPU 메모리 부족이면 `HG_MAX_INPUTS` 낮추기 |
| 특정 업스트림만 4xx·5xx | `by_status`·`by_upstream`, 업스트림 상태 페이지 | 게이트웨이 문제 아님. 사용자 키 오류(401)는 사용자에게 안내 |
| 첫 요청이 매우 느림 | `cold_starts` | 정상(서버리스). 중요한 시간대만 최소 1대 |
| 비용 급증 | Modal Usage, `post_check.est_cost_usd`, `by_key` | 원인 키의 한도를 낮추거나 폐기. `SCALEDOWN_SECONDS`가 너무 길지 않은지 |
| 키 유출 | 사용자 신고, `by_key` 급증 | 즉시 `revoke`, 새 키 발급 |
| 관리자 토큰 유출 | — | Secret의 `ADMIN_TOKEN`을 바꾸고 다시 배포 |
| 데이터 손상 의심 | `modal volume ls hallugate-data` | 백업에서 복구 (§6) |

장애가 있었으면 날짜, 증상, 원인, 조치, 재발 방지를 `docs/PROGRESS.md`에 남긴다(운영 기록).

## 6. 백업·데이터

- 백업(주 1회 권장):
  ```bash
  modal volume get hallugate-data /hallugate.sqlite3 backup/
  ```
  ```bash
  modal volume get hallugate-data /collections.sqlite3 backup/
  ```
  백업 파일에는 사용자 문서가 들어 있으므로 저장소에 넣지 않고 암호화된 곳에 둔다
- 측정 기록은 30일(`METRICS_RETENTION_DAYS`) 뒤 자동으로 지워진다. 컬렉션 문서는 사용자가 지우거나 `--purge`할 때까지 남는다
- SQLite는 컨테이너 1대만 쓰므로 쓰는 곳이 하나다. 최대 컨테이너 수를 늘리려면 저장소를 먼저 바꿔야 한다(SPEC §10.7)

## 7. 비용 감각 (추정, 실측으로 갱신)

- GPU: L4 약 $0.80/시간(공개 자료) + CPU·메모리. 컨테이너가 떠 있는 시간에만 과금 → (하루 사용 시간 + 요청마다 `SCALEDOWN_SECONDS` 대기) × 단가
- 예: 하루 1시간 쓰고 나머지는 꺼짐 → 하루 약 $1 내외. 최소 1대를 하루 종일 켜 두면 약 $20 이상
- 2단 검증: 응답마다 Claude Sonnet 3회 호출, `/admin/stats`의 추정 비용으로 확인
- 실제 수치는 배포 후 H8 측정으로 SSOT에 기록한다
