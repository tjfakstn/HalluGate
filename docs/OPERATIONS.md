# 운영 문서 — hallugate 관리형 베타 (Modal)

게이트웨이를 Modal 서버리스 GPU에 올려 초대한 사용자에게 운영하는 절차다. 동작 명세는 [SPEC.md §10](../SPEC.md), 결정 이력은 [SSOT.md §7](../SSOT.md).

- **대상**: 영어 RAG 응답만 검증한다(판정기가 영어로만 학습됨). 영어가 아니면 `verified: false, reason: "non_english"`로 본문은 그대로 전달
- **구성**: GPU 컨테이너 1개(최소 0·최대 1)에 게이트웨이 + 조합 판정기. 요청이 없으면 5분 뒤 0대로 줄고 초당 과금
- **평소엔 꺼 두고, 시연 때만 켠다**: A10G 콜드 스타트는 약 20초(L4는 3~16분이라 쓰지 않음). 첫 요청도 바로 응답해야 하는 시연 때만 `scripts/warm.py up`으로 켜 둔다(§2.1). 무료 크레딧 $30/월 안에서 한 달 약 25시간(A10G)
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
드라이브 `Colab Notebooks/HalluGate/final/judge_base`(F1 base)와 `Colab Notebooks/HalluGate/i2/judge_i2`(I2)를 받아 이 구조로 맞춘다. 가중치는 저장소에 넣지 않는다(`judge_combo/`는 `.gitignore`).
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
| `HG_GPU` | `A10G,L4` | GPU 종류(앞에서부터 가능한 것). L4만 쓰면 콜드 스타트가 수 분~십수 분 |
| `HG_MIN_CONTAINERS` | `0` | 항상 켜 둘 컨테이너 수. 시연 날만 `1` |
| `SCALEDOWN_SECONDS` | `300` | 요청이 없을 때 내리기까지(초) |
| `HG_MAX_INPUTS` | `32` | 컨테이너 하나가 동시에 받는 요청 수 |

### 2.1 쓸 때만 켜기 (평소 운영)
콜드 스타트(꺼진 상태에서 첫 요청)는 A10G에서 약 20초다(11/24 측정. L4는 같은 이미지로 3~16분이라 기본 GPU에서 뺌, SSOT §7). 상시 1대는 월 약 $630이라 쓰지 않는다. 시연처럼 첫 요청 20초도 곤란할 때만 1~2분 전에 켜고, 끝나면 끈다.
```bash
python scripts/warm.py up --hours 2 --wait --url $HG
```
```bash
python scripts/warm.py status
```
```bash
python scripts/warm.py down
```
- `up`은 재배포 없이 최소 1대로 바꾸고 끌 시각을 적는다. `--wait`은 준비될 때까지 기다린다. 최대 12시간
- 끄는 것을 잊어도 배포의 `auto_cooldown`(10분마다)이 끌 시각이 지나면 최소 0대로 되돌린다
- 다시 배포하면 최소 0대로 돌아간다. 초대 사용자에게는 "쓰기 전에 알려 주면 켜 둔다"고 안내한다
- **지출 상한**: Modal 대시보드 Settings → 사용량·결제 화면에서 워크스페이스 지출 상한을 $30(무료 크레딧)으로 건다. 계정 주인이 직접 한다

예외적으로 하루 종일 켜야 하면 `HG_MIN_CONTAINERS=1 modal deploy deploy/modal_app.py` (끝나면 기본값으로 다시 배포)

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
python scripts/keys.py --url $HG create --name budgeted --limits '{"monthly_usd": 20, "budget_action": "block"}'
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
- **LLM 사용량·예산**: 사용자는 `GET $HG/v1/usage?window=30d`(자기 키)로 모델별 토큰과 추정 비용, 이번 달 예산 상태를 본다. 키에 `monthly_tokens` 또는 `monthly_usd`를 주면 모든 응답에 `x-budget-used`가 붙고 80%에서 `x-budget-warning: near`, 100%에서 `exceeded`. `budget_action: block`이면 넘은 뒤 요청은 429 `insufficient_quota`
- **LLM 비용 절감 (근거 다이어트)**: 긴 문서를 근거로 보내는 사용자에게는 `"verification": {"context": "<문서>", "trim": true}`를 권한다. 질문과 관련된 조각 8개만 LLM에 보내 입력 토큰을 줄이고 같은 조각으로 판정한다(T1c: 긴 근거에서 입력 토큰 −47.6%, 답 정답성·충실성 비열등). 이때 문서는 메시지에 따로 넣지 않는다. `top_k`를 낮추면 더 줄지만 근거 밖 내용이 섞일 수 있다(T1b)
- 추정 비용은 `hallugate/prices.json` 단가표로 계산한다(캐시 할인 미반영). **배포 전과 매달 업체 공개 가격표를 보고 갱신**한다
- 사용자의 LLM 키는 고른 제공자에게만 넘기고 기록하지 않는다. 원문(근거·응답·질문)도 기록하지 않는다
- 데모 화면: `$HG/demo` (게이트웨이 키·LLM 키는 그 브라우저 탭에만 보관)
- `anthropic`은 Anthropic의 OpenAI SDK 호환 주소를 쓴다. Anthropic은 이 계층을 주로 시험·비교용으로 안내한다

## 4. 관측

**운영자 화면 `$HG/admin`**: 관리자 토큰을 넣으면 아래 통계와 키 목록을 표·숫자 카드로 본다. 키 발급(이름·한도)·폐기도 여기서 할 수 있다(새 키는 발급 직후 한 번만 보임). 사용자에게는 `$HG/start`(연결 방법)와 `$HG/console`(자기 사용 현황)을 안내한다.

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$HG/admin/stats?window=24h"
```
```bash
modal app logs hallugate -f
```

| 볼 것 (`/admin/stats`) | 정상 범위 (베타 목표, 실측 후 갱신) | 이상하면 |
|---|---|---|
| `latency.ttft_ms.p95` (첫 글자) | 업스트림 첫 토큰 + 수백 ms 이내 | 판정 시간, GPU 종류, 콜드 스타트 비율 확인 |
| `latency.judge_ms_p95` | ≤ 68ms (8토큰 생성 시간, A100 실측 61ms) | 배정된 GPU 종류 확인(로그) → `HG_GPU` |
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
| 첫 요청이 매우 느림(3분 이상) | `cold_starts`, `python scripts/warm.py status` | 정상(서버리스 콜드 스타트). 쓰기 전에 `warm.py up` |
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

- 단가(Modal 가격표 11/24, 초당 값을 시간당으로 환산): L4 약 $0.80, A10 약 $1.10, T4 약 $0.59 + CPU 약 $0.047/코어·메모리 약 $0.008/GiB. 컨테이너가 떠 있는 시간에만 과금
- 무료 요금제(Starter)는 매달 $30 크레딧 → A10G 컨테이너 기준 한 달 약 25시간(L4면 약 34시간, 추정). 상시 1대는 약 $630/월이라 쓰지 않는다
- `auto_cooldown`은 작은 CPU 함수라 10분마다 몇 초씩만 돈다(무시할 수준)
- 2단 검증: 응답마다 Claude Sonnet 3회 호출, `/admin/stats`의 추정 비용으로 확인
- 실제 수치는 배포 후 H8 측정으로 SSOT에 기록한다
