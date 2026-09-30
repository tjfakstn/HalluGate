# SPEC — hallugate 동작 명세

게이트웨이가 **무엇을 어떻게 해야 하는지**를 정의한다. 수치·실험 결과·결정 이력은 [SSOT.md](SSOT.md)에 있다.
구현이 이 명세와 다르면 둘 중 하나를 고치고, 명세를 바꾸는 경우 SSOT의 결정 기록에 남긴다.

상태 표기: ✅ 구현·테스트됨 · 🚧 부분 구현 · 📋 계획(S4 이후)

---

## 1. 개요

hallugate는 서비스와 LLM API 사이에 위치하는 **OpenAI Chat Completions 호환 프록시**다.
요청에 포함된 근거 문서를 기준으로, LLM 응답을 **스트리밍 중에 문장 단위로** 검증한다.

```
서비스 ──요청(근거 문서 포함)──▶ hallugate ──요청(검증 필드 제거)──▶ LLM API
       ◀──본문 청크 + 판정 청크──   │   ◀──────── 토큰 스트림 ────────
                                   ├─ IncrementalSplitter : 토큰 → 문장
                                   ├─ Judge               : (근거, 문장) → 점수
                                   └─ VerifyPipeline      : 정책(pass/hold, flag/cut)에 따라 전달
```

### 목표
- G1. 서비스 코드는 `base_url` 변경만으로 적용 가능해야 한다.
- G2. 검증이 켜져 있어도 본문은 한 글자도 바뀌거나 빠지면 안 된다 (cut으로 끊는 경우 제외).
- G3. 판정 결과를 표준 OpenAI 클라이언트가 깨지지 않는 방식으로 전달한다.
- G4. 정확도-지연 트레이드오프를 요청마다(향후 문장마다) 조절할 수 있어야 한다.
- G5. 모든 요청의 시간·판정 기록을 남겨 측정할 수 있어야 한다.

### 비목표
- 근거 문서 없는 사실성 검증, 웹 검색 기반 검증
- 응답 수정·재생성 (환각 문장을 고쳐 쓰기)
- 인증·과금·레이트 리밋 (운영용 게이트웨이 기능)

## 2. API

### 2.1 `POST /v1/chat/completions` ✅

OpenAI Chat Completions 요청을 그대로 받는다. 추가 필드 하나를 인식한다.

```jsonc
{
  "model": "gpt-4o-mini",
  "messages": [...],
  "stream": true,
  "verification": {                     // 선택. OpenAI SDK에서는 extra_body로 전달
    "context": "근거 문서" | ["문서1", "문서2"],
    "mode": "pass" | "hold" | "off",     // 기본: 서버 설정 VERIFY_MODE (pass)
    "action": "flag" | "cut"             // 기본: 서버 설정 VERIFY_ACTION (flag)
  }
}
```

**근거 문서 결정 순서** (`hallugate/context.py`)
1. `verification.context` (리스트면 빈 줄로 이어 붙임)
2. 모든 메시지 본문 안의 `<context>…</context>` 태그 내용
3. 둘 다 없으면 `mode = off` (검증 없이 전달)

**전달 규칙**
- `verification` 필드는 업스트림에 보내기 전에 제거한다 (실제 API는 모르는 필드를 거부함).
- 인증: 서버 설정 `UPSTREAM_API_KEY`가 있으면 그것을, 없으면 요청의 `Authorization` 헤더를 그대로 전달한다.
- 업스트림이 200이 아니면 상태 코드와 본문을 그대로 돌려준다.
- 잘못된 `mode`/`action` → 400 `invalid_request_error`.
- 응답 헤더: `x-verification-mode`, `x-verification-action`.
- 요청 헤더 `x-request-tag`는 측정 기록의 식별자로 저장한다.

### 2.2 스트리밍 응답 ✅

SSE(`text/event-stream`). 세 종류의 청크가 나간다.

| 청크 | 형태 | 비고 |
|---|---|---|
| 본문 | 표준 `chat.completion.chunk`, `choices[0].delta.content` | 첫 청크는 `delta.role = "assistant"` |
| 판정 | `choices: []` + `verification` 객체 | 표준 클라이언트는 무시, 필요하면 `chunk.verification`으로 읽음 |
| 종료 | `choices[0].finish_reason` = 업스트림 값(기본 `stop`) 또는 `content_filter`(cut) 후 `data: [DONE]` | |

판정 객체:
```jsonc
{"sentence_idx": 1, "start": 42, "text": "문장(앞뒤 공백 제거)",
 "score": 2.31, "hallucinated": true, "judge_ms": 14.8}
```
cut 발생 시 종료 청크에 `{"cut": true, "sentence_idx": n}`이 함께 실린다.

### 2.3 비스트리밍 응답 ✅
업스트림 응답을 받은 뒤 전체를 문장으로 나눠 판정하고, 응답 JSON에 `verification`을 추가한다.
```jsonc
"verification": {"mode": "hold", "action": "cut", "prepare_ms": 3.1, "cut": true,
                 "sentences": [ {판정 객체}, ... ]}
```
`hold + cut`이면 본문을 환각 문장 직전까지 자르고 `finish_reason = content_filter`.
그 밖의 조합에서는 본문을 바꾸지 않는다. `cut`이면 판정을 첫 환각 문장에서 멈춘다.

### 2.4 `GET /health` ✅
`{"ok": true, "judge": "<판정기 클래스>", "mode": "...", "action": "..."}`

## 3. 문장 분할 — `IncrementalSplitter` ✅

- pysbd(영어, `char_span`)를 사용한다. 버퍼에서 **다음 문장의 시작이 확인되면** 앞 문장을 완성으로 본다.
- 스트림 종료 시 `flush()`로 남은 텍스트를 마지막 문장으로 내보낸다.
- **불변식**
  - I1. 내보낸 문장의 `text`를 순서대로 이어 붙이면 원문과 같다 (공백 포함).
  - I2. `start`는 응답 전체에서의 문자 위치이며 앞 문장들의 길이 합과 같다.
  - I3. 조각 단위로 넣어도 한 번에 넣은 결과와 같은 문장 경계가 나온다 (테스트된 입력 기준).
- 공백뿐인 문장은 판정하지 않는다 (`skipped`).
- 📋 S4: 문장보다 짧은 단위(절, N토큰)와 위험 신호 기반 가변 단위.

## 4. 판정기 — `Judge` 인터페이스 ✅

```python
class Judge:
    def prepare(self, reference: str) -> RefIndex   # 요청당 1회: 근거 분할·임베딩
    def judge(self, ref: RefIndex, sentence: str) -> Verdict(score, hallucinated, judge_ms)
```

| 구현 | 용도 |
|---|---|
| `StudentJudge` | 운영. 40단어 조각 → 임베딩 top-k(원문 순서) → 교차 인코더 logit. 사양은 SSOT §4 |
| `LexicalJudge` | 테스트·데모 전용. 근거에 없는 단어 비율. **판별력 없음, 측정에 사용 금지** |

- `hallucinated = score >= JUDGE_THRESHOLD` (기본 0.0, 📋 S4에서 보정)
- 모델 하나를 여러 요청이 공유하므로 `StudentJudge`는 락으로 직렬화한다. 판정은 `asyncio.to_thread`로 이벤트 루프 밖에서 실행한다.

## 5. 전달 정책 — `VerifyPipeline` ✅

| mode | 본문 전달 | 판정 전달 | 추가 지연 | 누출 |
|---|---|---|---|---|
| `off` | 업스트림 바이트 그대로 | 없음 | 0 | — |
| `pass` | 토큰 도착 즉시 | 문장 완성 → 백그라운드 판정 → **완료 즉시** (다음 토큰을 기다리지 않음), 문장 순서 유지 | ≈ 0 | 판정 완료 전에 나간 글자 |
| `hold` | 문장 완성 → 판정 → 판정 청크 → 그 문장 본문 | 본문보다 먼저 | 문장 생성 시간 + 판정 시간 | 0 |

| action | 환각 판정 시 |
|---|---|
| `flag` | 판정 청크의 `hallucinated=true`로 표시만 함 |
| `cut` | 판정 청크 → 종료 청크(`content_filter`) → 스트림 종료, 업스트림 연결 닫음. hold면 해당 문장은 한 글자도 나가지 않음 |

- 업스트림 읽기는 별도 태스크(producer)로 돌려, 판정 중에도 토큰 수신이 멈추지 않게 한다.
- 📋 S4: 문장·구간별로 정책을 바꾸는 적응형 정책. 예: 숫자·고유명사가 있는 구간만 hold.

## 6. 측정 기록 ✅

요청마다 `METRICS_PATH`(JSONL)에 한 줄을 남긴다 (`RequestRecord`). 시각은 요청 수신 기준 ms.

| 필드 | 의미 |
|---|---|
| `tag`, `mode`, `action`, `cut` | 식별자와 정책 |
| `prepare_ms` | 근거 분할·임베딩 시간 |
| `t_upstream_first`, `t_upstream_done` | 업스트림 첫 토큰·종료 시각 |
| `t_client_first`, `t_client_done` | 사용자에게 첫 본문·마지막 이벤트를 보낸 시각 |
| `released_chars` | 사용자에게 보낸 글자 수 |
| `sentences[]` | `sentence_idx, start, n_chars, t_complete, t_verdict, t_release, score, hallucinated, judge_ms, leak_chars` |

`leak_chars` = 환각 판정이 나온 순간 기준으로, 그 문장 시작 이후 사용자에게 이미 나간 글자 수 (hold는 0).

`scripts/bench.py`가 이 기록과 클라이언트 측 시각을 합쳐 계산하는 항목:
본문 보존율, AUROC, 임계값 기준 재현율·정밀도, 판정 시간 p50/p95, 판정 전달 지연, TTFT 증가,
글자당 평균 지연(off 대비), 누출량.

## 7. 설정 (환경 변수) ✅

| 변수 | 기본값 | 설명 |
|---|---|---|
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | 실제 LLM API |
| `UPSTREAM_API_KEY` | (빈 값) | 비우면 요청의 Authorization 전달 |
| `UPSTREAM_TIMEOUT` | 120 | 초 |
| `JUDGE_MODEL_DIR` | `student_vark` | 학습된 판정기 폴더 |
| `JUDGE_EMBEDDER` | `sentence-transformers/all-MiniLM-L6-v2` | 근거 검색 임베딩 |
| `JUDGE_K` / `JUDGE_MAX_LEN` / `JUDGE_CHUNK_WORDS` | 8 / 512 / 40 | 판정기 입력 구성 |
| `JUDGE_THRESHOLD` | 0.0 | 환각 판정 logit 임계값 |
| `JUDGE_DEVICE` | 자동 | `cuda` / `cpu` |
| `VERIFY_MODE` / `VERIFY_ACTION` | `pass` / `flag` | 요청에 지정이 없을 때 기본 정책 |
| `METRICS_PATH` | `gateway_metrics.jsonl` | 빈 값이면 기록 안 함 |

## 8. 수용 기준

| ID | 기준 | 상태 |
|---|---|---|
| A1 | 모든 모드에서 본문 보존 100% (cut 제외) | ✅ 테스트 / 대체 판정기 재생 15개 |
| A2 | OpenAI 공식 SDK(스트리밍)로 호출 가능, 판정을 `chunk.verification`으로 읽을 수 있음 | ✅ 테스트 |
| A3 | hold+cut에서 환각 문장이 한 글자도 나가지 않음 | ✅ 테스트 |
| A4 | pass에서 판정이 다음 토큰을 기다리지 않고 전달됨 | ✅ 테스트 (토큰 간격 50ms, 전달 지연 < 40ms) |
| A5 | 게이트웨이 안 AUROC ≥ 0.80 (student_vark, RAGTruth dev 60개 재생) | 🚧 측정 대기 |
| A6 | GPU 판정 p95 ≤ 30ms, pass 판정 전달 p95 ≤ 50ms | 🚧 측정 대기 |

## 9. 로드맵

| 단계 | 내용 |
|---|---|
| S3 | 학습 판정기로 A5·A6 측정, CPU 판정 속도 측정, 중간보고 |
| S4 | 임계값 보정, 문장보다 짧은 검증 단위(부분 문장 판정 성능 측정 포함), 위험 신호 기반 가변 단위·정책, 상용 API 연결 |
| S5 | RAGTruth test 1회 개방 최종 평가, 기존 도구의 고정 설정(NeMo Guardrails의 chunk 200 등)과 비교 |
| S6 | 데모 (서비스 예제 + 판정 시각화), 보고서, SOFTCON |
