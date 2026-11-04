# SPEC — hallugate 동작 명세

게이트웨이가 **무엇을 어떻게 해야 하는지**를 정의한다. 수치·실험 결과·결정 이력은 [SSOT.md](SSOT.md)에 있다.
구현이 이 명세와 다르면 둘 중 하나를 고치고, 명세를 바꾸는 경우 SSOT의 결정 기록에 남긴다.

상태 표기: ✅ 구현·테스트됨 · 🚧 부분 구현 · 📋 계획(S4 이후)

---

## 1. 개요

hallugate는 서비스와 LLM API 사이에 위치하는 **OpenAI Chat Completions 호환 프록시**다.
요청에 포함된 근거 문서를 기준으로, LLM 응답을 **스트리밍 중에 문장 단위로** 검증한다.

```
[관리자/개발자] ── 문서 등록 API ──▶ 컬렉션 저장소 (조각 + 임베딩)
                                          │
서비스 ──요청(+collection 또는 context)──▶ hallugate ──요청(근거 삽입, 검증 필드 제거)──▶ LLM API (Claude / OpenAI 형식)
       ◀──본문 청크 + 판정 청크──   │   ◀──────── 토큰 스트림 ────────
                                   ├─ Retriever        : 질문 → 컬렉션 근거 조각 N개 (📋)
                                   ├─ IncrementalSplitter
                                   ├─ Judge             : (근거 조각, 문장 앞부분) → 점수
                                   └─ VerifyPipeline    : pass/hold(8토큰), flag/cut/regenerate(📋)
```

**RAG 게이트웨이 (📋 S4, SSOT §7 10/28)**: 근거 문서를 서비스가 매번 요청에 담는 대신, 관리자·개발자가 게이트웨이의 컬렉션에
미리 등록해 둘 수 있다. 요청이 `verification.collection`을 지정하면 게이트웨이가 마지막 사용자 질문으로 컬렉션을 검색하고,
찾은 근거를 시스템 메시지에 넣어 LLM을 호출한 뒤 **같은 근거로** 응답을 판정한다. 이렇게 하면 LLM이 본 근거와 판정 근거가 항상 같다.
요청에 근거를 직접 담는 기존 방식(검증만)도 그대로 지원한다.

### 목표
- G1. 서비스 코드는 `base_url` 변경만으로 적용 가능해야 한다.
- G2. 검증이 켜져 있어도 본문은 한 글자도 바뀌거나 빠지면 안 된다 (cut으로 끊는 경우 제외).
- G3. 판정 결과를 표준 OpenAI 클라이언트가 깨지지 않는 방식으로 전달한다.
- G4. 정확도-지연 트레이드오프를 요청마다(향후 문장마다) 조절할 수 있어야 한다.
- G5. 모든 요청의 시간·판정 기록을 남겨 측정할 수 있어야 한다.

### 비목표
- 근거 문서 없는 사실성 검증, 웹 검색 기반 검증
- 응답 문장 고쳐 쓰기(교정 모델). 단, hold에서 아직 나가지 않은 문장의 재생성은 지원(§5.2 📋)
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
    "action": "flag" | "cut",            // 기본: 서버 설정 VERIFY_ACTION (flag)
    "stride": 4,                         // 선택. 문장 앞부분 재판정 간격(현재 업스트림 조각 수, 📋 tiktoken cl100k 토큰으로 변경). 0이면 끔. 기본: JUDGE_STRIDE
    "collection": "id",                  // 📋 선택. 등록된 컬렉션에서 근거 검색 + 프롬프트 삽입 (§2.5, §2.6)
    "regenerate": true | false           // 📋 선택. hold에서 환각 문장 재생성 (§5.2). 기본 false
  }
}
```

**근거 문서 결정 순서** (`hallugate/context.py`)
1. `verification.context` (리스트면 빈 줄로 이어 붙임)
2. 모든 메시지 본문 안의 `<context>…</context>` 태그 내용
3. 📋 `verification.collection` (컬렉션 검색, §2.6)
4. 모두 없으면 `mode = off` (검증 없이 전달)

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
{"kind": "sentence" | "prefix", "sentence_idx": 1, "start": 42, "upto": 97,
 "text": "판정한 텍스트(앞뒤 공백 제거)", "score": 2.31, "hallucinated": true,
 "judge_ms": 14.8, "released_at_verdict": 120}
```
- `kind`: `sentence`는 완성 문장 판정, `prefix`는 문장이 끝나기 전 앞부분 판정(재판정 간격 `stride`가 켜졌을 때)
- `start`/`upto`: 판정한 텍스트가 응답 전체에서 차지하는 글자 구간 [start, upto)
- `released_at_verdict`: 이 판정이 나온 순간까지 사용자에게 보낸 글자 수 (누출 계산용)
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

### 2.5 문서 등록 API 📋
- `POST /v1/collections/{id}/documents` — 텍스트 문서 등록. 게이트웨이가 조각으로 나누고 임베딩해 저장한다.
- `GET /v1/collections/{id}/documents` — 등록된 문서 목록.
- `DELETE /v1/collections/{id}/documents/{doc_id}` — 문서 삭제.
- 저장소는 `Store` 인터페이스로 감싼다(처음 구현은 SQLite + 임베딩 배열). 판정기와 마찬가지로 나중에 별도 서비스로 분리할 수 있게 한다.
- 접근 토큰 하나로 보호한다.

### 2.6 프롬프트 삽입 📋
- `verification.collection`이 있으면 마지막 사용자 메시지로 컬렉션을 검색해 근거 조각을 고른다.
- 시스템 메시지에 "아래 근거만 사용해 답하라" 지시와 근거 블록을 넣는다. 서비스가 보낸 메시지는 바꾸지 않는다.
- 판정기의 근거 검색은 **이 근거 조각 안에서만** 한다(LLM에 넣은 근거 = 판정 근거).
- 판정 청크에 사용한 근거 조각 목록(`evidence`)을 싣는다.

### 2.7 업스트림 📋
- `UPSTREAM_PROVIDER=openai|anthropic`. OpenAI 형식은 그대로 통과, Claude는 Messages API로 변환해 호출하고 응답을 OpenAI 청크 형식으로 되돌린다.
- 스트리밍의 `tool_calls`·`usage` 청크는 판정 없이 그대로 통과한다. (현재는 `delta.content`만 넘겨 이 청크가 버려짐 — 버그)

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

- `hallucinated = score >= JUDGE_THRESHOLD` (기본 0.0, 📋 S4에서 보정 — 0.0은 문장 재현율 0.22로 너무 보수적, SSOT §5 S3)
- 📋 운영 판정기를 F1 base(DeBERTa-v3-base)로 교체(모델 v1.0 동결 후 ✅). `judge_config.json`(임계값·재판정 간격·임베더·k)을 읽는다. 배포 기본값은 8토큰·임계값 3.906(SSOT §5 F1).
- 📋 `Judge.judge_batch()`: 여러 판정을 한 번에 처리.
- 📋 재판정 간격 단위는 **tiktoken cl100k 토큰**. 현재는 업스트림 조각 수라 학습·임계값과 맞지 않는다(Claude는 조각당 약 5토큰).
- 📋 영어가 아닌 응답은 판정을 끄고 `verified: false`.
- 📋 판정기 장애·시간 초과 시 본문을 그대로 전달하고 `verified: false`(fail-open).
- 서버 시작 시 판정기를 예열한다 (근거 준비 1회 + 판정 3회) ✅
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

### 5.1 앞부분 재판정 (`stride` = s ≥ 1) ✅
문장이 끝나기를 기다리지 않고, 현재 문장에 업스트림 조각이 s개 쌓일 때마다 **지금까지의 문장 앞부분**을 판정한다
(앞부분 학습 판정기 `student_prefix`와 함께 쓰는 것을 전제로 한다, SSOT E2). 문장이 완성되면 그 문장 전체도 판정한다.

| mode | 동작 |
|---|---|
| `pass` | 본문은 즉시 전달. 앞부분 판정과 문장 판정을 백그라운드로 실행하고 결과를 **발생 순서대로** 전달 → 문장 도중에도 경보 가능 |
| `hold` | 본문을 보류하다가, 앞부분 판정이 **환각 아님**이면 그 시점까지 쌓인 본문을 내보낸다. 문장 완성 시 문장 판정 후 나머지를 내보낸다. 보류 지연이 "문장 길이"에서 "s 조각 + 판정 시간"으로 줄어든다 |

- 환각 판정 시 동작은 `action`을 따른다. `hold+cut`이면 환각으로 판정된 앞부분의 보류 중인 글자는 나가지 않는다.
- 불변식 G2(본문 보존)는 그대로 유지된다.
- 조각 수는 업스트림 SSE 조각 기준이다(상용 API에서는 대략 토큰 1개, 가짜 LLM에서는 단어 1개).

- 📋 S4: 위험도에 따라 재판정 간격과 보류 여부를 바꾸는 적응형 정책.
- 📋 기본 정책 변경(SSOT §7 10/28 A6): `VERIFY_MODE=hold`, `JUDGE_STRIDE=8`(토큰), `VERIFY_ACTION=flag`. `cut`은 기본값에서 제외.

### 5.2 재생성 (`regenerate`) 📋
hold에서 환각으로 판정된 문장은 아직 사용자에게 나가지 않았으므로 버리고, "근거만 사용" 지시와 함께 그 지점부터 이어 쓰기를 **1회** 재요청한다.
- 재생성 여부는 표시용보다 보수적인 임계값(예: cal-L 멀쩡한 응답 5%)으로 정한다.
- 재생성한 문장도 다시 판정하고, 또 환각이면 표시만(flag) 한다.
- 해소율·추가 지연을 측정 기록에 남긴다.
- 불변식 G2는 "사용자에게 전달된 본문 = 최종 업스트림 본문 중 버리지 않은 부분"으로 해석한다(버린 문장은 전달되지 않음).

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
| `JUDGE_STRIDE` | 0 | 앞부분 재판정 간격 (0 = 문장 끝에서만 판정) |
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
| A5 | 게이트웨이 안 AUROC ≥ 0.80 (student_vark, RAGTruth dev 60개 재생) | ✅ 0.853 |
| A6 | GPU 판정 p95 ≤ 30ms, pass 판정 전달 p95 ≤ 50ms | ✅ 20.6ms / 21.1ms |
| A8 | `stride`를 켜도 본문 보존 100%, hold에서 앞부분이 안전 판정되면 문장 완성 전에 그 부분이 전달됨 | ✅ 테스트 |
| A7 | 운영 임계값에서 멀쩡한 응답 오표시율과 환각 응답 탐지율이 서비스 요구를 만족 (기준은 S4에서 정의) | 📋 임계값 0에서 6% / 31% |
| A9 | 컬렉션 근거로 생성·검증할 때 본문 보존 100%, LLM에 넣은 근거 = 판정 근거 | 📋 |
| A10 | 스트리밍의 `tool_calls`·`usage` 청크 보존 | 📋 |
| A11 | 고정 예제(`parity_fixture.json`) 점수 일치 \|Δlogit\| ≤ 0.2 | 📋 |

## 9. 로드맵

| 단계 | 내용 |
|---|---|
| S3 | ~~학습 판정기로 A5·A6 측정, CPU 판정 속도 측정~~ ✅, 중간보고 |
| F | 모델 v1.0 동결: F1 base 판정기, `judge_config.json` v1.0(8토큰·서비스 10% 임계값, 재생성용 임계값), `judge_batch`, 고정 예제 일치 |
| S4 | 재판정 간격 토큰 단위, `tool_calls`·`usage` 통과(A10), Claude Messages API 변환(§2.7), 컬렉션 API·저장소·검색·프롬프트 삽입·`evidence`(§2.5·2.6, A9), 재생성(§5.2), 언어 감지, fail-open, 접근 토큰, 판정 묶음 처리, 배포 후 실제 LLM 속도로 종단 간 재측정 |
| S5 | RAGTruth test 1회 개방 최종 평가, 기존 도구의 고정 설정(NeMo Guardrails의 chunk 200 등)과 비교 |
| S6 | 데모 (서비스 예제 + 판정 시각화), 보고서, SOFTCON |
