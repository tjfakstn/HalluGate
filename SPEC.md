# SPEC — hallugate 동작 명세

게이트웨이가 **무엇을 어떻게 해야 하는지**를 정의한다. 수치·실험 결과·결정 이력은 [SSOT.md](SSOT.md)에 있다.
구현이 이 명세와 다르면 둘 중 하나를 고치고, 명세를 바꾸는 경우 SSOT의 결정 기록에 남긴다.

상태 표기: ✅ 구현·테스트됨 · 🧪 실험 기능(구현·테스트됨, 효과 미검증, 기본 꺼짐) · 🚧 부분 구현 · 📋 계획(S4 이후)

---

## 1. 개요

hallugate는 서비스와 LLM API 사이에 위치하는 **OpenAI Chat Completions 호환 프록시**다.
요청에 포함된 근거 문서를 기준으로, LLM 응답을 **스트리밍 중에 문장 단위로** 검증한다.

```
[관리자/개발자] ── 문서 등록 API ──▶ 컬렉션 저장소 (조각 + 임베딩)
                                          │
서비스 ──요청(+collection 또는 context)──▶ hallugate ──요청(근거 삽입, 검증 필드 제거)──▶ LLM API (Claude / OpenAI 형식)
       ◀──본문 청크 + 판정 청크──   │   ◀──────── 토큰 스트림 ────────
                                   ├─ Store            : 질문 → 컬렉션 근거 조각 N개 (✅ SQLite)
                                   ├─ IncrementalSplitter
                                   ├─ Judge             : (근거 조각, 문장 앞부분) → 점수
                                   └─ VerifyPipeline    : pass/hold(8토큰), flag/cut/regenerate(📋)
```

**RAG 게이트웨이 (✅ S4 11/05, SSOT §7 10/28)**: 근거 문서를 서비스가 매번 요청에 담는 대신, 관리자·개발자가 게이트웨이의 컬렉션에
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
- 과금·결제, 셀프 가입, 여러 지역 배포, SLA. (인증·사용자 분리·사용량 제한은 운영 모드에서만 지원 — §10)

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
    "mode": "pass" | "hold" | "off",     // 기본: 서버 설정 VERIFY_MODE (hold)
    "action": "flag" | "cut",            // 기본: 서버 설정 VERIFY_ACTION (flag)
    "stride": 8,                         // 선택. 문장 앞부분 재판정 간격(tiktoken cl100k 토큰, §5.1). 0이면 끔. 기본: JUDGE_STRIDE, 없으면 판정기 설정 파일(F1 base 8)
    "collection": "id",                  // 선택. 등록된 컬렉션에서 근거 검색 + 프롬프트 삽입 (§2.5, §2.6)
    "regenerate": true | false,          // 📋 선택. hold에서 환각 문장 재생성 (§5.2). 기본 false
    "post_check": true | false           // 🧪 선택. 응답 끝 2단 검증 (§5.3). 기본: 서버 설정 POST_CHECK (false)
  }
}
```

**근거 문서 결정 순서** (`hallugate/context.py`)
1. `verification.context` (리스트면 빈 줄로 이어 붙임)
2. 모든 메시지 본문 안의 `<context>…</context>` 태그 내용
3. `verification.collection` (컬렉션 검색, §2.6). 1·2가 있으면 컬렉션은 쓰지 않는다
4. 모두 없으면 `mode = off` (검증 없이 전달)

**전달 규칙**
- `verification` 필드는 업스트림에 보내기 전에 제거한다 (실제 API는 모르는 필드를 거부함).
- 인증: 서버 설정 `UPSTREAM_API_KEY`가 있으면 그것을, 없으면 요청의 `Authorization` 헤더를 그대로 전달한다.
- 업스트림이 200이 아니면 상태 코드와 본문을 그대로 돌려준다.
- 잘못된 `mode`/`action`, 불리언이 아닌 `post_check` → 400 `invalid_request_error`.
- 응답 헤더: `x-verification-mode`, `x-verification-action`, `x-verification-stride`, `x-verification-post-check`(`on`/`off`).
- 요청 헤더 `x-request-tag`는 측정 기록의 식별자로 저장한다.

### 2.2 스트리밍 응답 ✅

SSE(`text/event-stream`). 세 종류의 청크가 나간다.

| 청크 | 형태 | 비고 |
|---|---|---|
| 본문 | 표준 `chat.completion.chunk`, `choices[0].delta.content` | 첫 청크는 `delta.role = "assistant"` |
| 근거 | `choices: []` + `verification: {"kind": "evidence", ...}` | 컬렉션을 쓴 요청에서 본문보다 먼저 1회 (§2.6) |
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

검증 중단 청크(§4 fail-open·언어, ✅): `{"kind": "unverified", "verified": false, "reason": "judge_error" | "judge_timeout" | "non_english", "sentence_idx": n, "released_at_verdict": m}`.
요청당 한 번만 나가고, 이후 본문은 판정 없이 그대로 전달된다(hold면 보류 중이던 본문도 즉시 내보냄). `action=cut`이어도 끊지 않는다.

### 2.3 비스트리밍 응답 ✅
업스트림 응답을 받은 뒤 전체를 문장으로 나눠 판정하고, 응답 JSON에 `verification`을 추가한다.
```jsonc
"verification": {"mode": "hold", "action": "cut", "stride": 0, "threshold": 3.72, "prepare_ms": 3.1, "cut": true,
                 "verified": true, "sentences": [ {판정 객체}, ... ]}   // 검증 중단 시 "verified": false, "reason": ...
```
비스트리밍은 문장 끝에서만 판정하므로 문장 끝 임계값(설정 파일 `thresholds.end`)을 쓴다.
`hold + cut`이면 본문을 환각 문장 직전까지 자르고 `finish_reason = content_filter`.
그 밖의 조합에서는 본문을 바꾸지 않는다. `cut`이면 판정을 첫 환각 문장에서 멈춘다.

### 2.4 `GET /health` ✅
`{"ok": true, "judge": "<판정기 클래스>", "input_mode": "retrieval|full|combo", "mode": "...", "action": "...", "stride": 8, "stride_unit": "cl100k_base", "threshold": 2.7385, "post_check": {"available": true, "default": false, "model": "...", "votes": 3, "threshold": 0.5}}`

### 2.5 문서 등록 API ✅
| 요청 | 동작 |
|---|---|
| `POST /v1/collections/{id}/documents` | `{"text", "title"?, "doc_id"?}` 또는 `{"documents": [...]}`. 조각으로 나누고 임베딩해 저장, 같은 `doc_id`면 교체. 응답 `data: [{doc_id, title, n_chunks, n_chars}]` |
| `GET /v1/collections/{id}/documents` | 등록된 문서 목록 `{doc_id, title, n_chunks, n_chars, created}` |
| `DELETE /v1/collections/{id}/documents/{doc_id}` | 문서 삭제. 없으면 404 |
| `GET /v1/collections` | 컬렉션 목록 `{collection, n_docs}` |

- 저장소는 `Store` 인터페이스(`hallugate/store.py`)로 감싼다. 처음 구현 `SQLiteStore`: SQLite 파일 하나(`COLLECTION_DB`) + 조각 임베딩(float32), 코사인 유사도 전수 검색, 컬렉션별 행렬을 메모리에 캐시(문서 변경 시 무효화). 규모가 커지면 같은 인터페이스로 벡터 DB로 바꾼다.
- 조각: `COLLECTION_CHUNK_WORDS`(120)단어, `COLLECTION_OVERLAP`(20)단어 겹침. 판정기의 40단어 조각과는 별개다(판정기는 받은 근거를 다시 자기 방식으로 나눈다).
- 임베더: `COLLECTION_EMBEDDER`, 비우면 판정기(retrieval 방식)의 임베더를 같이 쓰고 없으면 all-MiniLM-L6-v2.
- 접근 토큰: `ADMIN_TOKEN`을 정하면 문서 API는 `Authorization: Bearer <ADMIN_TOKEN>`이 있어야 한다(401). 비우면 토큰 없이 열린다(로컬 개발용, `/health`의 `collections_auth`로 확인). 채팅 요청의 컬렉션 사용에는 토큰이 필요 없다.
- `COLLECTION_DB`를 빈 값으로 두면 컬렉션 기능이 꺼지고 관련 요청은 503.

### 2.6 프롬프트 삽입 ✅
- `verification.collection`이 있고 요청에 근거가 없으면, **마지막 사용자 메시지**로 컬렉션을 검색해 조각 `COLLECTION_TOP_K`(4)개를 고른다. 없는 컬렉션 → 404.
- 맨 앞에 시스템 메시지를 하나 더한다: 영어 지시("Answer using only the reference documents below. If they do not contain the answer, say that you do not know.") + `<documents>` 블록(`[1] 조각 …`, 유사도 순). 서비스가 보낸 메시지는 바꾸지 않는다.
- **판정 근거 = 이 조각들을 빈 줄로 이은 텍스트**(LLM에 넣은 근거 = 판정 근거, A9). retrieval 방식 판정기는 이 안에서만 조각을 고르고, full 방식은 이것 전체를 본다.
- 사용한 조각 목록은 스트리밍이면 본문보다 먼저 근거 청크 1회 `{"kind": "evidence", "collection", "retrieve_ms", "evidence": [{doc_id, title, chunk_idx, text, score}]}`, 비스트리밍이면 `verification.collection`·`verification.evidence`로 싣는다. 응답 헤더 `x-verification-evidence`(조각 수).
  (처음 계획은 판정 청크마다 싣는 것이었으나, 요청 안에서 근거가 바뀌지 않으므로 한 번만 보낸다.)
- `mode=off`여도 검색·삽입은 한다(RAG만 쓰고 검증은 끄는 경우). 질문이 비어 조각을 못 찾으면 삽입 없이 `mode=off`.
- 측정 기록: `collection`, `retrieve_ms`, `n_evidence`.

### 2.7 업스트림
- 📋 `UPSTREAM_PROVIDER=openai|anthropic`. OpenAI 형식은 그대로 통과, Claude는 Messages API로 변환해 호출하고 응답을 OpenAI 청크 형식으로 되돌린다.
- ✅ 스트리밍의 `delta.tool_calls`·`function_call`·`refusal`은 판정 없이 도착 즉시 그대로 통과한다(본문과 같은 청크에 있으면 분리해 보냄).
  `usage` 청크(`stream_options.include_usage`)는 종료 청크 뒤, `[DONE]` 앞에 보낸다. 업스트림 `finish_reason`(예: `tool_calls`)을 유지한다.

### 2.8 데모 화면 `GET /demo` ✅
발표·시연용 웹 페이지 한 장(`hallugate/static/demo.html`, 외부 리소스 없음 — 인터넷 없이 동작). 게이트웨이와 같은 주소에서 열리고, 일반 클라이언트처럼 `POST /v1/chat/completions`(스트리밍)만 호출한다. 서버 동작은 바꾸지 않는다.
- 입력: 근거 문서(또는 등록된 컬렉션), 질문, 모델 이름, 검증 옵션(`mode`·`action`·`stride`·`post_check`). 예제 시나리오(멀쩡한 답, 근거와 모순, 근거에 없는 내용)를 고르면 입력이 채워진다
- **가짜 LLM 응답**: 칸을 채우면 요청에 `mock_response`·`mock_token_ms`를 넣는다(가짜 LLM `mock_upstream.py` 전용). 실제 LLM에 연결했을 때는 비워 둔다(실제 API는 모르는 필드를 거부함)
- 출력: 본문이 흘러나오는 대로 표시하고, 판정 청크가 오면 해당 구간(`start`~`upto`)을 색으로 칠한다(문장 판정: 안전/환각, 앞부분 판정: 문장 도중 경보 표시). 2단 검증 결과는 문장별 표시와 응답 점수로, 근거 청크는 사용한 조각 목록으로, 검증 중단은 알림으로 보인다. 첫 글자까지 시간·전체 시간·판정 수·판정 시간을 함께 보인다
- `DEMO=0`이면 404 (운영에서 끄기)

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
    default_stride: int                                   # 설정 파일 기본 재판정 간격
    def threshold_for(self, stride) -> float              # 간격별 임계값
    def prepare(self, reference: str) -> RefIndex         # 요청당 1회: 근거 분할·임베딩
    def judge(self, ref, text, prior='', threshold=None) -> Verdict(score, hallucinated, judge_ms)
    #   text  = 판정할 문장 또는 문장 앞부분(앞뒤 공백 제거)
    #   prior = 응답에서 그 문장 앞까지의 원문 (full 방식만 사용)
```

| 구현 | 용도 |
|---|---|
| `StudentJudge` | 운영. 입력 방식은 설정 파일이 정한다(아래). 사양은 SSOT §4 |
| `LexicalJudge` | 테스트·데모 전용. 근거에 없는 단어 비율. **판별력 없음, 측정에 사용 금지** |

**판정기 폴더와 설정 파일** ✅ — `JUDGE_MODEL_DIR`(기본 `judge_base`)의 가중치·토크나이저와 `judge_config.json`을 읽는다(`load_judge`).
판정기를 바꿀 때는 폴더만 바꾸면 되고 코드는 바꾸지 않는다(모델 v1.0 F1 base ↔ I2 후보).

| 설정 파일 키 | 쓰임 |
|---|---|
| `input_mode` 또는 `input.retrieval` | `retrieval`(F1 base): 40단어 조각 → 임베딩 top-k(원문 순서) → [근거 조각, 문장 앞부분]. `full`(I2 노트북 형식, `input.retrieval: null`): [근거 전체, 응답 처음부터 판정 지점까지], 근거 검색 없음 |
| `embedder`, `k`, `chunk_words`, `max_len` | 입력 구성 (`truncation=only_first`) |
| `default.kind` | 기본 재판정 간격(`"8"` → 8토큰, `"end"` → 문장 끝만) |
| `thresholds[간격][default.threshold]`, `threshold` | 간격별 임계값(예: `thresholds["8"]["svc10_orig"]` = 3.969). 해당 간격이 없으면 `threshold` |

**조합 판정기 (`ComboJudge`, 배포 판정기 I3)** ✅ — 판정기 폴더의 `judge_config.json`이 `"type": "combo"`이면 하위 판정기 여러 개를 불러 점수를 합친다(SSOT §4).

```jsonc
{"type": "combo", "version": "1.0",
 "members": {"F": "f1_base", "I": "i2"},                 // 하위 판정기 폴더(조합 폴더 기준 상대 경로 또는 절대 경로), 각자 자기 judge_config.json을 가짐
 "standardize": {"F": {"mean": -3.679, "std": 2.187}, "I": {"mean": -4.456, "std": 2.401}},
 "default": {"kind": "8", "threshold": "svcB"},
 "thresholds": {"8": {"svcB": 2.7385, "svc10_orig": 2.9916}}, "threshold": 2.7385}
```
- 점수 = 하위 판정기 logit을 `(score − mean) / std`로 표준화한 값의 **평균**(클수록 환각). 노트북(I3·L3 `combine`)과 같은 식이다.
- 하위 판정기는 요청마다 각자 근거를 준비하고(`prepare`), 판정 1회에서 **동시에** 실행한다(스레드 + GPU면 판정기마다 별도 CUDA 스트림, L3와 같은 방식). `judge_ms`는 두 판정을 포함한 벽시계 시간이다.
- 하위 판정기의 임계값·재판정 간격 설정은 쓰지 않고, 조합 설정의 `thresholds`·`threshold`·`default`만 쓴다. `JUDGE_THRESHOLD`·`JUDGE_STRIDE`는 조합에 적용되고, `JUDGE_EMBEDDER`·`JUDGE_K` 등 입력 구성 변수는 하위 판정기에 적용하지 않는다(각자 설정 파일).
- 조합의 문장 끝(`end`) 임계값은 보정하지 않았으므로, 설정에 없으면 `threshold`(8토큰 서비스 임계값 B)를 쓴다.
- `/health`의 `judge`는 `ComboJudge`, `input_mode`는 `combo`. 컬렉션 검색 임베더를 비우면 retrieval 방식 하위 판정기의 임베더를 같이 쓴다.

- 환경 변수(`JUDGE_EMBEDDER`·`JUDGE_K`·`JUDGE_MAX_LEN`·`JUDGE_CHUNK_WORDS`·`JUDGE_THRESHOLD`·`JUDGE_STRIDE`)를 지정하면 설정 파일보다 우선한다. `JUDGE_THRESHOLD`를 지정하면 간격별 임계값 대신 그 값 하나를 쓴다.
- `hallucinated = score >= threshold_for(요청의 stride)`.
- ModernBERT 판정기는 `attn_implementation=sdpa`, `reference_compile=False`(설정에 넣음)로 올린다.
- 📋 `Judge.judge_batch()`: 여러 판정을 한 번에 처리.
- ✅ 영어가 아닌 텍스트(글자 3개 이상 중 라틴 문자 80% 미만)를 판정하게 되면 그 요청의 검증을 멈추고 `verified: false`(`non_english`).
- ✅ fail-open: 근거 준비·판정이 예외를 내거나 판정 1회가 `JUDGE_TIMEOUT_MS`(기본 2,000ms)를 넘으면 그 요청의 검증을 멈추고 본문을 그대로 전달, `verified: false`(`judge_error`·`judge_timeout`). 측정 기록에 `verified`, `unverified_reason`을 남긴다.
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
문장이 끝나기를 기다리지 않고, 현재 문장의 **tiktoken cl100k 토큰**이 문장 시작부터 s개, 2s개, … 확정될 때마다
그 토큰 끝까지의 **문장 앞부분**을 판정한다(학습·보정과 같은 지점, SSOT §4). 문장이 완성되면 그 문장 전체도 판정한다.
- 토큰 수는 현재 문장 텍스트를 토큰화해 센다. 마지막 토큰은 다음 글자가 와야 확정되므로 그때 판정한다.
- 한 조각이 s의 배수를 여러 개 넘으면 마지막 배수 하나만 판정한다.
- 문장 경계 토큰은 학습(응답 전체 토큰화)과 미세하게 다를 수 있다(앞 공백 처리). 고정 예제 일치는 문장 단위로 확인(A11).
- `STRIDE_TOKENIZER=words`는 네트워크 없는 테스트용 대체 단위다.

| mode | 동작 |
|---|---|
| `pass` | 본문은 즉시 전달. 앞부분 판정과 문장 판정을 백그라운드로 실행하고 결과를 **발생 순서대로** 전달 → 문장 도중에도 경보 가능 |
| `hold` | 본문을 보류하다가, 앞부분 판정이 **환각 아님**이면 그 시점까지 쌓인 본문을 내보낸다. 문장 완성 시 문장 판정 후 나머지를 내보낸다. 보류 지연이 "문장 길이"에서 "s 조각 + 판정 시간"으로 줄어든다 |

- 환각 판정 시 동작은 `action`을 따른다. `hold+cut`이면 환각으로 판정된 앞부분의 보류 중인 글자는 나가지 않는다.
- 불변식 G2(본문 보존)는 그대로 유지된다.

- 📋 S4: 위험도에 따라 재판정 간격과 보류 여부를 바꾸는 적응형 정책.
- ✅ 기본 정책(SSOT §7 10/28 A6): `VERIFY_MODE=hold`, 재판정 간격 8토큰(판정기 설정 파일), `VERIFY_ACTION=flag`. `cut`은 기본값에서 제외.

### 5.2 재생성 (`regenerate`) 📋
hold에서 환각으로 판정된 문장은 아직 사용자에게 나가지 않았으므로 버리고, "근거만 사용" 지시와 함께 그 지점부터 이어 쓰기를 **1회** 재요청한다.
- 재생성 여부는 표시용보다 보수적인 임계값(예: cal-L 멀쩡한 응답 5%)으로 정한다.
- 재생성한 문장도 다시 판정하고, 또 환각이면 표시만(flag) 한다.
- 해소율·추가 지연을 측정 기록에 남긴다.
- 불변식 G2는 "사용자에게 전달된 본문 = 최종 업스트림 본문 중 버리지 않은 부분"으로 해석한다(버린 문장은 전달되지 않음).

### 5.3 응답 끝 2단 검증 (`post_check`) 🧪 실험 기능

경량 판정기(1단)는 스트리밍 중 빠르게 판정하지만 환각을 많이 놓친다(SSOT §5 X2: FaithBench 탐지율 11.8%). 2단은 **응답이 끝난 뒤 강한 검증기(Claude)가 응답 전체를 근거와 다시 대조해** 결과를 한 번 보낸다.
J1은 미달, J2(탐지 최대화)는 보류 중이므로 **기본값은 꺼짐**이고 검증기 설정은 임시값이다. J2가 채택되면 J2가 고른 검증기·임계값으로 바꾼다(SSOT §6 J2).

- **켜기**: 요청 `verification.post_check: true` 또는 서버 `POST_CHECK=1`. 근거가 없어 `mode=off`이면 하지 않는다. 검증기가 준비되지 않았으면(`ANTHROPIC_API_KEY` 없음) 결과 청크 `verified: false, reason: "unavailable"`
- **검증기 `ClaudePostChecker`**: 응답 전체를 1단과 같은 문장 분할(`IncrementalSplitter`)로 나눠 번호를 붙이고, 근거와 함께 `POST_CHECK_MODEL`(기본 `claude-sonnet-5-5`)에 **`POST_CHECK_VOTES`번(기본 3) 동시에** 문장별 S/H/A 판정을 요청한다(J2 `sonnet_H`와 같은 지시문·파싱)
  - 응답 점수 `score` = 판정 중 H 문장이 하나라도 있다고 본 비율. `flagged = score ≥ POST_CHECK_THRESHOLD`(기본 0.5 = 과반)
  - 문장마다 `h_votes`(H로 본 판정 수)와 `flagged = h_votes / 유효 판정 수 ≥ 임계값`, 처음 H로 본 판정의 `phrase`·`reason`
  - 파싱에 실패한 판정은 H 없음으로 센다(J1·J2와 같음). 모든 판정이 실패하면 `verified: false, reason: "error"`
- **전달 시점**: 스트리밍은 종료 청크(`finish_reason`) **뒤**, `usage` 청크와 `[DONE]` 앞에 판정 청크 1회. 본문은 이미 다 나갔으므로 **본문은 바꾸지 않는다**(G2 그대로). 비스트리밍은 응답 `verification.post_check`
  ```jsonc
  {"kind": "post_check", "verified": true, "model": "claude-sonnet-5-5", "votes": 3, "valid_votes": 3,
   "score": 0.667, "threshold": 0.5, "flagged": true, "check_ms": 2310.4,
   "sentences": [{"sentence_idx": 1, "start": 42, "h_votes": 2, "flagged": true, "phrase": "...", "reason": "..."}]}
  ```
- **하지 않는 경우**: `cut`으로 응답이 끊겼으면 `verified: false, reason: "cut"`. 판정할 문장이 없으면 `reason: "empty"`
- **fail-open**: `POST_CHECK_TIMEOUT_MS`(기본 30,000)를 넘거나 API 오류면 `verified: false, reason: "timeout" | "error"`. 1단 결과와 본문에는 영향 없음
- 1단 점수로 2단에 보낼 응답을 거르는 기능(J1-3)은 아직 없다 — 켜면 모든 응답을 검증한다
- 측정 기록 `post_check`(결과 청크와 같은 내용)
- 비용·지연: 응답마다 검증 모델 호출 `POST_CHECK_VOTES`번(실시간 단가). 지연은 응답 끝 이후에만 생긴다

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
| `JUDGE_MODEL_DIR` | `judge_base` | 판정기 폴더 (가중치 + `judge_config.json`) |
| `JUDGE_EMBEDDER` | 설정 파일 | 근거 검색 임베딩 (retrieval 방식) |
| `JUDGE_K` / `JUDGE_MAX_LEN` / `JUDGE_CHUNK_WORDS` | 설정 파일 | 판정기 입력 구성 |
| `JUDGE_THRESHOLD` | 설정 파일(간격별) | 지정하면 모든 간격에 이 logit 임계값 |
| `JUDGE_STRIDE` | 설정 파일(F1 base 8) | 앞부분 재판정 간격, 토큰 (0 = 문장 끝에서만 판정) |
| `STRIDE_TOKENIZER` | `cl100k_base` | 재판정 간격 토큰 단위 (`words` = 테스트용) |
| `JUDGE_TIMEOUT_MS` | 2000 | 판정 1회 시간 제한, 넘으면 fail-open |
| `JUDGE_DEVICE` | 자동 | `cuda` / `cpu` |
| `VERIFY_MODE` / `VERIFY_ACTION` | `hold` / `flag` | 요청에 지정이 없을 때 기본 정책 |
| `COLLECTION_DB` | `collections.sqlite3` | 컬렉션 저장소 파일. 빈 값이면 컬렉션 기능 끔 |
| `COLLECTION_EMBEDDER` | 판정기 임베더 | 컬렉션 검색 임베더 |
| `COLLECTION_TOP_K` / `COLLECTION_CHUNK_WORDS` / `COLLECTION_OVERLAP` | 4 / 120 / 20 | 검색 조각 수, 조각 크기·겹침(단어) |
| `ADMIN_TOKEN` | (빈 값) | 문서 등록 API 접근 토큰. 비우면 열림 |
| `POST_CHECK` | `0` | `1`이면 요청에 지정이 없을 때 2단 검증을 켠다 (§5.3) |
| `POST_CHECK_MODEL` / `POST_CHECK_VOTES` / `POST_CHECK_THRESHOLD` | `claude-sonnet-5-5` / 3 / 0.5 | 2단 검증 모델, 판정 횟수, 응답 표시 임계값(H로 본 판정 비율) |
| `POST_CHECK_TIMEOUT_MS` | 30000 | 2단 검증 시간 제한, 넘으면 `verified: false` |
| `ANTHROPIC_API_KEY` | (빈 값) | 2단 검증용. 없으면 2단 검증 불가(`unavailable`) |
| `DEMO` | `1` | `0`이면 데모 화면(`/demo`)을 끈다 (§2.8) |
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
| A9 | 컬렉션 근거로 생성·검증할 때 본문 보존 100%, LLM에 넣은 근거 = 판정 근거 | ✅ 테스트 (가짜 LLM, pass·hold) |
| A10 | 스트리밍의 `tool_calls`·`usage` 청크 보존 | ✅ 테스트 (가짜 LLM) |
| A11 | 고정 예제(`parity_fixture.json`) 점수 일치 \|Δlogit\| ≤ 0.2 | ✅ F1 base 50개 최대 0.000016 (맥 CPU, `load_judge` 경로, `JUDGE_PARITY_DIR` 테스트) |
| A12 | 판정기 장애·시간 초과·비영어에서 본문 보존 100%, `verified: false` 1회 | ✅ 테스트 |
| A13 | 조합 판정기: 점수 = 하위 판정기 표준화 평균(노트북 `combine`과 같은 값), 간격별 임계값 B 적용, 게이트웨이에서 그대로 동작 | ✅ 테스트(작은 임의 모델). 실제 가중치 점수 일치는 가중치 확보 후 |
| A14 | 2단 검증: 켜면 응답 끝에 결과 청크 1회(종료 청크 뒤, `[DONE]` 앞), 본문 보존 100%, 검증기 오류·시간 초과·미설정에서 `verified: false`로 본문 영향 없음, 끄면 아무것도 안 보냄 | ✅ 테스트(가짜 Claude 클라이언트). 탐지 성능은 J2 |

## 9. 로드맵

| 단계 | 내용 |
|---|---|
| S3 | ~~학습 판정기로 A5·A6 측정, CPU 판정 속도 측정~~ ✅, 중간보고 |
| F | ~~조합 판정기(F1 base + I2) 게이트웨이 연결~~ ✅(11/23, A13). 모델 v1.0 동결: F1 base 판정기, `judge_config.json` v1.0(8토큰·서비스 10% 임계값, 재생성용 임계값), `judge_batch`, 고정 예제 일치 |
| S4 | ~~재판정 간격 토큰 단위, `tool_calls`·`usage` 통과(A10), 판정기 설정 파일, 언어 감지, fail-open~~ ✅(11/05), ~~컬렉션 API·저장소·검색·프롬프트 삽입·`evidence`(§2.5·2.6, A9), 문서 API 접근 토큰~~ ✅(11/05), Claude Messages API 변환(§2.7), 재생성(§5.2), 판정 묶음 처리, 배포 후 실제 LLM 속도로 종단 간 재측정 |
| S5 | RAGTruth test 1회 개방 최종 평가, 기존 도구의 고정 설정(NeMo Guardrails의 chunk 200 등)과 비교 |
| S6 | ~~데모 화면(판정 시각화, `/demo`)~~ ✅(11/23), 시연 시나리오·녹화, 보고서, SOFTCON |
| 운영 | 관리형 베타(§10): 인증·사용자 분리 → 사용량 제한·업스트림 허용 목록 → 원문 없는 기록·관리자 통계 → Modal 배포·운영 문서 → 실제 측정(H8) |

## 10. 운영 모드 (관리형 베타) 🚧

**목적**: 게이트웨이를 직접 운영하며 지연·비용·장애를 실제 트래픽으로 다뤄 본다(SSOT §7 11/23). 초대한 사용자에게만 API 키를 발급하는 **관리형 베타**이며, 결제·셀프 가입은 하지 않는다.
**대상**: **영어 RAG 응답만** 검증한다. 판정기가 영어 데이터로만 학습됐기 때문이다(§4 언어 감지: 영어가 아니면 `verified: false, reason: "non_english"`로 본문은 그대로 전달).

`HOSTED=1`일 때만 이 절의 규칙이 적용된다. 꺼져 있으면(기본) 지금까지의 셀프호스팅 동작이 그대로다(§1~9, 기존 테스트 그대로 통과).

### 10.1 인증 — 게이트웨이 API 키 ✅
- 모든 `/v1/*` 요청은 헤더 **`x-hallugate-key: hg_<키 ID>_<비밀>`**이 있어야 한다. 없거나 틀리거나 폐기된 키 → 401 `authentication_error`. `Authorization` 헤더는 쓰지 않는다(사용자의 LLM 키를 담아 업스트림에 넘기는 용도, §10.3)
  - OpenAI SDK: `OpenAI(base_url=..., api_key=<LLM 키>, default_headers={"x-hallugate-key": "hg_..."})`
- 키는 **해시(SHA-256)만** 저장한다(`KEYS_DB`, SQLite): 키 ID, 해시, 이름(사용자 메모), 만든 시각, 폐기 시각, 한도(§10.4). 비밀 원문은 발급할 때 한 번만 보여 준다
- 발급·조회·폐기는 관리자 스크립트로만 한다: `python scripts/keys.py create --name ...` / `list` / `revoke <키 ID>`. 셀프 가입 없음
- `/demo`(§2.8): 운영 모드에서는 화면에서 게이트웨이 키·LLM 키·업스트림을 입력받아 요청 헤더로 보낸다. 키는 그 브라우저 탭의 저장소(sessionStorage)에만 두고 서버에 저장하지 않는다. 가짜 LLM은 배포하지 않으므로 실제 업스트림으로만 동작한다
- `/health`는 인증 없이 열지만 운영 모드에서는 `{"ok": true}`와 버전만 준다(판정기 내부 값 숨김). `/admin/*`는 `ADMIN_TOKEN`(`Authorization: Bearer`)이 있어야 한다

### 10.2 사용자 분리 — 컬렉션 ✅
- 컬렉션은 **키 ID(사용자) 단위로 분리**한다. 같은 이름 `docs`라도 사용자마다 다른 컬렉션이다 (저장소 안의 이름은 `<키 ID>/<컬렉션>`, 사용자에게는 앞부분을 뗀 이름만 보인다)
- 남의 컬렉션은 목록·검색·문서 조회·삭제 어디서도 보이지 않는다(없는 것과 같은 404). 채팅 요청의 `verification.collection`도 자기 컬렉션만 찾는다
- 운영 모드에서 문서 API는 `ADMIN_TOKEN`이 아니라 사용자 키로 쓴다(사용자가 자기 문서를 관리)
- 한도: 사용자당 문서 수 `MAX_DOCS`(기본 200), 문서 하나 `MAX_DOC_CHARS`(기본 200,000자). 넘으면 413

### 10.3 업스트림 — 허용 목록
- 헤더 **`x-upstream: openai | anthropic`**(기본 `openai`)로 고른다. 주소는 서버의 허용 목록에서만 정한다(`openai` → `https://api.openai.com/v1`, `anthropic` → `https://api.anthropic.com/v1`, Anthropic의 OpenAI SDK 호환 주소). **요청으로 임의 주소를 지정할 수 없다**(서버가 사용자 지정 주소를 호출하는 위험 방지). 목록에 없는 값 → 400
- `Authorization`(사용자의 LLM 키)은 그대로 넘기고 **기록하지 않는다**. 없으면 업스트림의 401을 그대로 돌려준다. 운영 모드에서는 서버의 `UPSTREAM_API_KEY`를 **절대 쓰지 않는다**(사용자 요청에 운영자 키로 과금되지 않게)
- Anthropic 호환 주소는 Anthropic이 시험·비교용으로 안내하는 계층이다. 스트리밍·`usage` 동작은 배포 후 실제로 확인하고 결과를 SSOT에 남긴다

### 10.4 사용량 제한
키마다 아래 한도를 둔다(키 생성 때 바꿀 수 있음, 기본값은 환경 변수). 넘으면 429 `rate_limit_error` + `retry-after` 헤더.

| 한도 | 기본값 | 단위 |
|---|---|---|
| `RATE_RPM` | 30 | 분당 채팅 요청 |
| `MAX_CONCURRENT` | 4 | 동시 스트리밍 요청 |
| `POST_CHECK_DAILY` | 50 | 하루(UTC) 2단 검증 요청 — 2단은 운영자 Claude 키로 과금되므로 |
| `MAX_REF_CHARS` | 50,000 | 요청 하나의 근거 문서 글자 수 (넘으면 413) |

- 분당·동시 한도는 메모리에서 센다(컨테이너 1대 전제, §10.7). 하루 2단 검증 수는 SQLite에 남겨 컨테이너가 다시 떠도 유지한다

### 10.5 기록·개인정보
- 운영 모드의 측정 기록(§6)에는 **원문을 남기지 않는다**: 근거 문서, 응답 문장, 질문, 근거 조각 텍스트, 2단 `phrase`·`reason`을 빼고 **키 ID, 시각, 길이, 점수, 판정 여부, 지연, 업스트림 종류, 상태 코드, 검증 중단 이유**만 남긴다. 사용자에게 보내는 응답 청크는 바뀌지 않는다
- 요청마다 원문 없는 JSON 한 줄을 표준 출력으로도 남긴다(플랫폼 로그)
- 보관: 측정 기록 `METRICS_RETENTION_DAYS`(기본 30)일 뒤 지운다. 컬렉션 문서는 사용자가 지울 때까지 보관하고, 키를 폐기하며 `--purge`를 주면 그 사용자의 컬렉션·기록을 모두 지운다
- 사용자의 LLM 키(`Authorization`)와 게이트웨이 키 비밀은 어디에도 기록하지 않는다

### 10.6 관측 — `GET /admin/stats` (`ADMIN_TOKEN`)
기간(`window`, 기본 24h) 동안: 요청 수(사용자별·업스트림별), 상태 코드별 수, 첫 글자 지연·전체 시간 p50/p95, 판정 시간 p95, 검증 중단 비율(이유별), 2단 검증 수와 추정 비용(응답의 토큰 사용량 × §5.3 모델 단가표), 429 수, **콜드 스타트**(컨테이너 시작 시각·예열 시간·시작 직후 첫 요청의 첫 글자 지연).

### 10.7 배포 — Modal (서버리스 GPU)
- 게이트웨이 전체(FastAPI + 조합 판정기)를 **GPU 컨테이너 하나**에 올린다(`deploy/modal_app.py`). 판정기를 따로 떼면 8토큰마다 네트워크 왕복이 생겨 실시간 조건(§4, A6)을 맞추기 어렵기 때문이다
- 컨테이너 **최소 0대·최대 1대**(SQLite 쓰기 하나, 메모리 한도 계산), 요청이 없으면 `SCALEDOWN_SECONDS`(기본 300초) 뒤 내려간다. 시연 날처럼 필요하면 최소 1대로 바꿔 콜드 스타트를 없앤다
- 저장: Modal Volume `/data`(키·컬렉션·사용량·측정 기록 SQLite), `/models`(판정기 가중치, 저장소에 넣지 않음). 비밀 값: Modal Secret(`ADMIN_TOKEN`, 2단 검증용 `ANTHROPIC_API_KEY`)
- 컨테이너가 뜨면 판정기를 불러 예열하고(§4) 예열 시간을 기록한다
- 배포는 `modal deploy deploy/modal_app.py`(수동). GitHub Actions 자동 배포는 저장소 CI가 동작할 때 붙인다
- 운영 문서 `docs/OPERATIONS.md`: 배포, 키 발급·폐기, 한도 변경, 통계 보기, 장애 대응(판정기 오류·업스트림 장애·콜드 스타트·비용 급증), 비용 확인

### 10.8 운영 설정 (환경 변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `HOSTED` | `0` | `1`이면 이 절의 규칙 적용 |
| `KEYS_DB` | `/data/hallugate.sqlite3` | 키·사용량 저장소 (운영 모드). 컬렉션은 `COLLECTION_DB` |
| `RATE_RPM` / `MAX_CONCURRENT` / `POST_CHECK_DAILY` | 30 / 4 / 50 | 키별 기본 한도 (§10.4) |
| `MAX_REF_CHARS` / `MAX_DOCS` / `MAX_DOC_CHARS` | 50,000 / 200 / 200,000 | 크기 한도 |
| `METRICS_RETENTION_DAYS` | 30 | 측정 기록 보관 일수 |
| `UPSTREAM_ALLOWLIST` | `openai=https://api.openai.com/v1,anthropic=https://api.anthropic.com/v1` | `x-upstream` 값 → 주소 |
| `SCALEDOWN_SECONDS` | 300 | 요청이 없을 때 컨테이너를 내리기까지 (Modal) |

### 10.9 수용 기준 (운영 모드)

| ID | 기준 | 상태 |
|---|---|---|
| H1 | 키 없음·틀림·폐기 → 401, 저장소에는 해시만 있음 | ✅ 테스트 (11/23) |
| H2 | 사용자 A가 B의 컬렉션을 목록·검색·조회·삭제·채팅 근거 어디서도 볼 수 없음 | ✅ 테스트 (11/23, 문서 수·크기 한도 포함) |
| H3 | 업스트림은 허용 목록 주소로만 호출, 운영자 `UPSTREAM_API_KEY`를 쓰지 않음, 사용자 LLM 키를 기록하지 않음 | 📋 |
| H4 | 분당·동시·하루 2단·크기 한도를 넘으면 429/413, 한도 안에서는 영향 없음 | 📋 |
| H5 | 운영 모드 측정 기록·표준 출력 로그에 원문(근거·문장·질문·조각·2단 이유)이 없음 | 📋 |
| H6 | `HOSTED=0`이면 기존 동작·테스트 그대로 | ✅ 기존 테스트 전부 통과 (단계마다 다시 확인) |
| H7 | `/admin/stats`가 §10.6 항목을 돌려줌, `ADMIN_TOKEN` 없으면 401 | 📋 |
| H8 | Modal 배포 후 실제 측정: 콜드 스타트 시간, 따뜻한 상태 첫 글자 지연·판정 p95, 하루 비용 (SSOT에 기록) | 📋 |
