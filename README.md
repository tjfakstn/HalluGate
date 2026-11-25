# HalluGate

**RAG 응답을 스트리밍 도중 문장 단위로 근거 문서와 대조하는 LLM 환각 검증 게이트웨이**
2026-2 아주대학교 소프트웨어학과 자기주도프로젝트 · 설만수 · 지도 조현석 교수

서비스와 LLM API 사이에 끼우는 프록시입니다. 서비스는 SDK의 `base_url`만 바꾸면 되고, 게이트웨이는 LLM이 흘려보내는 답을
문장마다 근거 문서와 대조해 환각을 **표시**하거나 **차단**합니다. OpenAI 형식과 Anthropic(Claude) 형식을 모두 받습니다.

```
서비스 ──(요청 + 근거 문서)──▶ HalluGate ──▶ LLM API (OpenAI · Claude)
       ◀──(본문 + 문장별 판정)──   │    ◀── 토큰 스트림
                                   └─ 문장 분할 → 경량 판정기 2개 조합 (GPU, 판정 1회 p95 약 50ms)
```

## 왜 만들었나

기존 스트리밍 검증 도구(NeMo Guardrails, Guardrails AI 등)는 정확도와 지연 사이의 균형이 **고정**되어 있습니다.
정확한 전용 판정기(MiniCheck-FlanT5-L)는 문장당 약 380ms라 실시간에 쓰기 어렵고, 빠른 방식은 부정확합니다.
이 프로젝트는 두 가지를 만듭니다.

1. **스트리밍을 막지 않는 속도의 경량 판정기**: 공개 데이터(RAGTruth)로 직접 학습. 문장이 끝나기 전에 앞부분도 다시 판정
2. **그 판정기를 서비스에 붙이는 게이트웨이**: 본문을 바꾸지 않고, 전달 정책(즉시 전달 / 판정 후 전달, 표시 / 차단)을 요청마다 고르게 함

## 주요 결과 (실측)

모든 수치는 실제로 실행해 얻은 값이며, 사전에 정한 기준과 함께 [SSOT.md](SSOT.md)에 기록되어 있습니다.

| 항목 | 결과 |
|---|---|
| **판정 지연** | 실서비스(Modal A10G, 실제 Claude 응답 9건) 요청별 판정 p95 **44~55ms** — 목표 68ms 이내. A100 두 판정기 동시 실행 p95 61.3ms |
| **탐지 (RAGTruth, 학습과 근거가 겹치지 않는 dev-L)** | 환각 응답 탐지율 **0.534**, 탐지 지연 중앙값 **9토큰**(첫 환각 뒤), 멀쩡한 응답 오표시 7.3% (F1 base 단독 0.495 / 14토큰) |
| **외부 사람 정답 (FaithBench)** | 응답 단위 AUROC **0.629** [0.559, 0.691] — 공개 판정기 HHEM-2.1과 비슷한 수준 |
| **최신 LLM 응답** | Claude 응답에서 멀쩡한 응답 오표시 4.5~6.6%. 단 환각 탐지율은 **14~19%로 낮음** (아래 한계 참고) |
| **근거 다이어트** (`trim`) | 긴 근거에서 관련 조각만 LLM에 보내 **입력 토큰 −47.6%**, 답의 정답성·충실성 비열등 (대조군 포함 실험) |
| **본문 보존** | 게이트웨이를 거쳐도 사용자에게 가는 본문이 원문과 100% 같음 (차단 제외, 테스트 100개) |
| **배포** | 콜드 스타트 원인을 분리 측정해 **5~16분 → 약 20초**로 단축 (GPU를 L4에서 A10G로). 쓸 때만 켜는 구조로 비용 관리 |

### 한계 (알고 있는 것)
- **최신 LLM의 환각을 많이 놓칩니다.** 놓친 것의 약 68%는 근거 단어를 그대로 쓰면서 내용을 미세하게 바꾼 직접 모순입니다(M1a). 합성 데이터·최신 LLM 응답 재학습(K1·K2)으로도 기준만큼 개선되지 않아 기록으로 남겼습니다
- 근거에서 **계산해 낸 수치**(예: 성장률로 역산한 작년 매출)도 근거에 없는 내용으로 보고 표시합니다
- **영어 응답만** 검증합니다. 다른 언어는 검증하지 않고 본문을 그대로 전달합니다
- RAGTruth test 분할 최종 평가(S5)는 아직 하지 않았습니다. 위 dev-L 수치는 개발용 평가 세트 기준입니다

## 쓰는 방법

### OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(base_url="https://<게이트웨이>/v1", api_key="sk-...",
                default_headers={"x-hallugate-key": "hg_...", "x-upstream": "openai"})   # 운영 모드
stream = client.chat.completions.create(
    model="gpt-4o-mini", stream=True,
    messages=[{"role": "user", "content": f"<context>{document}</context>\n\nHow did the company do in 2023?"}])
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")         # 본문은 평소와 같음
    v = (chunk.model_extra or {}).get("verification")         # 판정은 choices가 빈 별도 청크
    if v and v.get("hallucinated"):
        print("\n[환각 의심]", v["text"])
```

### Claude 공식 SDK (Anthropic Messages 형식)
```python
import anthropic

client = anthropic.Anthropic(base_url="https://<게이트웨이>", api_key="sk-ant-...",
                             default_headers={"x-hallugate-key": "hg_..."})
msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                             system=f"Answer using only the document.\n<context>{document}</context>",
                             messages=[{"role": "user", "content": "How did the company do in 2023?"}])
for s in msg.model_extra["verification"]["sentences"]:
    if s["hallucinated"]:
        print("[환각 의심]", s["text"])
```
스트리밍에서는 판정이 `event: verification` SSE 이벤트로 섞여 옵니다. 공식 SDK는 이 이벤트를 건너뛰므로 기존 코드는 그대로 동작합니다.

### 근거 문서와 옵션
근거는 `<context>…</context>` 태그(메시지나 `system` 안), `verification.context` 필드, 또는 미리 올린 **컬렉션**(`verification.collection`, 검색해서 프롬프트에 넣고 같은 조각으로 판정) 중 하나로 줍니다. 근거가 없으면 검증 없이 중계만 합니다.

| `verification` 옵션 | 동작 |
|---|---|
| `mode: pass` | 즉시 전달하고 판정은 뒤따라 보냄 (지연 거의 없음, 틀린 문장이 먼저 나갈 수 있음) |
| `mode: hold` (기본) | 판정을 통과한 만큼만 내보냄 (첫 글자 약 0.25초 추가, 환각이 나가기 전에 막을 수 있음) |
| `action: flag` / `cut` | 환각 문장 표시만 / 거기서 응답 중단 |
| `trim: true` | 근거 다이어트 — 관련 조각만 LLM에 보내 입력 토큰 절감 |
| `post_check: true` | 🧪 응답이 끝난 뒤 Claude로 한 번 더 검증 (실험 기능, 기본 꺼짐) |

전체 규칙은 [SPEC.md](SPEC.md)에 있습니다.

## 운영 모드 (관리형 베타)

`HOSTED=1`로 띄우면 여러 사용자가 함께 쓰는 서비스가 됩니다. Modal 서버리스 GPU에 배포해 두었고, 초대 베타 전입니다.

- **사용자 분리**: 게이트웨이 API 키(해시만 저장), 사용자별 컬렉션, 업스트림 허용 목록(OpenAI·Anthropic), 사용자 LLM 키는 저장하지 않고 전달만
- **한도·비용**: 분당·동시 요청 한도, 키별 월 예산(경고 또는 차단), 사용자 LLM 토큰·추정 비용 집계
- **기록**: 기본은 원문 없이 숫자만(30일). 사용자가 `x-hallugate-trace: 1`을 붙인 요청만 LLM 입력·근거·문장별 판정을 7일 보관해 본인만 조회
- **화면**: `/start`(연결 안내), `/console`(내 사용 현황·요청 기록), `/admin`(운영 통계·키 관리), `/demo`(체험)
- **비용 관리**: 평소에는 0대로 두고 `scripts/warm.py up/down`으로 필요할 때만 켬, 끌 시각이 지나면 자동으로 끔

배포·운영 절차는 [docs/OPERATIONS.md](docs/OPERATIONS.md)에 있습니다.

## 로컬에서 실행

```bash
pip install -r requirements.txt
python -m pytest -q                       # 네트워크·모델 없이 도는 테스트 (100개, 가중치 필요 1개 건너뜀)

# 가짜 LLM + 게이트웨이 (학습 판정기 없이 동작 확인: --lexical)
python mock_upstream.py --port 9000 &
UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 python serve.py --lexical --port 8000 &
# 브라우저에서 http://127.0.0.1:8000/demo

# 학습 판정기로 실행: judge_combo/에 f1_base/·i2/ 가중치와 configs/judge_combo_v1.json(→ judge_config.json)
JUDGE_MODEL_DIR=judge_combo UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 python serve.py --port 8000
```

학습된 가중치는 저장소에 포함하지 않습니다. 종단 간 측정은 Colab(T4)에서 `experiments/notebooks/s3_gateway_bench.ipynb`로 할 수 있습니다.

## 연구 방식

연구 과제라서 **실험의 공정성**을 코드 품질만큼 중요하게 다룹니다.
- 모든 실험은 결과를 보기 전에 **채택 기준을 먼저 정하고**(노트북 첫 셀), 미달이면 미달로 기록합니다. 채택하지 못한 실험도 기준·결과·이유와 함께 [SSOT.md](SSOT.md)에 남아 있습니다
- 최종 평가용 RAGTruth test 분할은 마지막 평가에서 한 번만 엽니다
- 학습과 근거 문서가 겹치지 않는 평가 세트(dev-L)를 따로 만들어 오염을 측정했습니다(R0)

## 저장소 구조

```
hallugate/              게이트웨이 패키지
  app.py                  FastAPI 프록시 (OpenAI /v1/chat/completions, Anthropic /v1/messages)
  splitter.py · pipeline.py · judge.py   문장 분할, 전달 정책, 판정기
  context.py · store.py · trim.py        근거 추출, 컬렉션 검색, 근거 다이어트
  keys.py · limits.py · metrics.py · traces.py   운영 모드 (키, 한도, 기록, 요청 추적)
  static/                 /demo · /start · /console · /admin 화면
deploy/modal_app.py     Modal 배포
scripts/                warm.py(켜기·끄기), keys.py(키 관리), bench.py(종단 간 측정)
mock_upstream.py        API 키 없이 시험하는 가짜 LLM (OpenAI·Claude 형식)
tests/                  pytest
experiments/
  builders/             노트북 생성 스크립트 (노트북은 여기서만 만든다)
  notebooks/            Colab 실험 노트북
docs/                   진행 기록, 운영 문서, 연구 설계, 주간 보고서, 분석 결과
SSOT.md · SPEC.md       확정된 수치·결정 / 게이트웨이 동작 명세
AGENTS.md · CLAUDE.md   AI 코딩 에이전트 작업 규칙
```

## 문서

| 문서 | 내용 |
|---|---|
| [SSOT.md](SSOT.md) | 데이터 분할, 판정기 사양, 모든 실측 수치, 사전 기준, 결정 기록, 일정 |
| [SPEC.md](SPEC.md) | API, 전달 정책, 불변식, 운영 모드, 수용 기준 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 시간 순 진행 기록과 현재 위치·다음 할 일 |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | 판정기 v1.0 사양·학습 데이터·성능·한계 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 배포, 사용자 관리, 관측, 장애 대응, 비용 |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | 스트리밍 조기 탐지 연구 설계 |
| [AGENTS.md](AGENTS.md) | 작업 규칙 (SPEC → 테스트 → 코드, 연구 공정성) |

## 다음 할 일

1. **최종 평가(S5)**: RAGTruth test 1회 개방, 공개 판정기·LLM 직접 판정과 비교
2. **탐지 성능 향상(J2)** 과 **자동 교정 실험(C1)**: 판정기가 표시한 문장을 LLM이 근거에 맞게 고쳐 쓰게 해 환각이 실제로 줄어드는지 검증
3. 보고서·데모 영상, SOFTCON (12월)

## 데이터·모델

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (학습·평가), [FaithBench](https://github.com/vectara/FaithBench) (외부 평가), [HaluEval](https://github.com/RUCAIBox/HaluEval), 기준 판정기 [MiniCheck](https://github.com/Liyan06/MiniCheck). 판정기 백본은 DeBERTa-v3-base와 ModernBERT-large입니다.
