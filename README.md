# hallugate

**RAG 응답을 스트리밍 중에 문장 단위로 검증하는 LLM 환각 검증 게이트웨이**
2026-2 아주대학교 소프트웨어학과 자기주도프로젝트 · 설만수 · 지도 조현석 교수

최종 갱신 2026-11-13

- **무엇**: 서비스와 LLM API 사이에 끼우는 OpenAI 호환 프록시입니다. 서비스는 `base_url`만 바꾸면 되고, 게이트웨이는 스트리밍되는 답변을 문장마다 근거 문서와 대조해 환각을 표시(`flag`)하거나 끊습니다(`cut`).
- **측정된 것**: 종단 간 본문 보존 100%, 게이트웨이 판정 p95 20.6ms(이전 판정기 student_vark, T4, [SSOT §5 S3](SSOT.md#s3--게이트웨이-종단-간-측정)). 배포 판정기(I3 조합)는 RAGTruth dev-L 탐지율 0.534·멀쩡한 응답 표시율 7.3%([§5 I3](SSOT.md#i3--요약-약점-개선-f1-base--i2-점수-조합-1111-i3-1234-충족--채택)), 최신 Claude 응답(devM) 탐지율 18.8%·표시율 6.6%(서비스 임계값 B, [§5 K2](SSOT.md#k2--최신-llm-실제-응답으로-f1-base-재학습-1113-k2-1-미달--미채택-k2-0k2-2k2-3-충족-임계값-b-채택-규칙-충족)).
- **알려진 한계**: 최신 LLM이 근거 단어를 그대로 쓰며 내용을 바꾸는 직접 모순을 대부분 놓칩니다(놓친 것의 68%, [§5 M1a](SSOT.md#m1a--m1-환각-탐지율이-낮은-원인-1111-해석-규칙-나-해당-놓친-진짜-환각의-68가-직접-모순-가-해당-없음)). 요약 과제가 가장 약하고(dev-L Summary 0.266, [§5 I3](SSOT.md#i3--요약-약점-개선-f1-base--i2-점수-조합-1111-i3-1234-충족--채택)), 조합 판정기는 아직 게이트웨이에 연결되지 않았습니다(지금 게이트웨이는 판정기 하나만 불러옴).

```
서비스 ──(요청 + 근거 문서)──▶ hallugate ──▶ LLM API (OpenAI 등)
       ◀──(본문 + 문장별 판정)──   │   ◀── 토큰 스트림
                                   └─ 문장 분할 → 소형 판정기(GPU)
```

## 왜 만드나

기존 스트리밍 검증 도구는 정확도와 지연의 균형이 **고정**되어 있습니다.
정확한 전용 판정기(MiniCheck)는 문장당 약 400ms라 실시간에 못 쓰고, 빠른 방식은 부정확합니다.
이 프로젝트는 두 가지를 만듭니다.
1. **빠르고 정확한 소형 판정기**: 기존 판정기를 그대로 얹지 않고 공개 데이터로 직접 학습
2. **검증 단위와 전달 정책을 조절하는 게이트웨이**: 그 트레이드오프를 요청·문장마다 바꿀 수 있게 함

## 현재 상태

| 항목 | 상태 | 근거 |
|---|---|---|
| 배포 판정기 | I3 조합: F1 base(DeBERTa-v3-base) + I2(ModernBERT-large, 근거 전체 2,048토큰) 점수 표준화 평균, 8토큰 재판정. 서비스 임계값 B 2.7385(11/13). A100 두 판정기 동시 실행 판정 1회 p95 61.3ms | [§5 I3](SSOT.md#i3--요약-약점-개선-f1-base--i2-점수-조합-1111-i3-1234-충족--채택), [§5 K2](SSOT.md#k2--최신-llm-실제-응답으로-f1-base-재학습-1113-k2-1-미달--미채택-k2-0k2-2k2-3-충족-임계값-b-채택-규칙-충족) |
| 게이트웨이 | 본문 보존 100%, 게이트웨이 안 AUROC 0.853, 판정 p95 20.6ms(S3). `JUDGE_MODEL_DIR` 폴더의 `judge_config.json`으로 판정기 하나를 불러옴 — **조합 판정기는 연결 전** | [§5 S3](SSOT.md#s3--게이트웨이-종단-간-측정), [SPEC §4](SPEC.md) |
| 평가 세트 | 근거 문서가 학습과 겹치지 않는 cal-L(보정)·dev-L(주 평가, 환각 759개), 최신 LLM 응답 calM·devM | [§2](SSOT.md#2-데이터), [§5 R0](SSOT.md#r0--근거-문서가-겹치지-않는-평가-세트dev-l와-오염-점검-107), [§5 K2](SSOT.md#k2--최신-llm-실제-응답으로-f1-base-재학습-1113-k2-1-미달--미채택-k2-0k2-2k2-3-충족-임계값-b-채택-규칙-충족) |
| 최근 실험 | K1(합성 모순 재학습)·K2(최신 LLM 실제 응답 재학습) 모두 주 기준 미달 → **미채택**. K2 사전 규칙에 따라 서비스 임계값을 B로 바꿔 최신 LLM 탐지율 14.1% → 18.8% | [§5 K2](SSOT.md#k2--최신-llm-실제-응답으로-f1-base-재학습-1113-k2-1-미달--미채택-k2-0k2-2k2-3-충족-임계값-b-채택-규칙-충족), [§5 K1](SSOT.md#k1--직접-모순-합성-데이터로-f1-base-재학습-1112-k1-1-미달--미채택-k1-0k1-2k1-3-충족), [§7](SSOT.md#7-결정-기록) |
| 남은 문제 | 조합 판정기 게이트웨이 연결, `judge_config` v1.0·모델 카드, 사람 점검(30개), 최신 LLM 직접 모순 미탐, 요약 약점, Claude 업스트림 변환·재생성, 최종 RAGTruth test 1회 평가 | [PROGRESS 현재 위치](docs/PROGRESS.md) |

모든 수치와 결정 이력은 **[SSOT.md](SSOT.md)**, 동작 명세는 **[SPEC.md](SPEC.md)** 에 있습니다.

## 빠른 시작

Python 3.13 이상. 의존성의 정본은 `pyproject.toml`입니다(`requirements.txt`는 `-e .[judge,dev]`를 가리키는 단축 경로).

```bash
git clone https://github.com/tjfakstn/HalluGate.git && cd HalluGate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # 게이트웨이 경량 설치 (torch 없음)
python -m pytest -q                # 경량 테스트 38개 (모델 테스트 3개는 기본 실행에서 제외)
```

가짜 LLM과 대체 판정기(`--lexical`)로 파이프라인을 확인합니다. 대체 판정기는 단어 겹침만 보는 시험용이라 판별력 수치로 쓰지 않습니다.

```bash
python mock_upstream.py --port 9000 &
UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 python serve.py --lexical --port 8000 &
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mock", "stream": false,
  "messages": [{"role": "user", "content": "When was the company founded?"}],
  "verification": {"context": "The company was founded in 1998 in Seoul."},
  "mock_response": "The company was founded in 1998 in Seoul. It has 500 employees in Busan."}'
# → verification.sentences에 문장별 score·hallucinated (두 번째 문장 true)
```

경량 설치에서는 컬렉션 기능(문서 등록·검색)이 꺼진 채 시작합니다(SPEC §2.5). 학습 판정기와 컬렉션을 쓰려면:

```bash
pip install -e ".[judge]"                       # torch, transformers, sentence-transformers
python -m pytest -q -m "model or not model"     # 전체 40개 + 가중치 일치 테스트 1개(JUDGE_PARITY_DIR 지정 시)
JUDGE_MODEL_DIR=path/to/judge_base UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 python serve.py --port 8000
```

판정기 가중치(`judge_base/` 등)는 저장소에 없습니다. 모델 테스트는 torch와 가중치가 필요해 CI에서 돌리지 않고, CI(`.github/workflows/ci.yml`)는 경량 테스트만 Python 3.13·3.14에서 돌립니다.

종단 간 측정(RAGTruth 필요)과 Colab(T4) 실행:

```bash
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git
python scripts/make_replay_set.py --ragtruth RAGTruth/dataset --per-task 20
python scripts/bench.py --metrics gateway_metrics.jsonl
```

Colab에서는 `experiments/notebooks/s3_gateway_bench.ipynb`를 그대로 실행하면 됩니다.

### 서비스에서 쓰기

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-...")
stream = client.chat.completions.create(
    model="gpt-4o-mini", stream=True,
    messages=[{"role": "user", "content": "질문"}],
    extra_body={"verification": {"context": 근거문서, "mode": "hold", "action": "cut"}})
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
    if getattr(chunk, "verification", None):
        print("\n[판정]", chunk.verification)
```

| 정책 | 동작 |
|---|---|
| `pass` | 즉시 전달, 판정은 뒤따라 옴 (지연 0, 틀린 문장이 먼저 나갈 수 있음) |
| `hold` | 문장 단위로 보류 후 판정·전달 (누출 0, 문장 생성 시간만큼 지연) |
| `flag` / `cut` | 환각 문장 표시만 / 응답 중단 (`finish_reason=content_filter`) |

## 실험 관리 방식

- 평가 기준은 결과를 보기 전에 노트북 첫 셀과 [SSOT §6](SSOT.md#6-사전-정의-기준-결과-보기-전에-확정한-것)에 고정합니다.
- 노트북은 `experiments/builders/build_*.py`로 생성하고, `.ipynb`를 직접 고치지 않습니다.
- RAGTruth test 분할은 봉인해 두고 최종 평가에서 한 번만 엽니다.
- 기준에 미달하면 기준을 바꾸지 않고 미채택으로 기록합니다([SSOT §7](SSOT.md#7-결정-기록)). 상세 규칙은 [AGENTS.md](AGENTS.md).

## 저장소 구조

```
hallugate/            게이트웨이 패키지 (context, splitter, judge, pipeline, app, config)
serve.py              게이트웨이 실행
mock_upstream.py      API 키 없이 시험하는 가짜 LLM
scripts/              재생 데이터 생성, 종단 간 측정
tests/                pytest (기본: 경량, -m model: torch 필요)
pyproject.toml        패키지·의존성 정본 (기본 / [judge] / [dev])
.github/workflows/    CI (경량 테스트)
experiments/
  notebooks/          Colab 실험 노트북 (task1*, s1~s3, e1·e2·e4lite·e5lite, l1, r0~r2, r1·r1b·r1c, r4·r4b·r4c)
  builders/           노트북 생성 스크립트 — 노트북은 여기서 만든다
docs/
  reports/            주간 진행 보고서
  plan/               연구계획서
SSOT.md               확정된 사실·수치·결정 (단일 진실 공급원)
SPEC.md               게이트웨이 동작 명세
AGENTS.md / CLAUDE.md AI 코딩 에이전트 작업 규칙
```

## 문서

| 문서 | 내용 | 갱신 시점 |
|---|---|---|
| [docs/PROGRESS.md](docs/PROGRESS.md) | 시간 순 진행 기록과 **현재 위치·다음 할 일** | 작업 단위가 끝날 때 |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | 스트리밍 조기 탐지 연구 설계 (RQ, 선행 연구, 실험 E1~E6) | 설계가 바뀔 때 |
| [SSOT.md](SSOT.md) | 데이터 분할, 판정기 사양, 모든 실측 수치, 사전 기준, 결정 기록, 일정 | 실험 결과가 나오거나 결정이 바뀔 때 |
| [SPEC.md](SPEC.md) | API, 정책, 불변식, 측정 기록 형식, 수용 기준 | 게이트웨이 동작이 바뀔 때 |
| [AGENTS.md](AGENTS.md) | 에이전트 공통 작업 규칙 | 규칙이 바뀔 때 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 전용 추가 규칙 (AGENTS.md를 불러옴) | |

## 데이터·모델

- [RAGTruth](https://github.com/ParticleMedia/RAGTruth), [HaluEval](https://github.com/RUCAIBox/HaluEval), 기준 판정기 [MiniCheck](https://github.com/Liyan06/MiniCheck)
- 학습된 판정기 가중치(`judge_base/` 등)는 저장소에 포함하지 않습니다. 판정기 폴더의 `judge_config.json`이 입력 방식·임계값·재판정 간격을 정합니다(SPEC §4).
