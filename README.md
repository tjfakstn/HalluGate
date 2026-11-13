# hallugate

**RAG 응답을 스트리밍 중에 문장 단위로 검증하는 LLM 환각 검증 게이트웨이**
2026-2 아주대학교 소프트웨어학과 자기주도프로젝트 · 설만수 · 지도 조현석 교수

서비스와 LLM API 사이에 끼우는 OpenAI 호환 프록시입니다. 서비스는 `base_url`만 바꾸면 되고,
게이트웨이는 요청에 담긴 근거 문서와 LLM이 흘려보내는 답변을 문장마다 대조해 환각을 표시하거나 차단합니다.

```
서비스 ──(요청 + 근거 문서)──▶ hallugate ──▶ LLM API (OpenAI 등)
       ◀──(본문 + 문장별 판정)──   │   ◀── 토큰 스트림
                                   └─ 문장 분할 → 소형 판정기(문장당 약 15ms, GPU)
```

## 왜 만드나

기존 스트리밍 검증 도구는 정확도와 지연의 균형이 **고정**되어 있습니다.
정확한 전용 판정기(MiniCheck)는 문장당 약 400ms라 실시간에 못 쓰고, 빠른 방식은 부정확합니다.
이 프로젝트는 두 가지를 만듭니다.
1. **빠르고 정확한 소형 판정기**: 기존 판정기를 그대로 얹지 않고 공개 데이터로 직접 학습
2. **검증 단위와 전달 정책을 조절하는 게이트웨이**: 그 트레이드오프를 요청·문장마다 바꿀 수 있게 함

## 현재 상태 (2026-11-13)

| 항목 | 결과 |
|---|---|
| 배포 판정기 (I3, 11/11 채택) | **F1 base(DeBERTa-v3-base) + I2(ModernBERT-large, 근거 전체 2,048토큰) 점수 표준화 평균**, 8토큰 재판정, 임계값 2.992(cal-L 서비스 10%). dev-L 탐지율 **0.534 / 지연 9토큰**(F1 base 단독 0.495 / 14), Summary 0.156 → 0.266, 멀쩡한 응답 표시율 7.3%. 두 판정기 동시 실행(CUDA 스트림 2개) A100 판정 1회 p95 **61.3ms**(L3). 게이트웨이 연결 전 |
| 최신 LLM 점검 (M1, 11/11) | Claude Haiku·Sonnet 5.5 응답 536개(QA는 컬렉션 검색)에서 멀쩡한 응답 표시율 **4.5%**(기준 ≤ 15% 충족) → 임계값 유지. 단 최신 LLM 환각 탐지율은 **14%**(91개 중 13개, Opus 정답 기준)로 낮음. 원인은 근거 단어를 그대로 쓴 직접 모순(M1a, 놓친 것의 68%) |
| 직접 모순 재학습 (K1, 11/12) | Sonnet으로 합성한 모순 1,500개를 더해 F1 base 재학습 → 새로 생성한 최신 LLM 응답 탐지율 9.5% → 11.6%(기준 +10%p 미달) → **미채택**, 배포 판정기 그대로 |
| 최신 LLM 실제 응답 재학습 (K2, 11/13) | Claude Haiku·Sonnet 응답 5,855개(환각 문장 806개)를 더해 F1 base 재학습 → devM 탐지율 18.8% → 20.9%(같은 표시율 기준, 기준 +10%p 미달) → **미채택**. 사전 규칙에 따라 서비스 임계값은 최신 LLM 기준 B(2.7385)로 바꿔, 최신 LLM 탐지율 14.1% → 18.8%, 멀쩡한 응답 표시율 6.6% |
| 외부 비교 (X1, 보고용) | LettuceDetect-large가 dev-L에서 앞서지만(응답 AUROC 0.984 vs 0.860) 학습 응답 암기 정황이 강해 우열은 test에서. HHEM-2.1-open보다는 앞섬 |
| 이전 판정기 | `student_vark`(문장 끝, RAGTruth dev AUROC 0.859, 약 15ms/문장) → F1 base 단독(8토큰, 임계값 3.969, dev-L 0.495). 게이트웨이는 `JUDGE_MODEL_DIR` 폴더의 `judge_config.json`으로 단일 판정기를 불러옴 |
| 일반화 (HaluEval, 길이 통제) | 대화 우세, QA 비슷, **요약 약함** |
| 게이트웨이 | 종단 간 측정 기준 충족: 본문 보존 100%, AUROC 0.853, 판정 p95 20.6ms. 토큰 단위 재판정·판정기 설정 파일·fail-open·컬렉션 RAG(문서 등록·검색·같은 근거로 판정) 구현(S4, 테스트 39개) |
| 평가 세트 | 근거 문서가 학습과 겹치지 않는 cal-L(보정)·dev-L(주 평가, 환각 759개). 기존 dev는 참고 |
| 엄격 재현율 상한(약 0.50)의 원인 | 경보 규칙·학습 목표·판정기 크기 모두 같은 상한 → **RAGTruth 정답 누락**이 원인(R4 Claude Opus 재판정). 학습 정답을 고치면 앞부분 판정이 문장 끝 판정을 따라잡음 |
| 남은 문제 | 게이트웨이에 조합 판정기 연결(SPEC → 테스트 → 코드), `judge_config` v1.0 동결·모델 카드, **사람 점검**(Opus 재판정 정밀도, 30개), 최신 LLM 직접 모순 미탐(알려진 한계), Claude 업스트림 변환·재생성, 최종 RAGTruth test 1회 평가, 요약 과제 약점 |

전체 수치와 결정 이력은 **[SSOT.md](SSOT.md)**, 동작 명세는 **[SPEC.md](SPEC.md)** 에 있습니다.

## 빠른 시작

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 모델 없이 도는 테스트 23개

# 가짜 LLM + 게이트웨이 (판정기 없이 파이프라인 확인: --lexical)
python mock_upstream.py --port 9000 &
UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 python serve.py --lexical --port 8000 &

# 학습 판정기로 실행
JUDGE_MODEL_DIR=path/to/judge_base UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 python serve.py --port 8000 &

# 종단 간 측정 (RAGTruth 필요)
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git
python scripts/make_replay_set.py --ragtruth RAGTruth/dataset --per-task 20
python scripts/bench.py --metrics gateway_metrics.jsonl
```

Colab(T4)에서는 `experiments/notebooks/s3_gateway_bench.ipynb`를 그대로 실행하면 됩니다.

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

## 저장소 구조

```
hallugate/            게이트웨이 패키지 (context, splitter, judge, pipeline, app, config)
serve.py              게이트웨이 실행
mock_upstream.py      API 키 없이 시험하는 가짜 LLM
scripts/              재생 데이터 생성, 종단 간 측정
tests/                pytest
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
