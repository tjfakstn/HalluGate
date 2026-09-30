# AGENTS.md — AI 코딩 에이전트 공통 작업 규칙

이 저장소에서 작업하는 모든 에이전트(Claude Code, Codex, Cursor 등)가 따르는 규칙이다.
사람 기여자에게도 같은 규칙이 적용된다.

## 먼저 읽을 것

0. [docs/PROGRESS.md](docs/PROGRESS.md)의 "현재 위치": 지금 어디까지 왔고 다음에 무엇을 할지
1. [SSOT.md](SSOT.md): 확정된 사실·수치·결정. **수치를 말하거나 쓰기 전에 여기서 확인한다.**
2. [SPEC.md](SPEC.md): 게이트웨이가 지켜야 할 동작과 불변식
3. 작업 대상 코드와 그 테스트

## 프로젝트 한 줄 요약

RAG 응답을 스트리밍 중에 문장 단위로 근거 문서와 대조하는 OpenAI 호환 게이트웨이(`hallugate/`)와,
그 판정기를 만들고 검증하는 실험(`experiments/`). 연구 과제이므로 **실험의 공정성이 코드 품질만큼 중요하다.**

## 명령어

```bash
pip install -r requirements.txt
python -m pytest -q                                        # 전체 테스트 (네트워크·모델 다운로드 없음, 약 10초)
python mock_upstream.py --port 9000 &                      # 가짜 LLM
UPSTREAM_BASE_URL=http://127.0.0.1:9000/v1 python serve.py --lexical --port 8000   # 게이트웨이 (대체 판정기)
python scripts/make_replay_set.py --ragtruth RAGTruth/dataset --per-task 20
python scripts/bench.py --metrics gateway_metrics.jsonl    # 종단 간 측정
python experiments/builders/build_s3.py                    # 노트북 재생성 (다른 build_*.py도 동일)
```

## 코드 지도

| 경로 | 책임 | 관련 명세 |
|---|---|---|
| `hallugate/context.py` | 요청에서 근거 문서·검증 옵션 추출, `verification` 필드 제거 | SPEC §2.1 |
| `hallugate/splitter.py` | 스트림 → 문장. 불변식 I1~I3 | SPEC §3 |
| `hallugate/judge.py` | `StudentJudge`(운영), `LexicalJudge`(테스트 전용) | SPEC §4 |
| `hallugate/pipeline.py` | pass/hold, flag/cut 정책과 시간 기록 | SPEC §5, §6 |
| `hallugate/app.py` | FastAPI 프록시, SSE 청크 구성 | SPEC §2 |
| `scripts/bench.py` | 종단 간 지표 계산 | SPEC §6 |
| `experiments/builders/*.py` | 노트북 생성 스크립트 | — |

## 반드시 지킬 규칙

### A. 연구 공정성 (위반 시 결과 전체가 무효가 됨)

1. **RAGTruth test 분할(`split == 'test'`)을 열지 않는다.** S5 최종 평가 노트북에서 단 1회만 연다.
   데이터 로딩 코드는 `split == 'train'`으로 거른 풀에서만 표본을 뽑는다.
2. **평가 기준은 결과를 보기 전에 정한다.** 노트북 첫 마크다운 셀에 "사전 정의 기준"을 적고 나서 실행한다.
   결과를 본 뒤 기준을 바꿔야 하면, 바꾼 사실과 이유를 SSOT 결정 기록에 남긴다. 조용히 바꾸지 않는다.
3. **분할과 시드를 바꾸지 않는다.** dev(`random.seed(42)`, 과제별 120), teacher(`random.Random(11)`),
   gold-large 등 SSOT §2의 표를 그대로 재현한다. 새 분할이 필요하면 새 시드를 쓰고 SSOT에 추가한다.
   분할 간 응답이 겹치지 않음을 `assert`로 확인한다.
4. **dev로 모델을 고르지 않는다.** 학습 중 최적 시점 선택은 검증 분할로만 하고, dev는 최종 측정에만 쓴다.
5. **HaluEval은 더 이상 독립 검증 세트가 아니다** (S2-b 이후). 이것으로 개선하면 일반화 근거로 쓰지 않는다.
6. **`LexicalJudge`로 얻은 판별력 수치는 어디에도 보고하지 않는다.** 파이프라인 동작 확인용이다.
7. **실제로 실행해서 얻은 수치만** SSOT·보고서에 적는다. 추정은 "추정"이라고 적는다.

### B. 게이트웨이 코드

1. 본문 보존(SPEC G2, I1)은 절대 깨지면 안 된다. 관련 코드를 바꾸면 `test_splitter_streaming_matches_whole`과
   `test_stream_text_intact_and_verdicts`가 통과하는지 반드시 확인한다.
2. OpenAI 호환성을 유지한다. 표준 청크 형식을 바꾸지 말고, 추가 정보는 `choices: []` 판정 청크와 `verification` 필드로만 보낸다.
3. 동작이 바뀌면 **SPEC.md → 테스트 → 코드** 순서로 고친다. SPEC에 없는 동작을 추가하지 않는다.
4. 판정기 호출은 이벤트 루프를 막지 않게 `asyncio.to_thread`로 실행한다.
5. 테스트는 네트워크와 모델 다운로드 없이 돌아야 한다. 가짜 LLM(`mock_upstream`), `LexicalJudge`, 작은 임의 모델을 쓴다.
   시간 간격이 중요한 테스트는 `httpx.ASGITransport`가 아니라 실제 uvicorn 서버(`live` 픽스처)로 한다.
   ASGITransport는 응답을 한 번에 모아서 전달하기 때문이다.
6. 의존성을 추가하면 `requirements.txt`를 갱신한다.

### C. 실험 노트북

1. `.ipynb`를 직접 고치지 않는다. `experiments/builders/build_*.py`를 고치고 다시 생성한다.
2. `s3_gateway_bench.ipynb`는 `hallugate/` 코드를 `%%writefile`로 품고 있다.
   **패키지 코드를 바꾸면 `build_s3.py`로 노트북을 다시 만든다.**
3. 실행 환경은 Colab T4 GPU다. 오래 걸리는 단계(MiniCheck 레이블링 등)는 조각마다 저장해 이어서 실행할 수 있게 만든다.
4. 노트북은 "왜 하나 → 무엇을 하나 → 사전 정의 기준 → 실행 → 기준 판정 → 저장" 구조를 따른다.
5. 새 노트북을 커밋하기 전에 가짜 데이터·작은 모델로 끝까지 도는지 확인한다(smoke test).

### D. 문서

1. 수치가 바뀌면 SSOT 해당 표와 결정 기록을 함께 고친다. README의 "현재 상태"는 SSOT와 일치해야 한다.
   작업 단위(실험 하나, 기능 하나)가 끝나면 `docs/PROGRESS.md`에 한 절을 추가하고 "현재 위치"를 갱신한다.
2. 문서는 한국어로 쓴다. 코드 식별자, 지표 이름(AUROC 등), 모델 이름은 원문 그대로 쓴다.
3. 주간 보고서(`docs/reports/`) 작성 규칙:
   - 줄글만 쓴다 (목록·표 없음).
   - 한 문단은 5~7문장이다.
   - 문단 첫 문장(주제문)에 결론이나 선택 이유를 쓴다. 결론을 문단 끝으로 미루지 않는다.
   - 분량은 1.5~2쪽이다.
   - 실제로 수행한 작업과 실측 수치만 넣는다.

### E. 커밋

- 접두어: `feat:`(기능) `fix:`(버그) `test:` `exp:`(실험·노트북) `docs:` `refactor:` `chore:`
- 실험 하나 = 커밋 하나. 실험 결과를 반영할 때는 노트북, SSOT, 필요하면 보고서를 같은 커밋에 묶는다.
- 커밋하지 않는 것: 데이터셋(`RAGTruth/`, `HaluEval/`), 모델 가중치(`student_*/`), 측정 산출물(`*.jsonl`, `bench_*/`), 비밀 키. `.gitignore` 참고.
- 커밋 전에 `python -m pytest -q`가 통과해야 한다.

## 완료 조건 (Definition of Done)

- [ ] 테스트 통과, 새 동작에는 새 테스트가 있다
- [ ] SPEC과 코드가 일치한다 (상태 표기 ✅/🚧/📋 갱신)
- [ ] 새 수치·결정이 SSOT에 반영됐다
- [ ] 노트북이 바뀌었다면 builder에서 재생성했다
- [ ] 연구 공정성 규칙(A)을 어기지 않았다
