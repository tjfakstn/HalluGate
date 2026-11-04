import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = ROOT
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# S3 — 검증 게이트웨이 골격 종단 간 측정

## 왜 하나

S2-b까지 판정기(student_vark, k=8)는 **오프라인**으로만 쟀다. 즉 완성된 응답을 한꺼번에 문장으로 나눠 채점했다.
미들웨어로 쓰려면 실제 흐름에서 같은 성능이 나오는지 확인해야 한다. 실제 흐름은 LLM이 토큰을 조금씩 흘려보내고,
게이트웨이가 그걸 받아 문장을 끊고, 판정하고, 사용자에게 내보내는 과정이다.
또 사용자가 느끼는 지연은 판정 시간만으로 정해지지 않는다. **언제 내보내느냐(정책)**가 함께 결정한다.

## 무엇을 하나

OpenAI 호환 프록시(hallugate)를 띄우고, 가짜 LLM이 RAGTruth dev 응답 60개를 실제 API와 비슷한 속도로 재생한다
(첫 토큰 300ms, 토큰당 25ms). 각 응답을 세 모드로 게이트웨이에 통과시킨다.

| 모드 | 동작 |
|---|---|
| off | 검증 없이 전달 (기준선) |
| pass | 토큰 즉시 전달, 문장 완성 시 백그라운드 판정 → 결과를 뒤따라 전달 |
| hold | 문장 완성까지 보류 → 판정 → 전달 |

## 사전 정의 기준

- **본문 보존**: 세 모드 모두 받은 본문이 원래 응답과 글자 단위로 100% 같을 것
- **판별력 유지**: 게이트웨이 안에서 나온 문장 AUROC ≥ 0.80. 오프라인 채택 기준은 0.78이었고, 표본이 60개 응답이라 여유를 둔다.
- **판정 속도(GPU)**: 문장당 판정 p95 ≤ 30ms, pass 모드에서 문장 완성부터 판정 전달까지 p95 ≤ 50ms
- **정책 트레이드오프 기준선 (합격/불합격 없음, S4의 출발점)**:
  hold의 추가 지연(첫 글자, 글자당 평균)과 pass의 누출량(환각 문장을 잡기 전까지 이미 나간 글자 수)

> 런타임 **GPU(T4)**. S2-b에서 저장한 `student_vark/`가 필요하다(드라이브 또는 업로드). 전체 약 20~25분.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install pysbd pytest-asyncio openai fastapi uvicorn httpx sentence-transformers
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'
!mkdir -p gateway/hallugate gateway/scripts gateway/tests
""")

code(r"""
# student_vark 위치 — 드라이브에 저장했다면 경로를 맞추고, 아니면 zip을 업로드해 푼 경로
import os, shutil
USE_DRIVE = True
SRC = '/content/drive/MyDrive/selfproject/student_vark'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
if not os.path.isdir('student_vark'):
    shutil.copytree(SRC, 'student_vark')
print('판정기:', os.listdir('student_vark'))
""")

md("## 2. 게이트웨이 코드 (hallugate)\n\n아래 셀들은 파일을 만들기만 한다. 로컬에서 16개 테스트를 통과한 코드와 같다.")
files = ['hallugate/__init__.py', 'hallugate/config.py', 'hallugate/context.py', 'hallugate/splitter.py',
         'hallugate/judge.py', 'hallugate/pipeline.py', 'hallugate/app.py', 'serve.py', 'mock_upstream.py',
         'scripts/make_replay_set.py', 'scripts/bench.py', 'tests/test_gateway.py', 'pytest.ini']
for f in files:
    code(f'%%writefile gateway/{f}\n' + open(os.path.join(G, f)).read())

code(r"""
# 코드 점검 — 모델 없이 도는 테스트 16개
!cd gateway && python -m pytest -q 2>&1 | tail -3
""")

md("## 3. 재생 데이터 — RAGTruth dev에서 과제별 20개 (S1~S2-b와 같은 dev 표본)")
code(r"""
!cd gateway && python scripts/make_replay_set.py --ragtruth ../RAGTruth/dataset --per-task 20 --out replay_dev.jsonl
""")

md("## 4. 가짜 LLM과 게이트웨이 실행 (GPU 판정기)")
code(r"""
import subprocess, time, httpx, os

def start(cmd, env_extra, log):
    env = {**os.environ, **env_extra}
    return subprocess.Popen(cmd, cwd='gateway', env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT)

def wait(url, proc, log, sec=180):
    for _ in range(sec):
        try:
            return httpx.get(url, timeout=2).json()
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(1)
    print(open(log).read()[-3000:]); raise RuntimeError(f'{url} 시작 실패')

mock = start(['python', 'mock_upstream.py', '--port', '9000'], {}, 'mock.log')
gw = start(['python', 'serve.py', '--port', '8000'],
           {'JUDGE_MODEL_DIR': '../student_vark', 'UPSTREAM_BASE_URL': 'http://127.0.0.1:9000/v1',
            'METRICS_PATH': 'metrics_gpu.jsonl', 'JUDGE_K': '8', 'JUDGE_MAX_LEN': '512'}, 'gw.log')
time.sleep(2)
print(wait('http://127.0.0.1:8000/health', gw, 'gw.log'))
""")

md(r"""## 5. 종단 간 측정 (60개 응답 × 3모드, 약 15분)""")
code(r"""
!cd gateway && python scripts/bench.py --replay replay_dev.jsonl --gateway http://127.0.0.1:8000 \
    --metrics metrics_gpu.jsonl --modes off pass hold --out bench_gpu
""")

md(r"""## 6. CPU 판정기 속도 (GPU가 없는 서버에 배포할 경우)

GPU에서는 k와 관계없이 약 15ms였다. CPU에서도 그런지, 그리고 k가 다시 지연을 좌우하는지 확인한다.""")
code(r"""
cpu_runs = {}
for k in [2, 8]:
    port = 8100 + k
    p = start(['python', 'serve.py', '--port', str(port)],
              {'JUDGE_MODEL_DIR': '../student_vark', 'UPSTREAM_BASE_URL': 'http://127.0.0.1:9000/v1',
               'METRICS_PATH': f'metrics_cpu_k{k}.jsonl', 'JUDGE_DEVICE': 'cpu', 'JUDGE_K': str(k),
               'CUDA_VISIBLE_DEVICES': ''}, f'gw_cpu{k}.log')
    wait(f'http://127.0.0.1:{port}/health', p, f'gw_cpu{k}.log')
    !cd gateway && python scripts/bench.py --replay replay_dev.jsonl --gateway http://127.0.0.1:{port} \
        --metrics metrics_cpu_k{k}.jsonl --modes pass --n 15 --out bench_cpu_k{k} | grep -E "judge_ms|AUROC|verdict_lag"
    p.terminate()
""")

md("## 7. 사전 기준 판정")
code(r"""
import json, pandas as pd
S = json.load(open('gateway/bench_gpu/bench_summary.json'))
print(pd.DataFrame(S).round(3).to_string())

ok_text = all(S[m]['text_ok_rate'] == 1.0 for m in S)
auc = S['pass']['AUROC']
j95 = max(S['pass']['judge_ms_p95'], S['hold']['judge_ms_p95'])
lag95 = S['pass']['verdict_lag_ms_p95']
print(f"\n본문 보존 100%          : {'충족' if ok_text else '미달'}")
print(f"게이트웨이 내 AUROC     : {auc:.3f} → {'충족' if auc >= 0.80 else '미달'} (기준 0.80)")
print(f"판정 p95                : {j95:.1f} ms → {'충족' if j95 <= 30 else '미달'} (기준 30ms)")
print(f"pass 판정 전달 p95      : {lag95:.1f} ms → {'충족' if lag95 <= 50 else '미달'} (기준 50ms)")
print('\n=== 정책 트레이드오프 기준선 (S4 출발점) ===')
print(f"hold: 첫 글자 +{S['hold']['ttft_added_ms_p50']:.0f} ms, 글자당 평균 +{S['hold']['char_delay_ms_mean']:.0f} ms "
      f"(p95 +{S['hold']['char_delay_ms_p95']:.0f} ms), 누출 0")
print(f"pass: 첫 글자 +{S['pass']['ttft_added_ms_p50']:.0f} ms, 글자당 평균 +{S['pass']['char_delay_ms_mean']:.0f} ms, "
      f"잡아낸 환각 문장의 누출 p50 {S['pass']['leak_chars_p50_on_caught']} 글자")
print(f"임계값 0.0에서 재현율 {S['pass']['recall_at_threshold']:.2f}, 정밀도 {S['pass']['precision_at_threshold']:.2f} (S4에서 보정)")
""")

code(r"""
# 결과 파일 모으기
!cd gateway && zip -qr ../s3_results.zip bench_gpu bench_cpu_k2 bench_cpu_k8 metrics_gpu.jsonl
from google.colab import files
files.download('s3_results.zip')
""")

md(r"""## (선택) 실제 API로 데모

API 키가 있으면 가짜 LLM 대신 실제 모델 응답을 검증해 볼 수 있다.""")
code(r"""
# gw.terminate()
# real = start(['python', 'serve.py', '--port', '8001'],
#              {'JUDGE_MODEL_DIR': '../student_vark', 'UPSTREAM_BASE_URL': 'https://api.openai.com/v1',
#               'UPSTREAM_API_KEY': 'sk-...', 'METRICS_PATH': 'metrics_real.jsonl'}, 'gw_real.log')
# wait('http://127.0.0.1:8001/health', real, 'gw_real.log')
# from openai import OpenAI
# ctx = json.loads(open('gateway/replay_dev.jsonl').readline())['reference']
# client = OpenAI(base_url='http://127.0.0.1:8001/v1', api_key='unused')
# stream = client.chat.completions.create(model='gpt-4o-mini', stream=True,
#     messages=[{'role': 'user', 'content': 'Summarize the context in 3 sentences.'}],
#     extra_body={'verification': {'context': ctx, 'mode': 'hold'}})
# for ch in stream:
#     if ch.choices and ch.choices[0].delta.content: print(ch.choices[0].delta.content, end='')
#     if getattr(ch, 'verification', None): print('\n  [판정]', ch.verification)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 's3_gateway_bench.ipynb'), 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
