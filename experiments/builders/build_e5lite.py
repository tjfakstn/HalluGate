"""E5-lite 노트북 생성: 조기 탐지 판정기(student_prefix)를 게이트웨이에 올려 종단 간으로 측정한다.
게이트웨이 코드는 저장소의 hallugate/를 %%writefile로 품는다 (패키지를 바꾸면 이 스크립트로 다시 만들 것)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# E5-lite — 조기 탐지 판정기를 게이트웨이에 올리면 누출과 대기가 줄어드는가

## 왜 하나

E2에서 문장 앞부분 학습 판정기(`student_prefix`)는 오프라인 평가에서 문장 끝 판정과 같은 재현율로 탐지를 17토큰에서 6.5토큰으로 앞당겼다.
이것이 **실제 서비스 흐름**에서도 이득이 되는지 확인한다. S3에서 문장 끝 판정 게이트웨이는 두 가지 비용이 있었다.
hold는 첫 글자가 447ms 늦었고, pass는 환각 문장이 판정 전에 약 126자 나갔다.

## 무엇을 하나

게이트웨이에 **앞부분 재판정(stride)** 기능을 추가했다(SPEC §5.1).
- pass: 문장 도중에도 s토큰마다 앞부분을 판정해, 경보를 더 일찍 보낸다.
- hold: 앞부분이 안전 판정될 때마다 그때까지의 본문을 먼저 내보낸다.

세 가지 판정기 구성을 같은 조건에서 비교한다. 가짜 LLM은 **GPT 토큰 단위**로, 첫 토큰 300ms, 토큰당 25ms로 RAGTruth dev 60개 응답을 재생한다.

| 구성 | 판정기 | 판정 시점 | 임계값 |
|---|---|---|---|
| **B** (기준선, S3와 같음) | student_vark | 문장 끝 | E2 서비스 조건 보정값 (멀쩡한 응답 표시 10%) |
| **E-end** | student_prefix | 문장 끝 | 〃 |
| **E-s4** | student_prefix | 4토큰마다 앞부분 + 문장 끝 | 〃 |

임계값은 E2 노트북이 **검증 분할**로 보정해 저장한 값(`e2/e2_summary.csv`)을 그대로 쓴다. 이 노트북에서 새로 고르지 않는다.

## 지표 (환각 발생 시점 기준, SSOT 결정)

- 발생 위치 θ = 응답의 첫 환각 구간 시작 글자
- **발생 후 누출**: 첫 경보가 나온 순간까지 사용자에게 나간 글자 중 θ 이후 글자 수 (탐지된 응답 기준 중앙값)
- **엄격 재현율**: 첫 경보가 θ를 포함한 텍스트를 판정한 것일 때만 탐지
- **대기**: hold의 첫 글자 추가 지연, 글자당 평균 지연 (off 대비)
- 멀쩡한 응답 표시율, 본문 보존율, 판정 시간

## 사전 정의 기준 (SSOT §6에 고정, 결과 보기 전)

- **C1 대기**: E-s4 hold의 첫 글자 추가 지연 ≤ B hold의 50%
- **C2 누출**: E-s4 pass의 발생 후 누출 중앙값(탐지된 응답) ≤ B pass의 50%
- **C3 탐지 유지**: E-s4 pass의 엄격 재현율 ≥ B pass − 0.05
- **C4 기본**: 모든 구성에서 본문 보존 100%, 판정 p95 ≤ 30ms
- 60개 응답 중 환각 응답은 약 25개라 재현율은 변동이 크다. C3은 큰 퇴보가 없는지만 확인한다.

> 런타임 **GPU(T4)**. 드라이브에 `selfproject/student_vark`, `student_prefix`, `e2/e2_summary.csv`가 필요하다. 약 35~40분.
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install pysbd pytest-asyncio openai fastapi uvicorn httpx sentence-transformers tiktoken
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'
!mkdir -p gateway/hallugate gateway/scripts gateway/tests
""")
code(r"""
import os, shutil, json, pandas as pd
from google.colab import drive
drive.mount('/content/drive')
OUT_DIR = '/content/drive/MyDrive/selfproject'
for m in ['student_vark', 'student_prefix']:
    if not os.path.isdir(m):
        shutil.copytree(os.path.join(OUT_DIR, m), m)
e2s = pd.read_csv(os.path.join(OUT_DIR, 'e2', 'e2_summary.csv'))
svc = e2s[e2s.target_clean_flag == 0.10].set_index('scorer').thr
THR = {'B': float(svc['B1 문장 끝']), 'E-end': float(svc['E2 문장 끝']), 'E-s4': float(svc['E2 앞부분 (s=4)'])}
print('임계값 (E2 검증 보정, 멀쩡한 응답 표시 10%):', {k: round(v, 3) for k, v in THR.items()})
""")

md("## 2. 게이트웨이 코드 (hallugate)\n\n파일만 만든다. 저장소의 코드와 같다.")
files = ['hallugate/__init__.py', 'hallugate/config.py', 'hallugate/context.py', 'hallugate/splitter.py',
         'hallugate/judge.py', 'hallugate/pipeline.py', 'hallugate/app.py', 'serve.py', 'mock_upstream.py',
         'scripts/make_replay_set.py', 'scripts/bench.py', 'tests/test_gateway.py', 'pytest.ini']
for f in files:
    code(f'%%writefile gateway/{f}\n' + open(os.path.join(ROOT, f)).read())
code(r"""
!cd gateway && python -m pytest -q 2>&1 | tail -3
""")

md("## 3. 재생 데이터 — RAGTruth dev 과제별 20개 (S3와 같음)")
code(r"""
!cd gateway && python scripts/make_replay_set.py --ragtruth ../RAGTruth/dataset --per-task 20 --out replay_dev.jsonl
""")

md("## 4. 가짜 LLM(GPT 토큰 단위)과 게이트웨이 3개 실행")
code(r"""
import subprocess, time, httpx

def start(cmd, env_extra, log):
    return subprocess.Popen(cmd, cwd='gateway', env={**os.environ, **env_extra},
                            stdout=open(log, 'w'), stderr=subprocess.STDOUT)

def wait(url, proc, log, sec=180):
    for _ in range(sec):
        try:
            return httpx.get(url, timeout=2).json()
        except Exception:
            if proc.poll() is not None: break
            time.sleep(1)
    print(open(log).read()[-3000:]); raise RuntimeError(f'{url} 시작 실패')

mock = start(['python', 'mock_upstream.py', '--port', '9000'], {'MOCK_TOKENIZER': 'tiktoken'}, 'mock.log')
GW = {'B': (8001, '../student_vark'), 'E-end': (8002, '../student_prefix'), 'E-s4': (8003, '../student_prefix')}
procs = {}
for name, (port, model) in GW.items():
    procs[name] = start(['python', 'serve.py', '--port', str(port)],
                        {'JUDGE_MODEL_DIR': model, 'UPSTREAM_BASE_URL': 'http://127.0.0.1:9000/v1',
                         'METRICS_PATH': f'metrics_{name}.jsonl', 'JUDGE_THRESHOLD': str(THR[name]),
                         'JUDGE_K': '8', 'JUDGE_MAX_LEN': '512'}, f'gw_{name}.log')
    time.sleep(2)
for name, (port, _) in GW.items():
    print(name, wait(f'http://127.0.0.1:{port}/health', procs[name], f'gw_{name}.log'))
""")

md("## 5. 종단 간 측정 (구성마다 off·pass·hold, 약 35분)")
code(r"""
MODES = {'B': 'off pass hold', 'E-end': 'off pass hold', 'E-s4': 'off pass:4 hold:4'}
for name, (port, _) in GW.items():
    print(f'=== {name} ===')
    !cd gateway && python scripts/bench.py --replay replay_dev.jsonl --gateway http://127.0.0.1:{port} \
        --metrics metrics_{name}.jsonl --modes {MODES[name]} --out bench_{name} | tail -2
""")

md("## 6. 결과")
code(r"""
rows = []
for name in GW:
    S = json.load(open(f'gateway/bench_{name}/bench_summary.json'))
    for mode, s in S.items():
        if mode == 'off': continue
        rows.append({'config': name, 'mode': mode.split(':')[0], **s})
R = pd.DataFrame(rows)
cols = ['config', 'mode', 'text_ok_rate', 'ttft_added_ms_p50', 'char_delay_ms_mean', 'char_delay_ms_p95',
        'judge_ms_p95', 'onset_n_hallucinated', 'onset_recall_strict', 'onset_detect_delay_chars_p50',
        'onset_leak_chars_p50_detected', 'onset_expected_leak_chars_mean', 'clean_resp_flagged', 'AUROC']
pd.set_option('display.width', 250)
print(R[cols].round(3).to_string(index=False))
""")

md("## 7. 사전 기준 판정")
code(r"""
g = R.set_index(['config', 'mode'])
b_hold, e_hold = g.loc[('B', 'hold')], g.loc[('E-s4', 'hold')]
b_pass, e_pass = g.loc[('B', 'pass')], g.loc[('E-s4', 'pass')]
c1 = e_hold.ttft_added_ms_p50 <= 0.5 * b_hold.ttft_added_ms_p50
c2 = e_pass.onset_leak_chars_p50_detected <= 0.5 * b_pass.onset_leak_chars_p50_detected
c3 = e_pass.onset_recall_strict >= b_pass.onset_recall_strict - 0.05
c4 = (R.text_ok_rate == 1.0).all() and (R.judge_ms_p95 <= 30).all()
print(f"C1 대기: hold 첫 글자 추가 지연 B {b_hold.ttft_added_ms_p50:.0f}ms → E-s4 {e_hold.ttft_added_ms_p50:.0f}ms "
      f"(글자당 평균 {b_hold.char_delay_ms_mean:.0f} → {e_hold.char_delay_ms_mean:.0f}ms) → {'충족' if c1 else '미달'}")
print(f"C2 누출: pass 발생 후 누출 중앙값 B {b_pass.onset_leak_chars_p50_detected:.0f}자 → E-s4 "
      f"{e_pass.onset_leak_chars_p50_detected:.0f}자 → {'충족' if c2 else '미달'}")
print(f"C3 탐지: 엄격 재현율 B {b_pass.onset_recall_strict:.3f} → E-s4 {e_pass.onset_recall_strict:.3f} "
      f"(환각 응답 {int(e_pass.onset_n_hallucinated)}개) → {'충족' if c3 else '미달'}")
print(f"C4 기본: 본문 보존 {R.text_ok_rate.min():.0%}, 판정 p95 최대 {R.judge_ms_p95.max():.1f}ms → {'충족' if c4 else '미달'}")
print(f"\n참고 — 멀쩡한 응답 표시율: " + ', '.join(f'{c}/{m} {v:.0%}' for (c, m), v in g.clean_resp_flagged.items()))
""")

code(r"""
!cd gateway && zip -qr ../e5lite_results.zip bench_B bench_E-end bench_E-s4 metrics_B.jsonl metrics_E-end.jsonl metrics_E-s4.jsonl
shutil.copy('e5lite_results.zip', os.path.join(OUT_DIR, 'e5lite_results.zip'))
from google.colab import files
files.download('e5lite_results.zip')
print('드라이브에도 저장:', os.path.join(OUT_DIR, 'e5lite_results.zip'))
""")

md(r"""## 8. (L1 이후) 실제 LLM 속도로 다시 측정

L1 노트북을 먼저 실행해 드라이브 `selfproject/l1/timing_<모델>.jsonl`이 있으면, 가짜 LLM이 RAGTruth 응답을
**그 모델의 실제 첫 토큰 시간과 생성 속도**로 흘려보내도록 해서 B와 E-s4를 다시 비교한다.
시간을 줄이기 위해 모델마다 30개 응답만 쓴다(모델당 약 15분). 측정할 모델은 `TIMING_MODELS`에서 고른다.""")
code(r"""
import glob
L1 = os.path.join(OUT_DIR, 'l1')
avail = sorted(os.path.basename(f)[7:-6] for f in glob.glob(os.path.join(L1, 'timing_*.jsonl')))
print('실측 속도 파일:', avail)
TIMING_MODELS = avail                      # 일부만 하려면 예: ['claude-haiku-4-5-20251001', 'gpt-5.4-mini']
N_REAL = 30
for tm in TIMING_MODELS:
    shutil.copy(os.path.join(L1, f'timing_{tm}.jsonl'), f'gateway/timing_{tm}.jsonl')
    for name, modes in [('B', 'off pass hold'), ('E-s4', 'off pass:4 hold:4')]:
        port = GW[name][0]
        print(f'=== {tm} | {name} ===')
        !cd gateway && python scripts/bench.py --replay replay_dev.jsonl --gateway http://127.0.0.1:{port} --n {N_REAL} \
            --metrics metrics_{name}.jsonl --modes {modes} --timing timing_{tm}.jsonl --out real_{tm}_{name} | tail -1
""")
code(r"""
rows = []
for tm in TIMING_MODELS:
    for name in ['B', 'E-s4']:
        S = json.load(open(f'gateway/real_{tm}_{name}/bench_summary.json'))
        for mode, s in S.items():
            rows.append({'llm': tm, 'config': name, 'mode': mode.split(':')[0], 'total_ms_p50': s['total_ms_p50'],
                         'ttft_ms_p50': s['ttft_ms_p50'], 'ttft_added_ms_p50': s.get('ttft_added_ms_p50'),
                         'char_delay_ms_mean': s.get('char_delay_ms_mean'),
                         'onset_leak_chars_p50_detected': s.get('onset_leak_chars_p50_detected'),
                         'onset_recall_strict': s.get('onset_recall_strict')})
RR = pd.DataFrame(rows)
RR['added_vs_total'] = RR.ttft_added_ms_p50 / RR.total_ms_p50
print(RR.round(3).to_string(index=False))
RR.to_csv(os.path.join(OUT_DIR, 'e5lite_real_timing.csv'), index=False)
!cd gateway && zip -qr ../e5lite_real_timing.zip real_*
shutil.copy('e5lite_real_timing.zip', os.path.join(OUT_DIR, 'e5lite_real_timing.zip'))
print('저장:', os.path.join(OUT_DIR, 'e5lite_real_timing.csv'))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(ROOT, 'experiments', 'notebooks', 'e5lite_gateway_early_detection.ipynb'), 'w'),
          ensure_ascii=False, indent=1)
print('cells', len(cells))
