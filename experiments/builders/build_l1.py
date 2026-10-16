"""L1 노트북 생성: 실제 상용 LLM(Claude, GPT)의 스트리밍 생성 시간 측정."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# L1 — 실제 상용 LLM의 생성 시간 측정 (Claude, GPT)

## 왜 하나 (5주차 보고서 피드백)

> "애초에 전체 문장의 생성 시간을 측정해야 되는 것이 아닌가?"

지금까지의 지연 수치(판정 15~20ms, hold 첫 글자 +447ms 등)는 **가짜 LLM의 고정 속도**(첫 토큰 300ms, 토큰당 25ms) 위에서 잰 것이다.
판정 시간과 보류 지연이 큰지 작은지는 **실제 LLM이 문장 하나, 응답 하나를 만드는 데 걸리는 시간**과 비교해야 판단할 수 있다.
실제 서비스에서 가장 많이 쓰는 Claude와 GPT의 스트리밍 생성 시간을 직접 잰다.

## 무엇을 하나

- **입력**: RAGTruth dev에서 과제별 20개, 총 60개 질문. RAGTruth가 응답을 만들 때 쓴 **원래 프롬프트**(근거 문서 포함)를 그대로 보낸다.
  게이트웨이 측정(S3, E5-lite)과 같은 60개다.
- **모델**: Claude 2종(Haiku 4.5, Sonnet 5.5), GPT 2종(빠른 소형 모델 1개, 주력 모델 1개). 모델은 셀 3에서 바꿀 수 있다.
- **호출 방식**: 각 회사의 **공식 SDK 스트리밍**. Claude는 Messages API, GPT는 Chat Completions API를 쓴다.
  호환 계층을 거치면 변환 지연이 섞일 수 있어서 쓰지 않는다.
- 조각(chunk)이 도착할 때마다 시각과 텍스트를 기록한다. 응답이 끝나면 문장으로 나눠 **문장마다 생성 시간**을 계산한다.
  문장 생성 시간은 문장의 첫 글자가 도착한 때부터 마지막 글자가 도착한 때까지다.

## 지표

| 지표 | 의미 |
|---|---|
| 첫 토큰 시간(TTFT) | 요청 → 첫 글자 도착 |
| 전체 응답 시간 | 요청 → 마지막 글자 도착 |
| 글자당 생성 시간 | (전체 − TTFT) ÷ 글자 수 |
| **문장 생성 시간** | 문장 첫 글자 → 마지막 글자 |
| **4토큰 생성 시간** | 앞부분 재판정 간격(s=4)만큼 생성되는 데 걸리는 시간 |

## 사전 정의 점검 (결과 보기 전)

판정기 시간은 S3·E5 실측값을 쓴다(GPU p50 17.6ms, p95 20.6ms / CPU k=8 p50 96ms, p95 152ms).
- **P1 문장 판정이 밀리지 않는가**: GPU 판정 p95 < 모든 모델의 문장 생성 시간 하위 5% 값
- **P2 앞부분 재판정이 밀리지 않는가**: GPU 판정 p95 < 모든 모델의 4토큰 생성 시간 중앙값
- **보고(판정 없음)**: 판정 시간이 문장 생성 시간·전체 응답 시간에서 차지하는 비율, hold 방식별 예상 추가 지연이 전체 응답 시간에서 차지하는 비율

> **API 키 필요**: 코랩 왼쪽 열쇠 아이콘(보안 비밀)에 `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`를 등록하거나 실행 중 입력한다.
> 비용은 모델 4개 × 60문항 기준으로 대략 1달러 미만이다(입력 약 600토큰, 출력 최대 512토큰). GPU 필요 없음.
>
> **비용 안전장치**: Claude(4-A)와 GPT(4-B)는 **셀이 따로**라서 한쪽이 실패해도 다른 쪽은 호출되지 않는다. 각 셀은 자기 키만 묻는다.
> 인증·모델명·결제 오류(400/401/402/403/404, 쿼터 소진)는 **재시도 없이 그 모델을 즉시 중단**하고, 일시 오류(429, 5xx, 연결)만 최대 3번 재시도한다.
> 빈 응답은 재시도하지 않고, 연속 3문항 실패하면 그 모델을 멈춘다. 결과는 문항마다 드라이브에 저장되므로 같은 셀을 다시 돌려도 **끝난 문항은 다시 호출하지 않는다.**
""")

md("## 1. 환경 준비")
code(r"""
!pip -q install anthropic openai pysbd tiktoken
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'

import json, os, random, time, statistics as st
import numpy as np, pandas as pd, pysbd, tiktoken
from google.colab import drive
drive.mount('/content/drive')
OUT_DIR = '/content/drive/MyDrive/selfproject/l1'
os.makedirs(OUT_DIR, exist_ok=True)

def get_key(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v
    except Exception:
        pass
    from getpass import getpass
    return getpass(f'{name}: ')

import anthropic, openai, glob
print('준비 완료 (API 클라이언트는 4-A, 4-B 셀에서 각각 만든다)')
""")

md("## 2. 질문 60개 — 게이트웨이 측정과 같은 dev 표본, RAGTruth 원래 프롬프트")
code(r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
for r in responses:
    r['task_type'] = sources[r['source_id']]['task_type']
TASKS = ['QA', 'Summary', 'Data2txt']
train_pool = [r for r in responses if r['split'] == 'train']
random.seed(42)
dev = []
for tt in TASKS:
    dev += random.sample([r for r in train_pool if r['task_type'] == tt], 120)
items = []
for tt in TASKS:                       # scripts/make_replay_set.py와 같은 선택 (과제별 dev 앞 20개)
    for r in [x for x in dev if x['task_type'] == tt][:20]:
        items.append({'id': str(r['id']), 'task_type': tt, 'prompt': sources[r['source_id']]['prompt']})
print(f'{len(items)}개 질문 | 프롬프트 글자 수 중앙값 {st.median(len(x["prompt"]) for x in items):.0f}')
""")

md(r"""## 3. 측정할 모델과 공통 설정

Claude 두 종류(Haiku 4.5, Sonnet 5.5)와 GPT 두 종류를 잰다. GPT 모델 이름은 계정마다 다르므로 4-B에서 목록을 보고 정한다.""")
code(r"""
CLAUDE_MODELS = ['claude-haiku-4-5-20251001', 'claude-sonnet-5-5']
MAX_TOKENS = 512
""")

md("## 4. 스트리밍 호출과 기록 (문항마다 드라이브에 저장, 이어서 실행 가능)")
code(r"""
def stream_anthropic(model, prompt):
    t0, chunks = time.perf_counter(), []
    with A.messages.stream(model=model, max_tokens=MAX_TOKENS, messages=[{'role': 'user', 'content': prompt}]) as s:
        for text in s.text_stream:
            if text:
                chunks.append(((time.perf_counter() - t0) * 1000, text))
        msg = s.get_final_message()
    return chunks, {'input_tokens': msg.usage.input_tokens, 'output_tokens': msg.usage.output_tokens,
                    'stop': msg.stop_reason}

def stream_openai(model, prompt):
    t0, chunks, usage = time.perf_counter(), [], None
    kw = dict(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True,
              stream_options={'include_usage': True}, **GPT_EXTRA)
    try:
        s = O.chat.completions.create(max_completion_tokens=MAX_TOKENS, **kw)
    except openai.BadRequestError:
        s = O.chat.completions.create(max_tokens=MAX_TOKENS, **kw)       # 구형 모델
    stop = None
    for ev in s:
        if ev.usage:
            usage = ev.usage
        for ch in ev.choices or []:
            if ch.delta and ch.delta.content:
                chunks.append(((time.perf_counter() - t0) * 1000, ch.delta.content))
            stop = ch.finish_reason or stop
    return chunks, {'input_tokens': usage.prompt_tokens if usage else None,
                    'output_tokens': usage.completion_tokens if usage else None, 'stop': stop}

def is_fatal(e):
    # 다시 불러도 소용없는 오류(인증·권한·모델명·요청 형식·결제)는 재시도하지 않고 멈춘다.
    sc, msg = getattr(e, 'status_code', None), str(e).lower()
    if sc in (400, 401, 402, 403, 404): return True
    return sc == 429 and any(k in msg for k in ('insufficient_quota', 'credit balance', 'billing'))

def run_model(provider, model, retries=3, max_consec_fail=3):
    fn = stream_anthropic if provider == 'anthropic' else stream_openai
    path = os.path.join(OUT_DIR, f'raw_{model}.jsonl')
    done = {json.loads(l)['id'] for l in open(path)} if os.path.exists(path) else set()
    todo = [x for x in items if x['id'] not in done]
    print(f'[{model}] 완료 {len(done)}, 남음 {len(todo)}')
    saved = fails = consec = 0
    for i, x in enumerate(todo):
        ok = False
        for k in range(retries):
            try:
                chunks, meta = fn(model, x['prompt'])
                if not chunks:       # 추론 모델이 출력 예산을 생각에 다 쓴 경우 등 — 다시 불러도 같은 결과
                    print(f'  {x["id"]}: 빈 응답 (stop={meta["stop"]}, 출력 토큰={meta["output_tokens"]}) — 재시도 안 함')
                    break
                ok = True
                break
            except Exception as e:
                if is_fatal(e):
                    print(f'  ✗ 치명적 오류 — {model} 중단 (재시도 안 함): {type(e).__name__}: {str(e)[:200]}')
                    print(f'  이 모델은 {saved}개 저장하고 멈춤. 다른 공급사 셀은 영향 없음.')
                    return
                print(f'  재시도 {k+1}/{retries}: {type(e).__name__}: {str(e)[:150]}'); time.sleep(5 * (k + 1))
        if ok:
            with open(path, 'a') as f:
                f.write(json.dumps({'id': x['id'], 'task_type': x['task_type'], 'provider': provider, 'model': model,
                                    'chunks': chunks, **meta}, ensure_ascii=False) + '\n')
            saved += 1; consec = 0
        else:
            fails += 1; consec += 1
            print(f'  {x["id"]} 실패 — 건너뜀 (연속 {consec}/{max_consec_fail})')
            if consec >= max_consec_fail:
                print(f'  ✗ 연속 {max_consec_fail}문항 실패 — {model} 중단'); return
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(todo)}')
    print(f'[{model}] 이번 실행: 저장 {saved}, 실패 {fails}, 누적 {len(done) + saved}/{len(items)}')
""")

md(r"""### 4-A. Claude 실행 (Anthropic 키만 사용)
이 셀은 Anthropic API만 부른다. OpenAI 쪽 오류와 무관하다. `max_retries=0`으로 SDK 자체 재시도를 꺼서, 재시도는 위의 규칙으로만 일어난다.""")
code(r"""
A = anthropic.Anthropic(api_key=get_key('ANTHROPIC_API_KEY'), max_retries=0)
for model in CLAUDE_MODELS:
    run_model('anthropic', model)
""")

md(r"""### 4-B. GPT 실행 (OpenAI 키만 사용)
먼저 아래 첫 셀에서 계정이 쓸 수 있는 모델 목록을 보고, 둘째 셀의 `GPT_FAST`, `GPT_MAIN`을 정한다. 목록 조회는 과금되지 않는다.
추론(reasoning) 모델은 생각하는 동안 첫 글자가 늦게 나오고 출력 예산을 생각에 쓸 수 있으므로, RAG 서비스처럼 추론을 최소로 두려면 `GPT_EXTRA`에 설정을 넣는다(예: `{'reasoning_effort': 'minimal'}`, 모델이 지원할 때만).""")
code(r"""
O = openai.OpenAI(api_key=get_key('OPENAI_API_KEY'), max_retries=0)
try:
    print('OpenAI 사용 가능 모델:', sorted(m.id for m in O.models.list() if m.id.startswith(('gpt', 'o'))))
except Exception as e:
    print('모델 목록 조회 실패:', e)
""")
code(r"""
GPT_FAST, GPT_MAIN = 'gpt-5.4-mini', 'gpt-5.5'      # ← 위 목록을 보고 바꿀 것
GPT_EXTRA = {}                                       # 예: {'reasoning_effort': 'minimal'}
for model in [GPT_FAST, GPT_MAIN]:
    run_model('openai', model)
""")

md(r"""## 5. 분석 — 응답·문장·4토큰 생성 시간""")
code(r"""
ENC = tiktoken.get_encoding('cl100k_base')
seg = pysbd.Segmenter(language='en', clean=False, char_span=True)
JUDGE = {'GPU p50': 17.6, 'GPU p95': 20.6, 'CPU k8 p50': 96.0, 'CPU k8 p95': 152.0}   # S3 실측 (SSOT §5)

def char_arrival(chunks):
    # 글자 i가 도착한 시각
    t = []
    for ts, txt in chunks:
        t += [ts] * len(txt)
    return t

def analyze(rec):
    chunks = rec['chunks']
    if not chunks:
        return None, []
    text = ''.join(c[1] for c in chunks)
    arr = char_arrival(chunks)
    ttft, total = chunks[0][0], chunks[-1][0]
    sents = []
    for sp in seg.segment(text):
        if not sp.sent.strip(): continue
        a = sp.start + (len(text[sp.start:sp.end]) - len(text[sp.start:sp.end].lstrip()))
        b = sp.start + len(text[sp.start:sp.end].rstrip())
        if b > a:
            sents.append({'gen_ms': arr[b - 1] - arr[a], 'n_chars': b - a, 'first': not sents,
                          'end_ms': arr[b - 1]})
    # 4토큰 생성 시간: cl100k 토큰 경계 기준, 응답 전체에서 연속된 4토큰 묶음마다
    ids = ENC.encode(text)
    dec, starts = ENC.decode_with_offsets(ids)
    four = []
    if dec == text and len(starts) > 4:
        for i in range(0, len(starts) - 4, 4):
            four.append(arr[min(starts[i + 4], len(arr) - 1)] - arr[starts[i]])
    gen = total - ttft
    resp = {'id': rec['id'], 'task_type': rec['task_type'], 'model': rec['model'], 'ttft_ms': ttft, 'total_ms': total,
            'n_chars': len(text), 'n_chunks': len(chunks), 'output_tokens': rec.get('output_tokens'),
            'ms_per_char': gen / max(len(text), 1), 'n_sents': len(sents),
            'first_sent_done_ms': sents[0]['end_ms'] if sents else None,
            'four_tok_ms_med': float(np.median(four)) if four else None, 'stop': rec.get('stop')}
    return resp, [dict(s, model=rec['model'], id=rec['id']) for s in sents]

R, S = [], []
for path in sorted(glob.glob(os.path.join(OUT_DIR, 'raw_*.jsonl'))):      # 실행된 모델만 분석
    for l in open(path):
        r, s = analyze(json.loads(l))
        if r: R.append(r); S += s
R, S = pd.DataFrame(R), pd.DataFrame(S)
q = lambda p: (lambda x: np.nanpercentile(x, p))
summ = R.groupby('model').agg(n=('id', 'count'), ttft_p50=('ttft_ms', 'median'), ttft_p95=('ttft_ms', q(95)),
                              total_p50=('total_ms', 'median'), total_p95=('total_ms', q(95)),
                              chars_p50=('n_chars', 'median'), ms_per_char_p50=('ms_per_char', 'median'),
                              sents_p50=('n_sents', 'median'), first_sent_done_p50=('first_sent_done_ms', 'median'),
                              four_tok_p50=('four_tok_ms_med', 'median'))
ss = S.groupby('model').gen_ms.agg(sent_gen_p5=q(5), sent_gen_p50='median', sent_gen_p95=q(95))
summ = summ.join(ss)
pd.set_option('display.width', 250)
print('=== 모델별 생성 시간 (ms) ===')
print(summ.round(1).to_string())
print('\n종료 사유:', R.groupby('model').stop.value_counts().to_dict())
""")

md("## 6. 판정 시간과 비교 — 사전 점검 P1·P2, 비율 보고")
code(r"""
p95 = JUDGE['GPU p95']
P1 = all(summ.sent_gen_p5 > p95)
P2 = all(summ.four_tok_p50 > p95)
print(f"P1 문장 판정이 밀리지 않는가 (GPU 판정 p95 {p95}ms < 문장 생성 시간 하위 5%): {'충족' if P1 else '미달'}")
for m, r in summ.iterrows():
    print(f'   {m:28} 문장 생성 하위 5% {r.sent_gen_p5:7.1f} ms')
print(f"P2 앞부분 재판정이 밀리지 않는가 (GPU 판정 p95 < 4토큰 생성 중앙값): {'충족' if P2 else '미달'}")
for m, r in summ.iterrows():
    print(f'   {m:28} 4토큰 생성 중앙값 {r.four_tok_p50:7.1f} ms')

rows = []
for m, r in summ.iterrows():
    rows.append({'model': m,
                 '판정/문장생성 (GPU p50)': JUDGE['GPU p50'] / r.sent_gen_p50,
                 '판정/문장생성 (CPU k8 p50)': JUDGE['CPU k8 p50'] / r.sent_gen_p50,
                 '응답당 판정 합/전체 (GPU, 문장 끝)': JUDGE['GPU p50'] * r.sents_p50 / r.total_p50,
                 # hold 예상 추가 지연: 문장 끝 방식 = 첫 문장이 끝날 때까지 + 판정, 앞부분 방식 = 4토큰 + 판정
                 'hold 문장끝 추가지연 ms': r.first_sent_done_p50 - r.ttft_p50 + JUDGE['GPU p50'],
                 'hold 4토큰 추가지연 ms': r.four_tok_p50 + JUDGE['GPU p50'],
                 'hold 문장끝 / 전체 응답': (r.first_sent_done_p50 - r.ttft_p50 + JUDGE['GPU p50']) / r.total_p50,
                 'hold 4토큰 / 전체 응답': (r.four_tok_p50 + JUDGE['GPU p50']) / r.total_p50})
ratio = pd.DataFrame(rows).set_index('model')
print('\n=== 비율 (판정 없음, 보고용) ===')
print(ratio.round(3).to_string())
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for m, g in S.groupby('model'):
    ax[0].hist(g.gen_ms.clip(upper=3000), bins=40, histtype='step', label=m)
ax[0].axvline(JUDGE['GPU p95'], color='k', ls='--', label='judge GPU p95'); ax[0].axvline(JUDGE['CPU k8 p95'], color='gray', ls=':', label='judge CPU p95')
ax[0].set_xlabel('sentence generation time (ms)'); ax[0].set_ylabel('sentences'); ax[0].legend(fontsize=7)
x = np.arange(len(summ)); w = 0.25
ax[1].bar(x - w, summ.ttft_p50, w, label='TTFT p50'); ax[1].bar(x, summ.total_p50, w, label='total p50')
ax[1].bar(x + w, summ.sent_gen_p50, w, label='sentence gen p50')
ax[1].set_xticks(x); ax[1].set_xticklabels(summ.index, rotation=15, fontsize=8); ax[1].set_ylabel('ms'); ax[1].legend()
plt.tight_layout(); plt.show()
""")

md(r"""## 7. 저장 — 게이트웨이 재측정용 실측 속도 파일

`timing_<모델>.jsonl`(응답마다 첫 토큰 시간, 글자당 생성 시간)을 만든다. `scripts/bench.py --timing`에 넣으면
가짜 LLM이 RAGTruth 응답을 **이 모델의 실제 속도**로 흘려보낸다(E5-lite 8절).""")
code(r"""
for m, g in R.groupby('model'):
    with open(os.path.join(OUT_DIR, f'timing_{m}.jsonl'), 'w') as f:
        for r in g.itertuples():
            f.write(json.dumps({'id': r.id, 'ttft_ms': r.ttft_ms, 'ms_per_char': r.ms_per_char, 'total_ms': r.total_ms}) + '\n')
R.to_csv(os.path.join(OUT_DIR, 'l1_responses.csv'), index=False)
S.to_csv(os.path.join(OUT_DIR, 'l1_sentences.csv'), index=False)
summ.to_csv(os.path.join(OUT_DIR, 'l1_summary.csv'))
ratio.to_csv(os.path.join(OUT_DIR, 'l1_ratio.csv'))
tok = R.groupby('model')[['output_tokens']].sum()
print('출력 토큰 합계 (비용 확인용):', tok.output_tokens.to_dict())
print('저장:', OUT_DIR)
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(ROOT, 'experiments', 'notebooks', 'l1_real_llm_timing.ipynb'), 'w'), ensure_ascii=False, indent=1)
print('cells', len(cells))
