"""L2 노트북 생성: 긴 문맥 인코더(ModernBERT) 판정 지연 측정 — 후보 ①(근거 전체 + 앞 문장을 한 번에 보는 판정기)의 실행 가능성 관문.
학습 없이 (1) RAGTruth train의 근거+응답 길이 분포와 (2) 입력 길이별 판정 1회 지연만 잰다. 런타임마다(A100, T4) 따로 실행한다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(os.path.dirname(HERE), 'notebooks')

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n")})

md(r"""# L2 — 긴 문맥 인코더 판정 지연 (후보 ① 관문)

## 왜 하나

11/04 조사에서 가장 기대 이득이 큰 후보는 **근거 전체와 앞 문장을 한 번에 보는 긴 문맥 판정기**(ModernBERT, 최대 8,192토큰; LettuceDetect가 같은 백본)였다.
현재 판정기는 근거의 일부(512토큰)만 보고, 이것이 요약 과제 약점의 원인으로 의심된다. 그런데 스트리밍 중 8토큰마다 판정하려면
판정 1회가 **p95 ≤ 68ms**(8토큰 × 8.5ms, F1 배포 규칙)여야 하고, 긴 입력은 지연이 크게 늘 수 있다. 학습(수 시간)에 들어가기 전에 **지연만 먼저** 잰다.

## 무엇을 하나 (학습 없음)

1. RAGTruth **train**(`split == 'train'`만)의 근거+응답 길이를 ModernBERT 토큰으로 세어 과제별 분포(중앙값·p90·p95·최대)를 본다 → 필요한 입력 길이.
2. ModernBERT-base(149M)·large(395M)의 판정 1회(배치 1, 순전파만) 지연을 입력 길이 512 / 1,024 / 2,048 / 4,096 / 8,192에서 잰다. 비교: 현재 DeBERTa-v3-base @512.
   - bf16(미지원 GPU는 fp16), 어텐션은 sdpa. 분류 헤드는 임의 초기화(지연만 재므로 무관). 예열 10회 후 100회.

## 사전 정의 기준 (결과 보기 전, SSOT §6 L2에 고정)

- 필요 길이 **L_need** = train 근거+응답 길이의 **p90**(전체 과제), 2,048 단위로 올림.
- **L2-1 (관문)**: A100에서 ModernBERT-base 또는 large 중 하나라도 **L_need 길이에서 p95 ≤ 68ms** → 후보 ① 학습 노트북으로 진행. 둘 다 넘으면 ①은 "문장 끝에서만 긴 문맥 판정 + 문장 도중은 F1 base" 하이브리드로 설계를 바꾼다(재측정 없이 같은 표로 판단: 문장 끝 판정 예산은 평균 문장 길이 × 8.5ms).
- T4에서 실행한 결과는 배포 후보 GPU 참고로만 적는다(관문 판단은 A100).

> **런타임: A100** (선택: 같은 노트북을 T4로 한 번 더). 약 15분. 드라이브 불필요. 결과 `l2_latency.csv`·`l2_lengths.csv` 자동 다운로드.
""")

md("## 1. 환경")
code(r"""
!pip -q install -U "transformers>=4.48" sentencepiece
!git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git 2>/dev/null || echo 'RAGTruth already cloned'
import json, time, numpy as np, pandas as pd, torch, transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
DEVICE = 'cuda'
assert torch.cuda.is_available(), 'GPU 런타임 필요'
GPU = torch.cuda.get_device_name(0)
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(GPU, DTYPE, 'transformers', transformers.__version__)
""")

md("## 2. 근거+응답 길이 분포 (RAGTruth train만)")
code(r"""
BASE = 'RAGTruth/dataset'
responses = [json.loads(l) for l in open(f'{BASE}/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{BASE}/source_info.jsonl')}
train_pool = [r for r in responses if r['split'] == 'train']          # test 분할은 사용하지 않음
del responses

def build_reference(src):                                              # F1·R4c와 같은 근거 문자열
    si, tt = src['source_info'], src['task_type']
    if tt == 'QA':
        return f"Question: {si.get('question','')}\n{si.get('passages','')}"
    if tt == 'Summary':
        return si if isinstance(si, str) else json.dumps(si, ensure_ascii=False)
    return '\n'.join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}"
                     for k, v in (si.items() if isinstance(si, dict) else []))

MB_TOK = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
rows = []
for r in train_pool:
    src = sources[r['source_id']]
    n_ref = len(MB_TOK(build_reference(src), add_special_tokens=False)['input_ids'])
    n_resp = len(MB_TOK(r['response'], add_special_tokens=False)['input_ids'])
    rows.append({'task': src['task_type'], 'ref': n_ref, 'resp': n_resp, 'total': n_ref + n_resp + 3})
LEN = pd.DataFrame(rows)
q = lambda s: pd.Series({'n': len(s), 'p50': s.quantile(.5), 'p90': s.quantile(.9), 'p95': s.quantile(.95), 'max': s.max(),
                         'over512': (s > 512).mean(), 'over2048': (s > 2048).mean(), 'over4096': (s > 4096).mean()})
len_tab = pd.concat([LEN.groupby('task')['total'].apply(q).unstack(), q(LEN['total']).to_frame('all').T])
print(len_tab.round(3).to_string())
P90 = float(LEN['total'].quantile(.9))
L_NEED = int(min(8192, 2048 * np.ceil(P90 / 2048)))
print(f'L_need = p90 {P90:.0f} → {L_NEED}토큰')
""")

md("## 3. 입력 길이별 판정 1회 지연")
code(r"""
MODELS = {'ModernBERT-base': 'answerdotai/ModernBERT-base', 'ModernBERT-large': 'answerdotai/ModernBERT-large',
          'DeBERTa-v3-base (현재)': 'cross-encoder/nli-deberta-v3-base'}
LENS = [512, 1024, 2048, 4096, 8192]
N_WARM, N_RUN = 10, 100

def load(path):
    cfg = AutoConfig.from_pretrained(path, num_labels=1)
    kw = {'attn_implementation': 'sdpa'} if 'ModernBERT' in path else {}
    m = AutoModelForSequenceClassification.from_pretrained(path, config=cfg, ignore_mismatched_sizes=True, torch_dtype=DTYPE, **kw)
    return m.to(DEVICE).eval()

@torch.no_grad()
def bench(model, n, vocab):
    ids = torch.randint(5, vocab - 1, (1, n), device=DEVICE)
    att = torch.ones_like(ids)
    for _ in range(N_WARM): model(input_ids=ids, attention_mask=att)
    torch.cuda.synchronize(); ms = []
    for _ in range(N_RUN):
        t0 = time.perf_counter(); model(input_ids=ids, attention_mask=att); torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return np.percentile(ms, 50), np.percentile(ms, 95)

LAT = []
for name, path in MODELS.items():
    model = load(path)
    maxpos = getattr(model.config, 'max_position_embeddings', 512)
    for n in LENS:
        if n > maxpos: continue
        try:
            p50, p95 = bench(model, n, model.config.vocab_size)
        except torch.cuda.OutOfMemoryError:
            p50 = p95 = float('nan'); torch.cuda.empty_cache()
        LAT.append({'gpu': GPU, 'model': name, 'len': n, 'p50_ms': p50, 'p95_ms': p95})
        print(f'{name:24} {n:5}토큰: p50 {p50:6.1f} / p95 {p95:6.1f} ms')
    del model; torch.cuda.empty_cache()
LAT = pd.DataFrame(LAT)
""")

md("## 4. 관문 판정과 저장")
code(r"""
BUDGET = 68.0                     # ms, 8토큰 × 8.5ms (F1 배포 규칙)
cand = LAT[LAT.model.str.startswith('ModernBERT') & (LAT.len == L_NEED)]
ok = cand[cand.p95_ms <= BUDGET]
is_a100 = 'A100' in GPU
print(cand.round(1).to_string(index=False))
if is_a100:
    print(f"=== L2-1 관문 (A100, L_need {L_NEED}): {'충족 → ' + ', '.join(ok.model) + ' 로 후보 ① 학습 진행' if len(ok) else '미달 → 문장 끝 긴 문맥 + 문장 도중 F1 base 하이브리드로 설계 변경'} ===")
else:
    print(f'(참고) {GPU}: 관문 판단은 A100 결과로 한다. 위 표는 배포 후보 GPU 참고값')
sl = (LAT.p95_ms / 8.5).round(1)
LAT['tokens_of_budget'] = sl                       # 이 지연이 몇 토큰 생성 시간에 해당하는지 (8.5ms/토큰)
tag = GPU.replace(' ', '_')
LAT.to_csv(f'l2_latency_{tag}.csv', index=False); len_tab.to_csv('l2_lengths.csv')
json.dump({'gpu': GPU, 'L_need': L_NEED, 'p90_total': P90, 'gate_pass': bool(len(ok)) if is_a100 else None,
           'pass_models': list(ok.model)}, open(f'l2_summary_{tag}.json', 'w'), ensure_ascii=False, indent=1)
from google.colab import files
for f in [f'l2_latency_{tag}.csv', 'l2_lengths.csv', f'l2_summary_{tag}.json']:
    files.download(f)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(NB_DIR, 'l2_longctx_latency.ipynb')
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
