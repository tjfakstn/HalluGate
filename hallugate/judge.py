"""문장 판정기.

입력 방식은 판정기 폴더의 `judge_config.json`이 정한다 (SPEC §4).
  retrieval (F1 base): 근거 → 40단어 조각 → 임베딩 (요청당 1회, prepare)
                       판정 텍스트 → 임베딩 → 유사한 조각 top-k (원문 순서) → 교차 인코더 [근거 조각, 문장 앞부분]
  full      (I2식)    : 교차 인코더 [근거 전체, 응답 처음부터 판정 지점까지]. 근거 검색 없음
점수는 logit(클수록 환각). 임계값은 재판정 간격(stride)별로 설정 파일의 값을 쓴다.
"""
import json
import os
import threading
import time
from dataclasses import dataclass

import torch


@dataclass
class Verdict:
    score: float          # logit, 클수록 환각
    hallucinated: bool
    judge_ms: float       # 검색 + 판정 시간


@dataclass
class RefIndex:
    chunks: list
    emb: object           # torch.Tensor [n_chunks, d] (retrieval) / None (full)
    prepare_ms: float
    text: str = ''        # 근거 원문 (full 방식)


def chunk_text(text, n):
    w = text.split()
    return [' '.join(w[i:i + n]) for i in range(0, max(len(w), 1), n)] or [text]


def _stride_key(stride):
    return 'end' if not stride else str(stride)


def read_config(model_dir):
    p = os.path.join(model_dir, 'judge_config.json')
    return json.load(open(p)) if os.path.isfile(p) else {}


def input_mode(cfg):
    """설정 파일에서 입력 방식을 정한다. 명시(`input_mode`)가 없으면 I2 노트북 형식(retrieval: null)을 full로 본다."""
    if cfg.get('input_mode'):
        return cfg['input_mode']
    inp = cfg.get('input') or {}
    return 'full' if isinstance(inp, dict) and 'retrieval' in inp and inp['retrieval'] is None else 'retrieval'


class StudentJudge:
    def __init__(self, model_dir, embedder='sentence-transformers/all-MiniLM-L6-v2', k=8, max_len=512,
                 chunk_words=40, threshold=0.0, device='', mode='retrieval', thresholds=None, threshold_key='svc10_orig',
                 default_stride=0):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        cfg = AutoConfig.from_pretrained(model_dir)
        kw = {}
        if getattr(cfg, 'model_type', '') == 'modernbert':
            # 새 transformers는 reference_compile을 모델 인자로 받지 않으므로 설정에 넣는다
            if hasattr(cfg, 'reference_compile'):
                cfg.reference_compile = False
            kw['attn_implementation'] = 'sdpa'
        # 가중치가 fp16으로 저장돼 있어도 fp32로 올린다 (CPU 판정, 노트북 점수와 일치). GPU에서는 autocast가 반정밀도로 돌린다.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=cfg, **kw).float().to(self.device).eval()
        if mode == 'retrieval' and isinstance(embedder, str):
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(embedder, device=self.device)
        self.embedder = embedder
        self.k, self.max_len, self.chunk_words, self.threshold = k, max_len, chunk_words, threshold
        self.thresholds, self.threshold_key = thresholds or {}, threshold_key
        self.default_stride = default_stride
        self._lock = threading.Lock()     # 모델 1개를 여러 요청이 공유하므로 직렬화

    def threshold_for(self, stride, key=None):
        """재판정 간격별 임계값 (설정 파일 thresholds[간격][key]). 없으면 기본 임계값."""
        return (self.thresholds.get(_stride_key(stride)) or {}).get(key or self.threshold_key, self.threshold)

    def _embed(self, texts):
        return self.embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                                    show_progress_bar=False)

    def prepare(self, reference):
        t0 = time.perf_counter()
        if self.mode == 'full':
            return RefIndex([], None, (time.perf_counter() - t0) * 1000, reference)
        chunks = chunk_text(reference, self.chunk_words)
        with self._lock:
            emb = self._embed(chunks)
        return RefIndex(chunks, emb, (time.perf_counter() - t0) * 1000, reference)

    def _pair(self, ref, text, prior):
        if self.mode == 'full':
            return ref.text, (prior or '') + text
        e = self._embed([text])[0]
        top = torch.topk(ref.emb @ e, min(self.k, ref.emb.shape[0])).indices.tolist()
        return ' ... '.join(ref.chunks[i] for i in sorted(top)), text

    @torch.no_grad()
    def judge(self, ref, text, prior='', threshold=None):
        """text: 판정할 문장(또는 문장 앞부분, 앞뒤 공백 제거), prior: 응답에서 그 문장 앞까지의 원문."""
        t0 = time.perf_counter()
        with self._lock:
            a, b = self._pair(ref, text, prior)
            enc = self.tok([a], [b], truncation='only_first', max_length=self.max_len,
                           return_tensors='pt').to(self.device)
            with torch.autocast(self.device, enabled=(self.device == 'cuda')):
                score = float(self.model(**enc).logits.squeeze().float().cpu())
            if self.device == 'cuda':
                torch.cuda.synchronize()
        thr = self.threshold if threshold is None else threshold
        return Verdict(score, score >= thr, (time.perf_counter() - t0) * 1000)


def load_judge(settings):
    """판정기 폴더의 judge_config.json을 읽어 StudentJudge를 만든다. 환경 변수로 지정한 값이 설정 파일보다 우선한다."""
    cfg = read_config(settings.judge_model_dir)
    pick = lambda env, key, default: env if env is not None else cfg.get(key, default)
    dflt = (cfg.get('default') or {})
    kind = dflt.get('kind', 'end')
    return StudentJudge(settings.judge_model_dir,
                        embedder=pick(settings.judge_embedder, 'embedder', 'sentence-transformers/all-MiniLM-L6-v2'),
                        k=pick(settings.judge_k, 'k', 8), max_len=pick(settings.judge_max_len, 'max_len', 512),
                        chunk_words=pick(settings.judge_chunk_words, 'chunk_words', 40),
                        threshold=pick(settings.judge_threshold, 'threshold', 0.0), device=settings.judge_device,
                        mode=input_mode(cfg),
                        # 임계값을 환경 변수로 고정하면 간격별 임계값은 쓰지 않는다
                        thresholds={} if settings.judge_threshold is not None else cfg.get('thresholds', {}),
                        threshold_key=dflt.get('threshold', 'svc10_orig'),
                        default_stride=0 if kind == 'end' else int(kind))


class LexicalJudge:
    """모델 없이 파이프라인을 시험하기 위한 대체 판정기 (테스트·데모 전용).
    근거에 없는 단어 비율을 점수로 쓴다. 판별력은 없다고 봐야 한다."""

    default_stride = 0

    def __init__(self, threshold=0.5, delay_ms=0.0, fail=None):
        self.threshold, self.delay_ms = threshold, delay_ms
        self.fail = fail          # 테스트용: 이 함수가 True를 돌려주는 판정에서 예외를 낸다

    def threshold_for(self, stride, key=None):
        return self.threshold

    def prepare(self, reference):
        return RefIndex([reference], {w.lower().strip('.,') for w in reference.split()}, 0.0, reference)

    def judge(self, ref, text, prior='', threshold=None):
        t0 = time.perf_counter()
        if self.fail is not None and self.fail(text):
            raise RuntimeError('judge failure (test)')
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        words = [w.lower().strip('.,') for w in text.split() if w.strip('.,')]
        score = sum(w not in ref.emb for w in words) / max(len(words), 1)
        thr = self.threshold if threshold is None else threshold
        return Verdict(score, score >= thr, (time.perf_counter() - t0) * 1000)
