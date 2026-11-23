"""문장 판정기.

입력 방식은 판정기 폴더의 `judge_config.json`이 정한다 (SPEC §4).
  retrieval (F1 base): 근거 → 40단어 조각 → 임베딩 (요청당 1회, prepare)
                       판정 텍스트 → 임베딩 → 유사한 조각 top-k (원문 순서) → 교차 인코더 [근거 조각, 문장 앞부분]
  full      (I2식)    : 교차 인코더 [근거 전체, 응답 처음부터 판정 지점까지]. 근거 검색 없음
점수는 logit(클수록 환각). 임계값은 재판정 간격(stride)별로 설정 파일의 값을 쓴다.
설정 파일이 "type": "combo"이면 하위 판정기 여러 개의 표준화 점수를 평균하는 ComboJudge를 만든다 (배포 판정기 I3).
"""
import contextlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    def score(self, ref, text, prior='', stream=None):
        """logit 하나. stream(CUDA 스트림)을 주면 그 스트림에서 계산하고 그 스트림만 기다린다(조합 판정기의 동시 실행용)."""
        ctx = torch.cuda.stream(stream) if (stream is not None and self.device == 'cuda') else contextlib.nullcontext()
        with self._lock, ctx:
            a, b = self._pair(ref, text, prior)
            enc = self.tok([a], [b], truncation='only_first', max_length=self.max_len,
                           return_tensors='pt').to(self.device)
            with torch.autocast(self.device, enabled=(self.device == 'cuda')):
                score = float(self.model(**enc).logits.squeeze().float().cpu())   # .cpu()가 현재 스트림을 기다림
            if self.device == 'cuda' and stream is None:
                torch.cuda.synchronize()
        return score

    def judge(self, ref, text, prior='', threshold=None):
        """text: 판정할 문장(또는 문장 앞부분, 앞뒤 공백 제거), prior: 응답에서 그 문장 앞까지의 원문."""
        t0 = time.perf_counter()
        score = self.score(ref, text, prior)
        thr = self.threshold if threshold is None else threshold
        return Verdict(score, score >= thr, (time.perf_counter() - t0) * 1000)


@dataclass
class ComboRef:
    refs: list            # 하위 판정기마다 RefIndex (members 순서)
    prepare_ms: float


class ComboJudge:
    """하위 판정기 점수를 (score − mean) / std로 표준화해 평균한다 (SPEC §4, 노트북 I3·L3의 combine과 같은 식).
    판정 1회에서 하위 판정기를 동시에 돌린다(스레드, GPU면 판정기마다 별도 CUDA 스트림)."""

    mode = 'combo'

    def __init__(self, members, thresholds=None, threshold=0.0, threshold_key='svcB', default_stride=0):
        # members: [(이름, 판정기, mean, std), ...]
        assert members and all(sd > 0 for _, _, _, sd in members), '하위 판정기와 양수 표준편차가 필요'
        self.members = members
        self.thresholds, self.threshold, self.threshold_key = thresholds or {}, threshold, threshold_key
        self.default_stride = default_stride
        self._pool = ThreadPoolExecutor(len(members), thread_name_prefix='combo')
        self._streams = [torch.cuda.Stream() if getattr(j, 'device', '') == 'cuda' else None for _, j, _, _ in members]

    @property
    def embedder(self):
        # 컬렉션 검색 임베더를 비웠을 때 같이 쓸 임베더: retrieval 방식 하위 판정기의 것
        return next((e for _, j, _, _ in self.members for e in [getattr(j, 'embedder', None)] if hasattr(e, 'encode')), None)

    def threshold_for(self, stride, key=None):
        return (self.thresholds.get(_stride_key(stride)) or {}).get(key or self.threshold_key, self.threshold)

    def prepare(self, reference):
        t0 = time.perf_counter()
        refs = [j.prepare(reference) for _, j, _, _ in self.members]
        return ComboRef(refs, (time.perf_counter() - t0) * 1000)

    @staticmethod
    def _member_score(j, ref, text, prior, stream):
        if hasattr(j, 'score'):
            return j.score(ref, text, prior, stream=stream)
        return j.judge(ref, text, prior).score                  # 대체 판정기(테스트)

    def judge(self, ref, text, prior='', threshold=None):
        t0 = time.perf_counter()
        futs = [self._pool.submit(self._member_score, j, r, text, prior, st)
                for (_, j, _, _), r, st in zip(self.members, ref.refs, self._streams)]
        raw = [f.result() for f in futs]                         # 하위 판정기 예외는 그대로 올라가 fail-open으로 처리됨
        score = sum((x - m) / sd for x, (_, _, m, sd) in zip(raw, self.members)) / len(self.members)
        thr = self.threshold if threshold is None else threshold
        return Verdict(score, score >= thr, (time.perf_counter() - t0) * 1000)


def _load_combo(settings, cfg):
    base = settings.judge_model_dir
    std, members = cfg['standardize'], []
    for name, sub in cfg['members'].items():
        d = sub if os.path.isabs(sub) else os.path.join(base, sub)
        if not os.path.isfile(os.path.join(d, 'config.json')):
            raise FileNotFoundError(f'조합 하위 판정기 {name} 폴더가 없음: {d}')
        mcfg = read_config(d)
        j = StudentJudge(d, embedder=mcfg.get('embedder', 'sentence-transformers/all-MiniLM-L6-v2'), k=mcfg.get('k', 8),
                         max_len=mcfg.get('max_len', 512), chunk_words=mcfg.get('chunk_words', 40),
                         device=settings.judge_device, mode=input_mode(mcfg))
        members.append((name, j, float(std[name]['mean']), float(std[name]['std'])))
    dflt = cfg.get('default') or {}
    kind = dflt.get('kind', 'end')
    fixed = settings.judge_threshold
    return ComboJudge(members, thresholds={} if fixed is not None else cfg.get('thresholds', {}),
                      threshold=fixed if fixed is not None else cfg.get('threshold', 0.0),
                      threshold_key=dflt.get('threshold', 'svcB'), default_stride=0 if kind == 'end' else int(kind))


def load_judge(settings):
    """판정기 폴더의 judge_config.json을 읽어 StudentJudge(또는 "type": "combo"면 ComboJudge)를 만든다.
    환경 변수로 지정한 값이 설정 파일보다 우선한다."""
    cfg = read_config(settings.judge_model_dir)
    if cfg.get('type') == 'combo':
        return _load_combo(settings, cfg)
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
