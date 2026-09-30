"""문장 판정기.

S2/S2-b와 같은 방식:
  근거 문서 → 40단어 조각 → 조각 임베딩 (요청당 1회, prepare)
  문장 → 임베딩 → 유사한 조각 top-k (원문 순서로 정렬) → 교차 인코더 → logit (클수록 환각)
"""
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
    emb: object           # torch.Tensor [n_chunks, d]
    prepare_ms: float


def chunk_text(text, n):
    w = text.split()
    return [' '.join(w[i:i + n]) for i in range(0, max(len(w), 1), n)] or [text]


class StudentJudge:
    def __init__(self, model_dir, embedder='sentence-transformers/all-MiniLM-L6-v2', k=8, max_len=512,
                 chunk_words=40, threshold=0.0, device=''):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        if isinstance(embedder, str):
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(embedder, device=self.device)
        self.embedder = embedder
        self.k, self.max_len, self.chunk_words, self.threshold = k, max_len, chunk_words, threshold
        self._lock = threading.Lock()     # 모델 1개를 여러 요청이 공유하므로 직렬화

    def _embed(self, texts):
        return self.embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                                    show_progress_bar=False)

    def prepare(self, reference):
        t0 = time.perf_counter()
        chunks = chunk_text(reference, self.chunk_words)
        with self._lock:
            emb = self._embed(chunks)
        return RefIndex(chunks, emb, (time.perf_counter() - t0) * 1000)

    @torch.no_grad()
    def judge(self, ref, sentence):
        t0 = time.perf_counter()
        with self._lock:
            e = self._embed([sentence])[0]
            top = torch.topk(ref.emb @ e, min(self.k, ref.emb.shape[0])).indices.tolist()
            ctx = ' ... '.join(ref.chunks[i] for i in sorted(top))
            enc = self.tok([ctx], [sentence], truncation='only_first', max_length=self.max_len,
                           return_tensors='pt').to(self.device)
            with torch.autocast(self.device, enabled=(self.device == 'cuda')):
                score = float(self.model(**enc).logits.squeeze().float().cpu())
            if self.device == 'cuda':
                torch.cuda.synchronize()
        return Verdict(score, score >= self.threshold, (time.perf_counter() - t0) * 1000)


class LexicalJudge:
    """모델 없이 파이프라인을 시험하기 위한 대체 판정기 (테스트·데모 전용).
    근거에 없는 단어 비율을 점수로 쓴다. 판별력은 없다고 봐야 한다."""

    def __init__(self, threshold=0.5, delay_ms=0.0):
        self.threshold, self.delay_ms = threshold, delay_ms

    def prepare(self, reference):
        return RefIndex([reference], {w.lower().strip('.,') for w in reference.split()}, 0.0)

    def judge(self, ref, sentence):
        t0 = time.perf_counter()
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        words = [w.lower().strip('.,') for w in sentence.split() if w.strip('.,')]
        score = sum(w not in ref.emb for w in words) / max(len(words), 1)
        return Verdict(score, score >= self.threshold, (time.perf_counter() - t0) * 1000)
