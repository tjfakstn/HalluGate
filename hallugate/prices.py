"""사용자 LLM 추정 비용 (SPEC §10.10). 단가표는 prices.json — 운영자가 공개 가격표를 보고 갱신한다."""
import json
import os

_PATH = os.path.join(os.path.dirname(__file__), 'prices.json')
PRICES = json.load(open(_PATH, encoding='utf-8'))['models']


def price_for(model):
    """모델 이름의 가장 긴 접두어에 맞는 [입력, 출력] $/백만 토큰. 없으면 None."""
    if not model:
        return None
    keys = [k for k in PRICES if model.startswith(k)]
    return tuple(PRICES[max(keys, key=len)]) if keys else None


def cost_usd(model, tokens_in, tokens_out):
    p = price_for(model)
    if p is None or tokens_in is None or tokens_out is None:
        return None
    return tokens_in / 1e6 * p[0] + tokens_out / 1e6 * p[1]
