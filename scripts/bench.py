"""게이트웨이 종단 간(end-to-end) 측정.

가짜 LLM(mock_upstream)이 RAGTruth dev 응답을 실제 API와 비슷한 속도로 흘려보내고,
게이트웨이를 거쳐 받은 결과를 모드별로 비교한다.

  off  : 검증 없음 (기준선)
  pass : 즉시 전달 + 사후 판정
  hold : 문장 단위 보류 후 판정·전달

측정 항목
  [정확성] 받은 본문이 원래 응답과 글자 단위로 같은지 (게이트웨이가 텍스트를 망가뜨리지 않는지)
  [판별력] 게이트웨이 안에서 나온 문장 점수의 AUROC — 오프라인 실험(S2-b)과 같은 수준인지
  [지연]   첫 글자까지 시간(TTFT), 글자당 평균 지연(off 대비), 판정 소요 시간, pass 모드 판정 도착 지연
  [누출]   pass 모드에서 실제 환각 문장을 판정하기 전까지 사용자에게 이미 나간 글자 수

  python scripts/bench.py --replay replay_dev.jsonl --gateway http://127.0.0.1:8000 \
      --metrics gateway_metrics.jsonl --out bench_out
"""
import argparse
import asyncio
import json
import os
import statistics as st
import time

import httpx


def auroc(y, s):
    pos = [v for v, t in zip(s, y) if t]
    neg = [v for v, t in zip(s, y) if not t]
    if not pos or not neg:
        return float('nan')
    ranked = sorted([(v, i) for i, v in enumerate(s)])
    ranks, i = [0.0] * len(s), 0
    while i < len(ranked):                      # 동점 평균 순위
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(r for r, t in zip(ranks, y) if t)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float('nan')
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def parse_cfg(cfg):
    """'pass' → ('pass', 0), 'hold:4' → ('hold', 4). 콜론 뒤 숫자는 앞부분 재판정 간격(stride)."""
    m, _, s = cfg.partition(':')
    return m, int(s or 0)


def item_timing(item, a):
    """재생 속도. --timing 파일이 있으면 그 응답의 실측 값(L1: 실제 LLM의 첫 토큰 시간, 글자당 생성 시간)을 쓴다.
    글자당 시간 × (이 응답의 평균 글자/토큰)으로 토큰 간격을 만든다 — 가짜 LLM은 GPT 토큰 단위로 흘려보낸다."""
    t = TIMING.get(item['id'])
    if not t:
        return a.ttft_ms, a.token_ms
    n_tok = len(_ENC.encode(item['response'])) if _ENC else max(1, len(item['response']) / 4.0)
    return t['ttft_ms'], t['ms_per_char'] * len(item['response']) / max(n_tok, 1)


async def one(client, gw, item, mode, a):
    vmode, stride = parse_cfg(mode)
    ttft_ms, token_ms = item_timing(item, a)
    body = {'model': 'mock', 'stream': True,
            'messages': [{'role': 'system', 'content': 'Answer using only the provided context.'},
                         {'role': 'user', 'content': 'Respond based on the context.'}],
            'verification': {'context': item['reference'], 'mode': vmode, 'action': 'flag', 'stride': stride},
            'mock_response': item['response'], 'mock_ttft_ms': ttft_ms, 'mock_token_ms': token_ms}
    tag = f"{item['id']}|{mode}"
    t0 = time.perf_counter()
    timeline, text, verdicts = [], '', []
    async with client.stream('POST', f'{gw}/v1/chat/completions', json=body,
                             headers={'x-request-tag': tag}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith('data:') or line.endswith('[DONE]'):
                continue
            c = json.loads(line[5:])
            now = (time.perf_counter() - t0) * 1000
            if c.get('verification'):
                verdicts.append({**c['verification'], 't_client': now})
            for ch in c.get('choices') or []:
                d = (ch.get('delta') or {}).get('content')
                if d:
                    text += d
                    timeline.append((now, len(text)))
    return {'id': item['id'], 'task_type': item['task_type'], 'mode': mode, 'tag': tag,
            'text_ok': text == item['response'], 'timeline': timeline, 'verdicts': verdicts,
            't_total': (time.perf_counter() - t0) * 1000}


def onset_metrics(rs, items):
    """환각 발생 시점 기준 지표 (SSOT 결정: 누출은 '환각 발생부터' 센다).

    발생 위치 θ = 응답에서 첫 환각 구간의 시작 글자. 클라이언트가 받은 판정 이벤트를 받은 순서대로 본다.
    - 탐지(엄격): 응답의 첫 경보(hallucinated=true)가 θ를 포함하는 텍스트를 판정한 것(upto > θ)일 때만 탐지.
      첫 경보가 θ 이전 텍스트에 대한 것이면 발생 전 오경보로 끝난 것(미탐지).
    - 탐지 지연(글자): 경보 시점에 판정한 텍스트의 끝 − θ (생성이 θ를 지나 얼마나 진행됐을 때 잡았나)
    - 발생 후 누출(글자): 경보가 나온 순간까지 사용자에게 나간 글자 중 θ 이후 글자 수
      미탐지면 θ 이후 전부가 누출된 것으로 보고 '기대 누출'에 포함
    - 멀쩡한 응답 표시율: 환각 없는 응답 중 경보가 한 번이라도 나간 비율
    """
    info = {it['id']: it for it in items}
    det, delay, leak_det, leak_all, clean_flag = [], [], [], [], []
    for r in rs:
        it = info[r['id']]
        alarms = [v for v in r['verdicts'] if v.get('hallucinated') and 'upto' in v]
        if not it['labels']:
            clean_flag.append(bool(alarms)); continue
        theta = min(l['start'] for l in it['labels'])
        after = len(it['response']) - theta
        first = alarms[0] if alarms else None
        if first is not None and first['upto'] > theta:
            det.append(True)
            delay.append(first['upto'] - theta)
            lk = max(0, first.get('released_at_verdict', 0) - theta)
            leak_det.append(lk); leak_all.append(lk)
        else:
            det.append(False); leak_all.append(after)
    n = max(len(det), 1)
    return {'onset_n_hallucinated': len(det), 'onset_recall_strict': sum(det) / n,
            'onset_detect_delay_chars_p50': pct(delay, 50),
            'onset_leak_chars_p50_detected': pct(leak_det, 50),
            'onset_leak_chars_mean_detected': st.mean(leak_det) if leak_det else float('nan'),
            'onset_expected_leak_chars_mean': st.mean(leak_all) if leak_all else float('nan'),
            'clean_resp_flagged': sum(clean_flag) / max(len(clean_flag), 1)}


def char_times(timeline, n):
    """글자 i가 사용자 화면에 나타난 시각 리스트."""
    out, j = [], 0
    for i in range(n):
        while j < len(timeline) and timeline[j][1] < i + 1:
            j += 1
        out.append(timeline[j][0] if j < len(timeline) else None)
    return out


TIMING = {}
try:
    import tiktoken
    _ENC = tiktoken.get_encoding('cl100k_base')
except Exception:          # 오프라인 등
    _ENC = None


async def main(a):
    items = [json.loads(l) for l in open(a.replay)][:a.n or None]
    if a.timing:
        for l in open(a.timing):
            t = json.loads(l)
            TIMING[str(t['id'])] = t
        print(f'실측 재생 속도 사용: {a.timing} ({len(TIMING)}개 응답)')
    if os.path.exists(a.metrics):
        os.remove(a.metrics)                      # 이번 측정분만 남기기 (게이트웨이와 같은 경로여야 함)
    runs = []
    async with httpx.AsyncClient(timeout=300) as client:
        for i, it in enumerate(items):
            for mode in a.modes:
                runs.append(await one(client, a.gateway, it, mode, a))
            if (i + 1) % 10 == 0:
                print(f'  {i + 1}/{len(items)} 응답 완료')
    await asyncio.sleep(0.5)
    recs = {}
    if os.path.exists(a.metrics):
        for l in open(a.metrics):
            r = json.loads(l)
            recs[r['tag']] = r

    by = {(r['id'], r['mode']): r for r in runs}
    labels = {it['id']: it['labels'] for it in items}
    lens = {it['id']: len(it['response']) for it in items}
    os.makedirs(a.out, exist_ok=True)
    summary, sent_rows = {}, []

    for mode in a.modes:
        rs = [r for r in runs if r['mode'] == mode]
        s = {'n_responses': len(rs), 'text_ok_rate': sum(r['text_ok'] for r in rs) / len(rs),
             'ttft_ms_p50': pct([r['timeline'][0][0] if r['timeline'] else None for r in rs], 50),
             'total_ms_p50': pct([r['t_total'] for r in rs], 50)}
        if mode != 'off' and 'off' in a.modes:
            delays = []
            for r in rs:
                off = by[(r['id'], 'off')]
                n = lens[r['id']]
                ct, co = char_times(r['timeline'], n), char_times(off['timeline'], n)
                delays += [x - y for x, y in zip(ct, co) if x is not None and y is not None]
            s['char_delay_ms_mean'] = st.mean(delays) if delays else float('nan')
            s['char_delay_ms_p95'] = pct(delays, 95)
            s['ttft_added_ms_p50'] = pct([(r['timeline'][0][0] - by[(r['id'], 'off')]['timeline'][0][0])
                                          for r in rs if r['timeline']], 50)
        if mode != 'off':
            rows = []
            prep = []
            for r in rs:
                rec = recs.get(r['tag'])
                if not rec:
                    continue
                prep.append(rec['prepare_ms'])
                for sr in rec['sentences']:
                    if sr.get('skipped') or 'score' not in sr:
                        continue
                    a0, a1 = sr['start'], sr['start'] + sr['n_chars']
                    y = int(any(min(a1, l['end']) > max(a0, l['start']) for l in labels[r['id']]))
                    rows.append({'id': r['id'], 'task_type': r['task_type'], 'mode': mode,
                                 'sentence_idx': sr['sentence_idx'], 'y': y, 'score': sr['score'],
                                 'hallucinated': sr['hallucinated'], 'judge_ms': sr['judge_ms'],
                                 'verdict_lag_ms': sr['t_verdict'] - sr['t_complete'],
                                 'release_delay_ms': (sr['t_release'] - sr['t_complete'])
                                 if sr.get('t_release') is not None else None,
                                 'leak_chars': sr.get('leak_chars')})
            sent_rows += rows
            ys = [x['y'] for x in rows]
            tp = sum(1 for x in rows if x['y'] and x['hallucinated'])
            s.update({'n_sentences': len(rows), 'pos_rate': sum(ys) / max(len(ys), 1),
                      'AUROC': auroc(ys, [x['score'] for x in rows]),
                      'recall_at_threshold': tp / max(sum(ys), 1),
                      'precision_at_threshold': tp / max(sum(x['hallucinated'] for x in rows), 1),
                      'judge_ms_p50': pct([x['judge_ms'] for x in rows], 50),
                      'judge_ms_p95': pct([x['judge_ms'] for x in rows], 95),
                      'verdict_lag_ms_p50': pct([x['verdict_lag_ms'] for x in rows], 50),
                      'verdict_lag_ms_p95': pct([x['verdict_lag_ms'] for x in rows], 95),
                      'prepare_ms_p50': pct(prep, 50)})
            if parse_cfg(mode)[0] == 'pass':
                # (구 지표) 실제 환각이면서 잡아낸 문장: 판정이 나올 때까지 그 문장 시작부터 사용자에게 나간 글자 수
                caught = [x['leak_chars'] for x in rows if x['y'] and x['hallucinated']]
                s['leak_chars_p50_on_caught'] = pct(caught, 50)
                s['leak_chars_p95_on_caught'] = pct(caught, 95)
            s.update(onset_metrics(rs, items))
        summary[mode] = s

    json.dump(summary, open(f'{a.out}/bench_summary.json', 'w'), indent=1, ensure_ascii=False)
    with open(f'{a.out}/bench_sentences.jsonl', 'w') as f:
        for row in sent_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    speed = f'실측 속도 {a.timing}' if a.timing else f'토큰 간격 {a.token_ms}ms, 첫 토큰 {a.ttft_ms}ms'
    print(f'\n=== 결과 ({len(items)}개 응답, {speed}) ===')
    for mode, s in summary.items():
        print(f'\n[{mode}]')
        for k, v in s.items():
            print(f'  {k:26} {v:.3f}' if isinstance(v, float) else f'  {k:26} {v}')
    print(f'\n저장: {a.out}/bench_summary.json, {a.out}/bench_sentences.jsonl')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replay', default='replay_dev.jsonl')
    p.add_argument('--gateway', default='http://127.0.0.1:8000')
    p.add_argument('--metrics', default='gateway_metrics.jsonl', help='게이트웨이의 METRICS_PATH와 같은 파일')
    p.add_argument('--modes', nargs='+', default=['off', 'pass', 'hold'],
                   help="검증 설정 목록. 'pass:4'처럼 콜론 뒤에 앞부분 재판정 간격(stride)을 붙일 수 있다")
    p.add_argument('--n', type=int, default=0, help='앞에서 n개만 (0=전체)')
    p.add_argument('--ttft-ms', type=float, default=300)
    p.add_argument('--token-ms', type=float, default=25)
    p.add_argument('--timing', default='', help='L1 실측 재생 속도 파일 (id, ttft_ms, ms_per_char). 없으면 고정값')
    p.add_argument('--out', default='bench_out')
    asyncio.run(main(p.parse_args()))
