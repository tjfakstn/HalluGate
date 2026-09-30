"""RAGTruth dev 응답을 게이트웨이 재생용 JSONL로 만든다.

S1~S2-b와 같은 dev 표본(시드 42, 과제별 120개)에서 과제별 N개를 뽑는다.
각 줄: {id, task_type, reference, response, labels:[{start,end}]}

  python scripts/make_replay_set.py --ragtruth RAGTruth/dataset --per-task 20 --out replay_dev.jsonl
"""
import argparse
import json
import random

p = argparse.ArgumentParser()
p.add_argument('--ragtruth', default='RAGTruth/dataset')
p.add_argument('--per-task', type=int, default=20)
p.add_argument('--out', default='replay_dev.jsonl')
a = p.parse_args()

responses = [json.loads(l) for l in open(f'{a.ragtruth}/response.jsonl')]
sources = {json.loads(l)['source_id']: json.loads(l) for l in open(f'{a.ragtruth}/source_info.jsonl')}
for r in responses:
    r['task_type'] = sources[r['source_id']]['task_type']


def build_reference(src):          # S2와 동일
    si, tt = src['source_info'], src['task_type']
    if tt == 'QA':
        return f"Question: {si.get('question','')}\n{si.get('passages','')}"
    if tt == 'Summary':
        return si if isinstance(si, str) else json.dumps(si, ensure_ascii=False)
    return '\n'.join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}"
                     for k, v in (si.items() if isinstance(si, dict) else []))


TASKS = ['QA', 'Summary', 'Data2txt']
train_pool = [r for r in responses if r['split'] == 'train']
random.seed(42)
dev = []
for tt in TASKS:
    dev += random.sample([r for r in train_pool if r['task_type'] == tt], 120)

rows = []
for tt in TASKS:
    for r in [x for x in dev if x['task_type'] == tt][:a.per_task]:
        rows.append({'id': str(r['id']), 'task_type': tt, 'reference': build_reference(sources[r['source_id']]),
                     'response': r['response'],
                     'labels': [{'start': l['start'], 'end': l['end']} for l in (r.get('labels') or [])]})
with open(a.out, 'w') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
print(f'{len(rows)}개 응답 → {a.out}')
