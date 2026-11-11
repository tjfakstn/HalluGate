"""M1a — M1 환각 탐지율이 낮은 원인 분석 (SSOT §6 M1a).

입력 폴더(드라이브 `m1/`에서 내려받은 것): cases_devL.json, gen_devL.json, label_devL.json, scores_devL.json
 + docs/analysis/m1/m1_devL.csv

  python experiments/analysis/m1a_review.py prepare <입력폴더> <작업폴더> [--parts 5]
      → 가린 검토 묶음 part_*.json (판정기 점수·표시 여부 없음, 응답 순서 섞음, 시드 11)
  python experiments/analysis/m1a_review.py merge <입력폴더> <작업폴더> [--i2-tok <i2 판정기 폴더>]
      → 검토 결과 review_*.json을 점수와 합쳐 m1a_sentences.csv, m1a_summary.json

원문(RAGTruth 근거·생성 응답)은 작업폴더에만 두고 커밋하지 않는다. 커밋은 요약·문장 단위 표(원문 제외)만.
"""
import argparse, glob, json, os, random, sys

import numpy as np, pandas as pd, pysbd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'builders'))
import build_m1 as B  # noqa: E402  (노트북 문자열만 가져옴, 노트북은 만들지 않음)

NS = {'json': json, 'SEG': pysbd.Segmenter(language='en', clean=False, char_span=True)}
exec(B.M1_LABEL, NS)
sentences, parse = NS['sentences'], NS['parse']
THR = B.L3.COMBO['threshold'] if isinstance(B.L3.COMBO, dict) else 2.991638706756592

VALIDITY = ['real', 'ambiguous', 'over']
TYPES = ['numeric', 'outside_fact', 'generalize', 'contradiction', 'other']


def load(d, name):
    return json.load(open(os.path.join(d, name)))


def labels_of(cases, gens, raw):
    out = {}
    for c in raw:
        if c not in gens: continue
        labs, ok = parse(raw[c]['text'], len(sentences(gens[c]['text'])))
        if ok: out[c] = labs
    return out


def prepare(a):
    cases, gens, raw = load(a.inp, 'cases_devL.json'), load(a.inp, 'gen_devL.json'), load(a.inp, 'label_devL.json')
    labels = labels_of(cases, gens, raw)
    items = []
    for c, labs in labels.items():
        ss = sentences(gens[c]['text'])
        hs = [i for i in range(1, len(ss) + 1) if labs[i]['label'] == 'H']
        if not hs: continue
        x = cases[c]
        items.append({'rid': None, 'cid': c, 'task': x['task'], 'reference': x['ref_label'],
                      'sentences': [f'[{i + 1}] {s}' for i, (_, _, s) in enumerate(ss)],
                      'opus_H': [{'i': i, 'type': labs[i]['type'], 'phrase': labs[i]['phrase'], 'reason': labs[i]['reason']} for i in hs]})
    print(f'환각 응답 {len(items)}개 (정답 파싱 {len(labels)}/{len(raw)})')
    random.Random(11).shuffle(items)
    key = {}
    for k, it in enumerate(items):
        it['rid'] = f'r{k + 1:03d}'; key[it['rid']] = it.pop('cid')   # 생성 모델 이름도 가림
    os.makedirs(a.work, exist_ok=True)
    json.dump(key, open(os.path.join(a.work, 'key.json'), 'w'), indent=1)
    n = a.parts
    for p in range(n):
        part = items[p::n]
        json.dump(part, open(os.path.join(a.work, f'part_{p + 1}.json'), 'w'), ensure_ascii=False, indent=1)
        print(f'part_{p + 1}.json: 응답 {len(part)}개, H 문장 {sum(len(x["opus_H"]) for x in part)}개')


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all(): return float('nan')
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def merge(a):
    cases, gens, raw = load(a.inp, 'cases_devL.json'), load(a.inp, 'gen_devL.json'), load(a.inp, 'label_devL.json')
    scores, key = load(a.inp, 'scores_devL.json'), load(a.work, 'key.json')
    labels = labels_of(cases, gens, raw)
    rev = {}
    for f in sorted(glob.glob(os.path.join(a.work, 'review_*.json'))):
        for e in json.load(open(f)):
            rev[(key[e['rid']], int(e['i']))] = e
    tok = None
    if a.i2_tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.i2_tok)
    rows = []
    for c, labs in labels.items():
        g = gens[c]['text']; ss = sentences(g); pts = scores.get(c, [])
        hs = [i for i in range(1, len(ss) + 1) if labs[i]['label'] == 'H']
        if not hs: continue
        first_alarm = next((p[0] for p in pts if p[4] >= THR), None)
        ref = cases[c]['ref_judge']
        ref_tok = len(tok(ref, add_special_tokens=False)['input_ids']) if tok else None
        resp_tok = len(tok(g, add_special_tokens=False)['input_ids']) if tok else None
        for i in hs:
            a0 = ss[i - 1][0]
            sp = [p for p in pts if p[0] == a0]
            r = rev.get((c, i), {})
            rows.append({'cid': c, 'gen': cases[c]['gen'], 'task': cases[c]['task'], 'i': i, 'n_sent': len(ss),
                         'pos': i / len(ss), 'opus_type': labs[i]['type'],
                         'validity': r.get('validity'), 'type': r.get('type'),
                         'sent_max_c': max([p[4] for p in sp], default=np.nan),
                         'sent_max_f': max([p[2] for p in sp], default=np.nan),
                         'sent_max_i': max([p[3] for p in sp], default=np.nan),
                         'sent_flag': bool(any(p[4] >= THR for p in sp)),
                         'resp_flag': first_alarm is not None,
                         'ref_tok': ref_tok, 'resp_tok': resp_tok})
    S = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    S.to_csv(os.path.join(a.out, 'm1a_sentences.csv'), index=False)
    missing = int(S.validity.isna().sum())
    summ = {'n_H_sent': len(S), 'n_resp': int(S.cid.nunique()), 'unreviewed': missing}
    summ['validity'] = S.validity.value_counts().to_dict()
    summ['validity_by_task'] = S.groupby('task').validity.value_counts().unstack(fill_value=0).to_dict('index')
    summ['over_rate'] = float((S.validity == 'over').mean())
    summ['sent_flag_by_validity'] = S.groupby('validity').sent_flag.agg(['sum', 'count']).to_dict('index')
    T = S[S.validity.isin(['real', 'ambiguous'])]
    summ['type'] = T.type.value_counts().to_dict()
    summ['sent_flag_by_type'] = T.groupby('type').sent_flag.agg(['sum', 'count']).to_dict('index')
    miss = S[(S.validity == 'real') & ~S.sent_flag]
    summ['missed_real_type'] = miss.type.value_counts().to_dict()
    summ['missed_real_n'] = len(miss)
    real_resp = S[S.validity == 'real'].groupby('cid').resp_flag.first()
    summ['real_only_resp_detect'] = {'k': int(real_resp.sum()), 'n': int(len(real_resp))}
    summ['sent_score_median_by_task'] = S.groupby('task')[['sent_max_c', 'sent_max_f', 'sent_max_i']].median().round(3).to_dict('index')
    if tok:
        S['i2_over'] = S.ref_tok + S.resp_tok > 2048
        summ['i2_over_2048_by_task'] = S.drop_duplicates('cid').groupby('task').i2_over.mean().round(3).to_dict()
    # ③ Data2txt: 응답 단위
    D = pd.read_csv(a.csv)
    D = D[D.status == 'ok'].copy(); D['hal'] = ~D.clean.astype(bool)
    d2 = D[D.task == 'Data2txt']
    summ['data2txt'] = {'auroc_max_c': auroc(d2.hal, d2.max_c), 'auroc_max_f': auroc(d2.hal, d2.max_f),
                        'auroc_n_sent': auroc(d2.hal, d2.n_sent),
                        'max_c_median': d2.groupby('hal').max_c.median().round(3).to_dict(),
                        'n_sent_median': d2.groupby('hal').n_sent.median().to_dict(),
                        'by_gen': {g: {'auroc_max_c': auroc(x.hal, x.max_c), 'n_hal': int(x.hal.sum()), 'n': len(x)} for g, x in d2.groupby('gen')}}
    json.dump(summ, open(os.path.join(a.out, 'm1a_summary.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'merge'])
    ap.add_argument('inp'); ap.add_argument('work')
    ap.add_argument('--parts', type=int, default=5)
    ap.add_argument('--i2-tok', default=None)
    ap.add_argument('--csv', default=os.path.join(ROOT, 'docs', 'analysis', 'm1', 'm1_devL.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'analysis', 'm1a'))
    a = ap.parse_args()
    prepare(a) if a.cmd == 'prepare' else merge(a)
