#!/usr/bin/env python3
"""EXP-05: SciDocs field-aware blending (paper-inspired), plus RRF ablation.
Not an exact reproduction of Sawarkar et al. 2024's OpenSearch query configurations.
"""
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from tqdm import tqdm

TOKEN_RE = re.compile(r"(?u)\b\w+\b")
def tok(s):
    return TOKEN_RE.findall((s or '').lower())

def read_trec(path):
    out = defaultdict(dict)
    with Path(path).open(encoding='utf8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip(): continue
            a = line.split()
            if len(a) < 6: raise ValueError(f'{path}:{line_num}: expected six TREC columns')
            qid, _, did, rank, score, tag = a[:6]
            if did in out[qid]: raise ValueError(f'Duplicate doc {did} for {qid} in {path}')
            out[qid][did] = float(score)
    return out

def ordered(run, qid):
    # stable tie-breakers; inputs may contain fewer than candidate_k results
    return [did for did, score in sorted(run.get(qid, {}).items(), key=lambda z: (-z[1], z[0]))]

def fuse_rankings(rankings, k=60, topk=10):
    scores = defaultdict(float)
    for ranking in rankings:
        for r, did in enumerate(ranking, 1):
            scores[did] += 1.0 / (k + r)
    ranked = sorted(scores.items(), key=lambda z: (-z[1], z[0]))[:topk]
    return dict(ranked)

def best_k(scores, docids, k):
    if k >= len(scores): idx = np.argsort(-scores, kind='stable')
    else:
        idx = np.argpartition(-scores, k-1)[:k]
        idx = idx[np.argsort(-scores[idx], kind='stable')]
    # Exclude zero-scored candidates; field retrieval falls back to combined below
    return [docids[i] for i in idx if scores[i] > 0]

def write_trec(path, run, name):
    with path.open('w', encoding='utf8') as f:
        for qid in sorted(run):
            for r, (did, score) in enumerate(sorted(run[qid].items(),key=lambda z:(-z[1],z[0])),1):
                f.write(f'{qid} Q0 {did} {r} {score:.12f} {name}\n')

def evaluate(run, qrels, outdir, name):
    k = [10]
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run, k)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, run, k, metric='mrr')
    result = {'precision_at_10': precision['P@10'], 'recall_at_10':recall['Recall@10'],
              'mrr_at_10':mrr['MRR@10'], 'ndcg_at_10':ndcg['NDCG@10'], 'map_at_10':_map['MAP@10'],
              'queries':len(qrels), 'method':name, 'evaluation':'BEIR EvaluateRetrieval'}
    (outdir/f'{name}_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    write_trec(outdir/f'{name}_run.trec',run,name)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/scidocs')
    ap.add_argument('--bm25-run',default='inputs/bm25_run.trec')
    ap.add_argument('--splade-run',default='inputs/splade_run.trec')
    ap.add_argument('--bge-run',default='inputs/bge_m3_run.trec')
    ap.add_argument('--output',default='results/blended')
    ap.add_argument('--candidate-k',type=int,default=100)
    ap.add_argument('--rrf-k',type=int,default=60)
    args=ap.parse_args()
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    for path in [args.bm25_run,args.splade_run,args.bge_run]:
        if not Path(path).is_file(): ap.error(f'Missing {path}. Copy the saved run.trec here (see README).')
    print('Loading SciDocs corpus, test queries and ground truth...')
    corpus,queries,qrels=GenericDataLoader(data_folder=args.data).load(split='test')
    qids=sorted(qrels)
    for qid in qids:
        if qid not in queries: raise RuntimeError(f'Missing query text: {qid}')
    print(f'Documents={len(corpus)}; test queries={len(qids)}')
    runs={x:read_trec(path) for x,path in [('BM25',args.bm25_run),('SPLADE',args.splade_run),('BGE_M3',args.bge_run)]}
    for name,run in runs.items():
        absent=set(qids)-set(run)
        if absent: raise ValueError(f'{name}: missing {len(absent)} test queries')
    # Field-based query styles: title-only and abstract-only BM25, and a best-fields blend.
    # These are approximations to the paper's query strategies, NOT OpenSearch best_fields.
    docids=list(corpus)
    print('Preparing title and abstract BM25 indexes (CPU)...')
    title_model=BM25Okapi([tok(corpus[d].get('title','')) or ['_empty_'] for d in docids])
    abstract_model=BM25Okapi([tok(corpus[d].get('text','')) or ['_empty_'] for d in docids])
    results={name:{} for name in ['bm25_best_fields','blended_three_rrf','blended_field_aware_rrf']}
    t0=time.perf_counter()
    for qid in tqdm(qids,desc='Field retrieval + fusion'):
        qt=tok(queries[qid])
        title=best_k(title_model.get_scores(qt),docids,args.candidate_k) if qt else []
        abstract=best_k(abstract_model.get_scores(qt),docids,args.candidate_k) if qt else []
        pre=[ordered(runs[name],qid) for name in ['BM25','SPLADE','BGE_M3']]
        results['bm25_best_fields'][qid]=fuse_rankings([title,abstract],args.rrf_k)
        results['blended_three_rrf'][qid]=fuse_rankings(pre,args.rrf_k)
        results['blended_field_aware_rrf'][qid]=fuse_rankings([title,abstract,*pre],args.rrf_k)
        # fill any thin queries with original BM25 candidates deterministically
        for name in results:
            if len(results[name][qid]) < 10:
                filled=results[name][qid]
                for did in pre[0]:
                    if did not in filled: filled[did]= -1.0 - 0.0001*len(filled)
                    if len(filled)==10:break
    duration=time.perf_counter()-t0
    for name, run in results.items():
        metrics=evaluate(run,qrels,outdir,name)
        print(name, json.dumps(metrics,indent=2))
    summary={'dataset':'BEIR SciDocs','top_k':10,'candidate_k':args.candidate_k,
             'fusion_rrf_k':args.rrf_k,'source_run_paths':{'bm25':args.bm25_run,'splade':args.splade_run,'bge_m3':args.bge_run},
             'elapsed_seconds_field_scoring_and_fusion':duration,
             'note':'Paper-inspired field-query variants with fixed equal-weight RRF. NOT exact Blended RAG 2024 reproduction. Source TREC input depth determines fusion recall; existing runs often contain top-10 only. No test label-based hyperparameter tuning.'}
    (outdir/'experimental_protocol.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved outputs in', outdir)

if __name__=='__main__':main()
