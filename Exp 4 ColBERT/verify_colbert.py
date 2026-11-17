"""Official BEIR scoring on an existing ColBERTv2 TREC run; no GPU required."""
import argparse, json
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

p=argparse.ArgumentParser()
p.add_argument('--data', default='data/scidocs')
p.add_argument('--run', default='results/colbert/run.trec')
a=p.parse_args()
_, queries, qrels=GenericDataLoader(data_folder=a.data).load(split='test')
results={}
with open(a.run, encoding='utf-8') as f:
    for line in f:
        parts=line.split()
        if len(parts)!=6: raise ValueError('Malformed run.trec line: '+line[:100])
        qid,_,docid,rank,score,_=parts
        results.setdefault(qid,{})[docid]=float(score)
print('Expected queries:', len(qrels))
print('Queries in ColBERT results:',len(results))
print('Missing queries:',len(set(qrels)-set(results)))
ndcg,_map,recall,precision=EvaluateRetrieval.evaluate(qrels,results,[10])
mrr=EvaluateRetrieval.evaluate_custom(qrels,results,[10],metric='mrr')
metrics={'precision_at_10':precision['P@10'],'recall_at_10':recall['Recall@10'],
         'mrr_at_10':mrr['MRR@10'],'ndcg_at_10':ndcg['NDCG@10'],'map_at_10':_map['MAP@10']}
print(json.dumps(metrics,indent=2))
out=Path(a.run).parent/'beir_verified.json'
out.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
print('Saved',out)
