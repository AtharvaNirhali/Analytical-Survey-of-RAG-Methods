#!/usr/bin/env python3
"""EXP-06: HyDE adapted to SciDocs citation-oriented queries; official BEIR evaluator.
No SciDocs relevance labels are used during retrieval. Uses cached EXP-04 BGE-M3 vectors.
"""
import argparse, csv, json, os, time
from pathlib import Path
from collections import Counter
import numpy as np
import requests
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

CHECKPOINT = 'BAAI/bge-m3'
PROMPT_TEMPLATE = ("You are preparing a scientific literature review. The following text is the "
                   "title of a research paper, not a question. Write one plausible, concise "
                   "abstract for a different scientific paper that this titled paper might cite "
                   "or consider closely related. Describe the research topic, methods, and "
                   "contributions using relevant technical terms. Do not claim to know the "
                   "actual citations or invent specific authors, DOI, or numerical findings. "
                   "Write only the hypothetical abstract (about 90-130 words).\n\n"
                   "Research paper title: {title}\n\nHypothetical related-paper abstract:")

def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/scidocs')
    p.add_argument('--output-dir',default='results/hyde')
    p.add_argument('--bge-dir',default='inputs/bge_m3',help='EXP-04 results folder with cached vectors and doc_ids.json')
    p.add_argument('--ollama-url',default='http://127.0.0.1:11434/api/generate')
    p.add_argument('--llm',default='llama3.2:1b',help='Installed Ollama model; check ollama list')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--max-new-tokens',type=int,default=180)
    p.add_argument('--temperature',type=float,default=0.0)
    p.add_argument('--query-maxlen',type=int,default=64)
    p.add_argument('--hyde-maxlen',type=int,default=256)
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--hypotheticals',type=int,default=1,help='Number of independently generated hypothetical texts per query')
    p.add_argument('--mix-original',action='store_true',help='Ablation: average original query and hypothetical vectors (not primary HyDE result)')
    p.add_argument('--regenerate',action='store_true',help='Ignore existing generation cache; overwrites it')
    p.add_argument('--limit',type=int,default=0,help='Smoke test only. Results MUST NOT be reported as full benchmark.')
    return p.parse_args()

def encode(model, tok, text, maxlen, device):
    batch=tok([text],truncation=True,max_length=maxlen,return_tensors='pt').to(device)
    with torch.inference_mode():
        if device.type=='cuda':
            with torch.autocast('cuda',dtype=torch.float16): result=model(**batch)
        else: result=model(**batch)
        return F.normalize(result.last_hidden_state[:,0,:].float(),dim=-1)[0]

def generate(session,url,model,prompt,temperature,max_tokens,seed):
    payload={'model':model,'prompt':prompt,'stream':False,
             'options':{'temperature':temperature,'num_predict':max_tokens,'seed':seed}}
    for attempt in range(3):
        try:
            response=session.post(url,json=payload,timeout=(15,240))
            response.raise_for_status()
            text=(response.json().get('response') or '').strip()
            if not text: raise ValueError('Ollama returned an empty response')
            return text
        except (requests.RequestException, ValueError) as error:
            if attempt==2: raise RuntimeError(f'Ollama generation failed: {error}') from error
            time.sleep(2*(attempt+1))

def main():
    a=cli()
    if a.top_k!=10: raise ValueError('This experiment evaluates top-10; keep --top-k 10.')
    if a.hypotheticals<1: raise ValueError('--hypotheticals must be >=1')
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    corpus,queries,qrels=GenericDataLoader(data_folder=a.data_dir).load(split='test')
    qids=[qid for qid in queries if qid in qrels]
    if a.limit: qids=qids[:a.limit]
    print(f'Loaded {len(corpus)} docs, {len(qids)} queries; full benchmark requires 1000.',flush=True)
    bge=Path(a.bge_dir)
    ids_path=bge/'doc_ids.json'
    vec_path=bge/f'doc_vectors_{len(corpus)}_maxlen256.npy'
    if not ids_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f'Missing BGE cache. Need {ids_path} and {vec_path}. Copy them from EXP-04 results.')
    doc_ids=json.loads(ids_path.read_text())
    if doc_ids!=sorted(corpus): raise ValueError('BGE doc_ids mismatch with SciDocs corpus; STOP to avoid invalid rankings.')
    vectors=np.load(vec_path,mmap_mode='r')
    if vectors.shape!=(len(doc_ids),1024): raise ValueError(f'Unexpected vector shape: {vectors.shape}')
    norms=np.linalg.norm(vectors,axis=1)
    if not np.allclose(norms,1.0,atol=0.03):
        raise ValueError('Cached document vectors are not normalized like EXP-04.')
    print('Reusing EXP-04 BGE-M3 vectors (model BAAI/bge-m3; doc maxlen 256).',flush=True)
    device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; use --device cpu')
    # Keep document matrix on CPU to avoid needless use of 4GB GPU VRAM; NumPy matmul is exact.
    doc_mat=np.asarray(vectors,dtype=np.float32)
    tok=AutoTokenizer.from_pretrained(CHECKPOINT)
    model=AutoModel.from_pretrained(CHECKPOINT).to(device).eval()
    print(f'Embedding model ready on {device}. Generating with Ollama model={a.llm}',flush=True)
    cache_path=out/'hypothetical_documents.jsonl'
    cached={}
    if cache_path.exists() and not a.regenerate:
        for line in cache_path.open():
            if line.strip():
                obj=json.loads(line)
                if obj.get('llm')==a.llm and obj.get('prompt_version')==1 and obj.get('temperature')==a.temperature and obj.get('max_new_tokens')==a.max_new_tokens:
                    cached[(obj['qid'],obj['sample'])]=obj['text']
    if a.regenerate: cache_path.write_text('')
    session=requests.Session()
    # Preflight Ollama rather than completing model download and crashing at first query.
    try:
        r=session.get(a.ollama_url.rsplit('/api/',1)[0]+'/api/tags',timeout=10)
        r.raise_for_status()
        installed=[x.get('name','') for x in r.json().get('models',[])]
        if not any(x==a.llm or x.startswith(a.llm+':') for x in installed):
            print(f'WARNING: {a.llm} not listed in Ollama models {installed}; try ollama pull {a.llm}',flush=True)
    except requests.RequestException as e:
        raise RuntimeError('Ollama is not responding. Start with: ollama serve') from e
    results={}; latencies=[]; generated=0
    runpath=out/'run.trec'; csvpath=out/'ranked_papers.csv'
    with cache_path.open('a',encoding='utf-8') as cache_f, runpath.open('w') as run_f, csvpath.open('w',newline='') as csv_f:
        writer=csv.writer(csv_f); writer.writerow(['query_id','query','rank','doc_id','score','title'])
        for qid in tqdm(qids,desc='HyDE queries'):
            title=queries[qid].strip()
            start=time.perf_counter(); hypos=[]
            for i in range(a.hypotheticals):
                key=(qid,i)
                if key in cached: hypo=cached[key]
                else:
                    hypo=generate(session,a.ollama_url,a.llm,PROMPT_TEMPLATE.format(title=title),a.temperature,a.max_new_tokens,42+i)
                    entry={'qid':qid,'sample':i,'text':hypo,'llm':a.llm,'prompt_version':1,
                           'temperature':a.temperature,'max_new_tokens':a.max_new_tokens}
                    cache_f.write(json.dumps(entry,ensure_ascii=False)+'\n'); cache_f.flush()
                    generated+=1
                hypos.append(hypo)
            # Original HyDE averages generated hypothetical document embeddings;
            # optional original-query vector is a clearly labeled ablation.
            embeds=[encode(model,tok,h,a.hyde_maxlen,device) for h in hypos]
            if a.mix_original: embeds.append(encode(model,tok,title,a.query_maxlen,device))
            vec=torch.stack(embeds).mean(0)
            vec=F.normalize(vec.float(),dim=0).cpu().numpy()
            scores=doc_mat@vec
            # Stable tie handling by doc ID sorting on exact score ties.
            top=np.argpartition(-scores,a.top_k-1)[:a.top_k]
            top=sorted(top,key=lambda idx:(-float(scores[idx]),doc_ids[idx]))
            results[qid]={}
            for rank,idx in enumerate(top,1):
                did=doc_ids[idx]; score=float(scores[idx]); results[qid][did]=score
                run_f.write(f'{qid} Q0 {did} {rank} {score:.10f} HyDE-BGE-M3\n')
                writer.writerow([qid,title,rank,did,f'{score:.10f}',corpus[did].get('title','')])
            if device.type=='cuda': torch.cuda.synchronize()
            latencies.append((time.perf_counter()-start)*1000)
    if len(qids)!=len(qrels):
        print('SMOKE TEST only: BEIR scores for subset are NOT full-benchmark scores.',flush=True)
        # Restrict qrels so BEIR does not silently average over the full set.
        eval_qrels={qid:qrels[qid] for qid in qids}
    else: eval_qrels=qrels
    ndcg,ap,recall,prec=EvaluateRetrieval.evaluate(eval_qrels,results,[10])
    mrr=EvaluateRetrieval.evaluate_custom(eval_qrels,results,[10],metric='mrr')
    metrics={'precision_at_10':prec['P@10'],'recall_at_10':recall['Recall@10'],
             'mrr_at_10':mrr['MRR@10'],'ndcg_at_10':ndcg['NDCG@10'],'map_at_10':ap['MAP@10'],
             'evaluated_queries':len(qids),'full_benchmark':len(qids)==len(qrels),
             'dataset':'BEIR SciDocs','corpus_size':len(doc_ids),
             'method':'HyDE-inspired hypothetical related-paper abstract + BGE-M3 dense retrieval',
             'checkpoint':CHECKPOINT,'llm':a.llm,'hypotheticals_per_query':a.hypotheticals,
             'temperature':a.temperature,'max_new_tokens':a.max_new_tokens,
             'mix_original_query':a.mix_original,'doc_maxlen':256,'hyde_maxlen':a.hyde_maxlen,
             'top_k':a.top_k,'device':str(device),'new_hypotheticals_generated_this_run':generated,
             'mean_query_latency_ms':round(float(np.mean(latencies)),3),
             'median_query_latency_ms':round(float(np.median(latencies)),3),
             'p95_query_latency_ms':round(float(np.percentile(latencies,95)),3),
             'evaluation':'Official BEIR EvaluateRetrieval (in-process)',
             'note':'Adaptation, not Gao et al exact reproduction; differs in LLM, encoder and paper-title task. Query latency includes generation if not cached; cache hits exclude generation.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (out/'beir_verified.json').write_text(json.dumps({k:metrics[k] for k in ['precision_at_10','recall_at_10','mrr_at_10','ndcg_at_10','map_at_10']},indent=2))
    (out/'protocol.json').write_text(json.dumps({'query_prompt':PROMPT_TEMPLATE,'llm':a.llm,
       'embedding_model':CHECKPOINT,'dataset':'BEIR SciDocs','qrels_used_for_retrieval':False,
       'llm_generation_is_local':True,'corpus_vectors_source':str(bge),
       'baseline':'EXP-04 BGE-M3 dense-only, nDCG@10 0.16253',
       'comparison_note':'The only intended model change is query transformation through HyDE; caching changes latency measurement.'},indent=2))
    print('\nBEIR EVALUATION\n'+json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__': main()
