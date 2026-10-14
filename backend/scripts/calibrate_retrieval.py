"""Collect dev candidates for human relevance labels; sweep a dense-score threshold offline."""
import argparse,json,hashlib
from pathlib import Path
import httpx

def sweep(rows):
    if not rows or any(type(r.get('relevant')) is not bool for r in rows):
        raise ValueError('Every candidate needs an explicit boolean relevant label after human review.')
    if not any(r['relevant'] for r in rows) or all(r['relevant'] for r in rows):
        raise ValueError('Calibration needs both relevant and irrelevant candidates.')
    results=[]
    for t in sorted({-1.0,1.000001,*[float(r['dense_score']) for r in rows]}):
        tp=sum(r['relevant'] and r['dense_score']>=t for r in rows);fp=sum(not r['relevant'] and r['dense_score']>=t for r in rows);fn=sum(r['relevant'] and r['dense_score']<t for r in rows)
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
        results.append({'threshold':t,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'tp':tp,'fp':fp,'fn':fn})
    best=max(results,key=lambda x:(x['f1'],x['precision'],x['threshold']))
    return {'recommended_dense_min_score':best['threshold'],'selected':best,'sweep':results,'labelled_candidates':len(rows),'warning':'Candidate-level dev F1 suggestion only. Inspect query-level false acceptance/abstention on held-out pilot queries before freezing. Never tune on final test questions.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['collect','score']);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--api',default='http://localhost:8000/api/v1');args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output file.')
    data=json.loads(args.input.read_text(encoding='utf-8-sig'))
    if args.command=='collect':
        rows=[]
        with httpx.Client(timeout=300,trust_env=False) as client:
            for q in data:
                r=client.post(args.api+'/retrieval/search',json={'query':q['question'],'top_k':5});r.raise_for_status()
                for c in r.json().get('candidates',[]):
                    rows.append({'question_id':q['question_id'],'group_id':q.get('group_id'),'question':q['question'],'chunk_id':c['chunk_id'],'document_id':c['document_id'],'dense_score':c['dense_score'],'text':c['text'],'relevant':None,'notes':''})
        result={'split':'pilot_dev','input_hash':hashlib.sha256(args.input.read_bytes()).hexdigest(),'rows':rows}
    else:
        if data.get('split')!='pilot_dev':raise ValueError('Expected a pilot_dev candidate label file.')
        result={**sweep(data['rows']),'labels_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output)
if __name__=='__main__':main()
