"""Offline paper tables and plots from a completed, exported factual experiment."""
import argparse,csv,json
from pathlib import Path

def export(bundle,out):
    exp=bundle['experiment'];a=exp.get('aggregate') or {}
    if exp.get('evaluation_kind')!='factual' or exp.get('status')!='completed' or not a.get('valid_for_reporting'):
        raise ValueError('Only completed factual comparisons can be exported as final paper results.')
    out.mkdir(parents=True,exist_ok=True)
    configs=a['conditions'];rows=[]
    for kind,sections in [('overall',{'all':a}),('language',a['by_language']),('domain',a['by_domain'])]:
        for label,section in sections.items():
            for c,v in section['systems'].items():
                ret=v.get('retrieval') or {}
                rows.append({'breakdown':kind,'label':label,'condition':c,'question_groups':section['group_count'],
                    'rows':section['count'],'accuracy_percent':100*v['factual_accuracy'],
                    'chunk_hit_at_1':ret.get('chunk_hit_at_1'),'chunk_hit_at_3':ret.get('chunk_hit_at_3'),
                    'chunk_hit_at_5':ret.get('chunk_hit_at_5'),'chunk_mrr':ret.get('chunk_mrr'),
                    'median_latency_seconds':v['median_seconds'],'failure_rate':v['failure_rate'],
                    'invalid_output_rate':v['invalid_output_rate'],'retry_count':v['retry_count']})
    with (out/'performance.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    items=bundle['items'];composition=[]
    for domain in sorted({r['domain'] for r in items}):
        selected=[r for r in items if r['domain']==domain]
        composition.append({'domain':domain,'groups':len({r['group_id'] for r in selected}),
                            **{l:sum(r['language']==l for r in selected) for l in ('en','hi','hinglish')}})
    with (out/'dataset_composition.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['domain','groups','en','hi','hinglish']);w.writeheader();w.writerows(composition)
    (out/'paired_comparisons.json').write_text(json.dumps(a['paired_comparisons'],indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(exp['manifest'],indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'llm_only':'LLM-only','rag_no_rerank':'RAG without reranking','rag_rerank':'RAG with reranking'}
    langs=[l for l in ('en','hi','hinglish') if l in a['by_language']]
    colors=['#64748b','#2563eb','#0e806d']
    fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained')
    for i,c in enumerate(configs):
        ax.bar([j+(i-1)*.25 for j in range(len(langs))],[100*a['by_language'][l]['systems'][c]['factual_accuracy'] for l in langs],.25,label=labels[c],color=colors[i])
    ax.set_xticks(range(len(langs)),[{'en':'English','hi':'Hindi','hinglish':'Hinglish'}[l] for l in langs]);ax.set_ylabel('Factual-answer accuracy (%)');ax.set_ylim(0,100);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'language_accuracy.png',dpi=300);fig.savefig(out/'language_accuracy.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained')
    for c,color in zip(configs,colors):
        v=a['systems'][c]
        ax.scatter(v['median_seconds'],100*v['factual_accuracy'],label=labels[c],color=color,s=65)
    ax.set_xlabel('Median response latency (seconds)');ax.set_ylabel('Factual-answer accuracy (%)');ax.set_ylim(0,100);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'accuracy_latency.png',dpi=300);fig.savefig(out/'accuracy_latency.svg');plt.close(fig)
    (out/'READ_ME.txt').write_text('These are measured factual-answer scores, not semantic legal correctness of explanations. Retrieval labels: '+', '.join(a['label_semantics'])+'. Latencies include natural cache/model reload effects. Report retries and use paired group confidence intervals. Missing CSV cells are N/A, not zero.\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    export(json.loads(args.bundle.read_text(encoding='utf-8-sig')),args.output);print('Paper exports:',args.output.resolve())
if __name__=='__main__':main()
