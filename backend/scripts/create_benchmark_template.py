import csv
from pathlib import Path
OUT=Path('data/datasets/legalqa_60_template.csv')
DOMAINS=['fundamental_rights']*2+['criminal_law']*3+['consumer_law']*3+['cyber_law']*3+['family_law']*2+['property_law']*2+['labour_law']*2+['general_legal_procedure']*3
langs=[('EN','en'),('HI','hi'),('HG','hinglish')]
fields=['question_id','group_id','language','domain','difficulty','question','reference_answer','gold_document_ids','gold_chunk_ids','answerable_from_kb','requires_multiple_sources','notes']
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for i,domain in enumerate(DOMAINS,1):
        g=f'QG{i:03d}'
        for suffix,lang in langs:
            w.writerow({'question_id':f'{g}_{suffix}','group_id':g,'language':lang,'domain':domain,'difficulty':'','question':'','reference_answer':'','gold_document_ids':'','gold_chunk_ids':'','answerable_from_kb':'true','requires_multiple_sources':'false','notes':'Fill only after legal-source verification.'})
print(OUT)
