import re
CITE_RE=re.compile(r'\[S(\d+)\]')
def citation_reference_validity(answer:str,evidence_count:int)->dict:
    cited=[int(x) for x in CITE_RE.findall(answer or '')]
    invalid=[x for x in cited if x<1 or x>evidence_count]
    return {'cited_source_numbers':cited,'has_citations':bool(cited),'invalid_source_numbers':invalid,'reference_validity':1.0 if cited and not invalid else (0.0 if invalid else None)}
