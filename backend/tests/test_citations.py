from app.evaluation.citations import citation_reference_validity

def test_citation_syntax_gate():
    x=citation_reference_validity('Supported [S1], not [S4].',2)
    assert x['invalid_source_numbers']==[4]
    assert x['reference_validity']==0.0
