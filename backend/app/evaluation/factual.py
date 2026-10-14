"""Exact scoring; never extract correct-looking numbers from explanations."""
import json
import re
import unicodedata
from app.schemas.factual import FactualLabel, FactualResponse

SCORER_VERSION = 'factual_exact_v1'

def normalize(value: str) -> str:
    text = unicodedata.normalize('NFKC', value).casefold().strip()
    text = ''.join(str(unicodedata.digit(c)) if c.isdecimal() else c for c in text)
    return ' '.join(text.split())

def parse_response(raw: str) -> FactualResponse:
    text = raw.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.S)
        if not match: raise ValueError('Incomplete JSON fence.')
        text = match.group(1)
    def unique_keys(pairs):
        out = {}
        for k, v in pairs:
            if k in out: raise ValueError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    return FactualResponse.model_validate(json.loads(text, object_pairs_hook=unique_keys))

def score_answer(label: dict, raw: str | None, error: str | None = None) -> dict:
    gold = FactualLabel.model_validate(label)
    result = {'scorer_version': SCORER_VERSION, 'correct': 0, 'valid_output': False,
              'request_failed': error is not None, 'abstained': None, 'reason': error, 'parsed_answer': None}
    if error is not None: return result
    try: response = parse_response(raw or '')
    except (ValueError, TypeError) as exc:
        result['reason'] = 'Invalid factual response: ' + str(exc)
        return result
    result.update(valid_output=True, abstained=response.abstained, parsed_answer=response.model_dump())
    if response.abstained:
        result['reason'] = 'Abstained on an answerable factual question.'
        return result
    got, expected = response.answer, gold.canonical_answer
    if normalize(got.act_id) != normalize(expected.act_id) or got.provision_type != expected.provision_type:
        result['reason'] = 'Wrong Act or provision type.'
        return result
    values, wanted = [normalize(v) for v in got.values], [normalize(v) for v in expected.values]
    if gold.answer_type == 'short_text':
        correct = len(values) == 1 and values[0] in {wanted[0], *map(normalize, gold.accepted_aliases)}
    elif gold.answer_type == 'provision_set':
        correct = set(values) == set(wanted) and len(values) == len(set(values))
    else: correct = values == wanted
    result.update(correct=int(correct), reason='Exact canonical/verified-alias match.' if correct else 'Canonical answer mismatch.')
    return result

def public_task(question: dict) -> dict:
    """Explicit allow-list excludes reference, gold values, source IDs, quotes and aliases."""
    label = FactualLabel.model_validate(question['factual'])
    return {'question': question['question'], 'language': question['language'], 'answer_type': label.answer_type,
            'act_id': label.act_id, 'act_name': label.act_name, 'source_version': label.source_version}
