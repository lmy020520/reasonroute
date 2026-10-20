"""One offline annotation policy for independent training and evaluation data.

Reference answers and evidence metadata are annotation inputs only. They must
never be passed to the online query classifier or route gate.
"""
import hashlib
import inspect
import json
from functools import lru_cache

from tools import reclassify_public_eval as rules


POLICY = 'dataset_agnostic_rule_v1_shared'


@lru_cache(maxsize=1)
def policy_sha256():
    definitions = {name: getattr(rules, name) for name in (
        'EXPLANATION_PATTERNS', 'COMPARISON_PATTERNS', 'CHAIN_PATTERNS', 'DIRECT_FACT_STARTS')}
    payload = json.dumps(definitions, sort_keys=True) + ''.join(inspect.getsource(fn) for fn in (
        rules.flatten_answers, rules.token_len, rules.has_pattern, rules.support_stats,
        rules.classify_item, classify))
    return hashlib.sha256(payload.encode()).hexdigest()


def classify(row):
    answers = rules.flatten_answers(row.get('answers') or row.get('answer'))
    if 'support_stats' in row:
        stats = row['support_stats']
    elif 'supporting_evidence' in row or 'context_doc_ids' in row:
        stats = rules.support_stats(row)
    elif 'supporting_evidence_count' in row:
        count = int(row['supporting_evidence_count'])
        stats = {'support_doc_count': count, 'context_doc_count': count,
                 'support_chunk_count': 0, 'evidence_item_count': count,
                 'num_hops': int(row.get('num_hops') or max(1, count))}
    else:
        raise ValueError('Offline annotation requires explicit evidence metadata')
    return rules.classify_item(row, answers, stats)


def annotate(row):
    label, reasons = classify(row)
    if label == 'invalid':
        raise ValueError('Cannot annotate an empty question or reference answer')
    return {**row, 'question_type': label, 'label_policy': POLICY,
            'label_policy_sha256': policy_sha256(), 'label_reason': reasons}


def verify_labels(rows):
    digest = policy_sha256()
    for row in rows:
        if row.get('label_policy') != POLICY or row.get('label_policy_sha256') != digest:
            raise ValueError('Missing or obsolete label policy; audit and version the dataset explicitly')
        if classify(row)[0] != row.get('question_type'):
            raise ValueError('Stored type disagrees with the shared annotation policy')
    return {'name': POLICY, 'sha256': digest,
            'inputs': ['question', 'reference_answer', 'support_structure'],
            'source_dataset_used': False, 'human_annotations': False}
