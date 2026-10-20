"""Normalize public QA and map support annotations into a frozen document corpus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonroute.annotation import annotate
from reasonroute.corpus import verify_corpus
from reasonroute.datasets import digest_file, normalize, write_new_json
from reasonroute.preprocessing import SupportMapper, adapt_question


def read_source(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            yield from batch.to_pylist()
    elif path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        raw = json.loads(path.read_text(encoding='utf-8'))
        yield from raw if isinstance(raw, list) else raw['data']


def normalize_source(rows, dataset, split, mapper):
    seen = set()
    for parent in rows:
        questions = parent.get('questions', []) if dataset == 'iirc' else [parent]
        for question in questions:
            source_id = question.get('id') or question.get('_id') or question.get('qid')
            try:
                record = adapt_question(question, dataset, parent if dataset == 'iirc' else None)
                key = normalize(record['question'])
                if key in seen:
                    raise ValueError('Duplicate normalized question')
                evidence, missing = mapper.map_all(record['references'])
                if not evidence or (missing and record['reference_policy'] == 'all'):
                    raise ValueError('Incomplete supporting-evidence mapping in the frozen corpus')
                doc_ids = sorted({e['doc_id'] for e in evidence})
                sample_id = record['id'] or hashlib.sha256(key.encode()).hexdigest()[:20]
                output = annotate({'id': f'{dataset}__{split}__{sample_id}',
                    'question': record['question'], 'answers': record['answers'],
                    'dataset': dataset, 'split': split, 'supporting_evidence': evidence,
                    'context_doc_ids': doc_ids, 'num_hops': max(1, len(doc_ids)),
                    'provenance': {'source_dataset': dataset, 'source_split': split,
                                   'source_sample_id': source_id, 'corpus_modified': False,
                                   'alignment_methods': sorted({e['alignment'] for e in evidence})}})
                seen.add(key)
                yield output, None
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                yield None, {'source_id': source_id, 'reason': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['naturalquestions', 'hotpotqa', '2wikimultihopqa', 'musique', 'iirc'], required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--kb', type=Path, required=True)
    parser.add_argument('--corpus-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    verify_corpus(args.kb, args.corpus_manifest)
    with args.kb.open(encoding='utf-8') as handle:
        mapper = SupportMapper(json.loads(line) for line in handle if line.strip())
    args.output.mkdir(parents=True, exist_ok=False)
    accepted = rejected = 0
    counts = Counter()
    with (args.output / 'questions.jsonl').open('x', encoding='utf-8') as good, \
         (args.output / 'rejected.jsonl').open('x', encoding='utf-8') as bad:
        for row, error in normalize_source(read_source(args.qa), args.dataset, args.split, mapper):
            if error:
                rejected += 1
                bad.write(json.dumps(error, ensure_ascii=False) + '\n')
            else:
                accepted += 1
                counts[row['question_type']] += 1
                good.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_new_json(args.output / 'manifest.json', {'dataset': args.dataset, 'split': args.split,
        'accepted': accepted, 'rejected': rejected, 'types': dict(counts),
        'qa_sha256': digest_file(args.qa), 'kb_sha256': digest_file(args.kb),
        'corpus_manifest_sha256': digest_file(args.corpus_manifest), 'kb_modified': False,
        'alignment_policy': 'standard support text/sentence IDs; NQ context or answer position approximation explicitly tagged'})
    print(f'Accepted {accepted}; rejected {rejected}; frozen corpus unchanged')


if __name__ == '__main__':
    main()
