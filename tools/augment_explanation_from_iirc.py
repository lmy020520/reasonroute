"""Select IIRC questions covered by an existing fixed corpus; never append evidence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonroute.annotation import annotate
from reasonroute.corpus import verify_corpus
from reasonroute.datasets import digest_file, normalize, read_rows, write_new_json
from tools.reclassify_public_eval import flatten_answers


def reference_answers(question):
    answer = question.get('answer') or {}
    if not isinstance(answer, dict):
        return flatten_answers(answer)
    spans = [span.get('text', '') for span in answer.get('answer_spans') or []]
    return flatten_answers(spans or answer.get('answer_value') or answer.get('answer'))


def select_questions(passages, chunks, excluded, total):
    by_title = defaultdict(list)
    for chunk in chunks:
        by_title[normalize(chunk.get('title'))].append(chunk)
    output, stats = [], Counter()
    seen = set(excluded)
    for passage in passages:
        for question in passage.get('questions') or []:
            key = normalize(question.get('question'))
            answers = reference_answers(question)
            if not key or key in seen or not answers:
                stats['empty_or_duplicate'] += 1
                continue
            evidence = []
            covered = True
            for context in question.get('context') or []:
                title = passage.get('title') if context.get('passage') == 'main' else context.get('passage')
                text = normalize(context.get('text'))
                matches = [c for c in by_title[normalize(title)] if text and text in normalize(c.get('content'))]
                if not matches:
                    covered = False
                    break
                for chunk in matches:
                    evidence.append({'doc_id': chunk['doc_id'], 'title': chunk.get('title', ''),
                                     'chunk_ids': [chunk['chunk_id']]})
            if not covered or not evidence:
                stats['support_not_in_fixed_corpus'] += 1
                continue
            row = annotate({'id': f"iirc_train__{question['qid']}", 'question': question['question'],
                            'answers': answers, 'dataset': 'iirc', 'split': 'new_evaluation',
                            'supporting_evidence': evidence,
                            'context_doc_ids': sorted({e['doc_id'] for e in evidence}),
                            'provenance': {'source_split': 'train', 'source_sample_id': question['qid'],
                                           'support_mapping': 'exact substring in a pre-existing document chunk'}})
            if row['question_type'] != 'explanation':
                stats['not_explanation_under_shared_policy'] += 1
                continue
            output.append(row)
            seen.add(key)
            if len(output) == total:
                return output, dict(stats)
    return output, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-iirc-train', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--corpus-manifest', required=True, type=Path)
    parser.add_argument('--exclude', required=True, type=Path, nargs='+')
    parser.add_argument('--total', type=int, required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.total <= 0:
        raise ValueError('total must be positive')
    verify_corpus(args.corpus, args.corpus_manifest)
    with args.corpus.open(encoding='utf-8') as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    excluded = {normalize(r['question']) for p in args.exclude for r in read_rows(p)}
    rows, stats = select_questions(json.loads(args.raw_iirc_train.read_text(encoding='utf-8')),
                                   chunks, excluded, args.total)
    if len(rows) < args.total:
        raise ValueError(f'Only {len(rows)} covered explanation questions; need {args.total}. Do not inject missing gold contexts.')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'questions.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_new_json(args.output / 'manifest.json', {'N': len(rows), 'stats': stats,
        'source_sha256': digest_file(args.raw_iirc_train), 'kb_sha256': digest_file(args.corpus),
        'kb_modified': False, 'added_chunks': 0, 'historical_results_apply': False})
    print(f'Saved {len(rows)} questions. Knowledge base unchanged; added chunks: 0')


if __name__ == '__main__':
    main()
