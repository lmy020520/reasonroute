"""Audit labels without changing them, or explicitly create a new annotated version."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonroute.annotation import annotate, classify, POLICY, policy_sha256
from reasonroute.datasets import digest_file, read_rows, write_new_json


def audit_labels(rows):
    changes, invalid = [], []
    for row in rows:
        label, reasons = classify(row)
        if label == 'invalid':
            invalid.append(row.get('id'))
        elif row.get('question_type') != label:
            changes.append({'id': row.get('id'), 'old_type': row.get('question_type'),
                            'proposed_type': label, 'reasons': reasons})
    return {'N': len(rows), 'policy': POLICY, 'policy_sha256': policy_sha256(),
            'changed_labels': len(changes), 'changes': changes, 'invalid_ids': invalid,
            'stored_type_counts': dict(Counter(row.get('question_type') for row in rows)),
            'scope': 'offline deterministic labels; not human annotation or classifier predictions'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['audit', 'annotate'])
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New output directory')
    parser.add_argument('--accept-label-changes', action='store_true',
                        help='Explicitly approve a new dataset version; never modifies the original')
    args = parser.parse_args()
    rows = read_rows(args.input)
    report = {**audit_labels(rows), 'input_sha256': digest_file(args.input)}
    if args.task == 'annotate':
        if report['invalid_ids']:
            raise ValueError('Repair invalid references from source before creating an annotated version')
        if report['changed_labels'] and not args.accept_label_changes:
            raise ValueError('Labels would change. Run audit first; explicit approval is required for a new version')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(args.output / 'audit.json', report)
    if args.task == 'annotate':
        path = args.output / 'questions.jsonl'
        with path.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(annotate(row), ensure_ascii=False) + '\n')
        write_new_json(args.output / 'manifest.json', {
            'label_policy': POLICY, 'label_policy_sha256': policy_sha256(),
            'input_sha256': report['input_sha256'], 'output_sha256': digest_file(path),
            'N': len(rows), 'labels_changed': report['changed_labels'],
            'historical_results_apply': False, 'original_preserved': True})
    print(json.dumps({k: v for k, v in report.items() if k != 'changes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
