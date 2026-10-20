"""Fixed-corpus checks; support documents may exist, QA injections may not."""
import json
from pathlib import Path

from reasonroute.datasets import digest_file


def verify_corpus(kb, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('build_policy') != 'full_documents_without_qa_v1':
        raise ValueError('Use an independently built full-document corpus, not question-specific contexts')
    if manifest.get('kb_sha256') != digest_file(kb):
        raise ValueError('Knowledge base changed after its corpus manifest was frozen')
    count = 0
    with Path(kb).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if any(str(row.get(k, '')).startswith('iirc_supp__') for k in ('doc_id', 'chunk_id')) or row.get('corpus') == 'iirc_context':
                raise ValueError('Legacy question-context supplement detected in the knowledge base')
            count += 1
    if count != manifest.get('chunks') or not count:
        raise ValueError('Corpus manifest chunk count mismatch or empty knowledge base')
    return manifest
