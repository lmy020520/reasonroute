"""Chunk a complete document JSONL snapshot without reading QA or support labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonroute.datasets import digest_file, write_new_json


FORBIDDEN = {'question', 'questions', 'answer', 'answers', 'supporting_facts',
             'supporting_evidence', 'gold_context', 'is_supporting'}


def make_chunks(rows, size=1000, overlap=150):
    if size <= overlap or overlap < 0:
        raise ValueError('Require chunk_chars > overlap_chars >= 0')
    seen = set()
    documents = set()
    for row in rows:
        if FORBIDDEN & row.keys():
            raise ValueError('Input must be complete source documents, not QA or annotated evidence rows')
        text, doc_id = str(row.get('text') or '').strip(), str(row.get('doc_id') or '').strip()
        if not text or not doc_id or doc_id.startswith('iirc_supp__'):
            raise ValueError('Each full document needs text and a stable, non-question-specific doc_id')
        if doc_id in documents:
            raise ValueError('Duplicate source document ID')
        documents.add(doc_id)
        for i, start in enumerate(range(0, len(text), size - overlap)):
            content = text[start:start + size]
            key = hashlib.sha256(content.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            yield {'doc_id': doc_id, 'chunk_id': f'{doc_id}__chunk{i:05d}',
                   'title': row.get('title', ''), 'content': content,
                   'links': row.get('links', []), 'corpus': 'full_document_snapshot',
                   'metadata': {'chunk_index': i, 'char_start': start,
                                'char_count': len(content), 'source_url': row.get('source_url', '')}}
            if start + size >= len(text):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    parser.add_argument('--chunk-chars', type=int, default=1000)
    parser.add_argument('--overlap-chars', type=int, default=150)
    args = parser.parse_args()
    with args.documents.open(encoding='utf-8') as handle:
        chunks = list(make_chunks((json.loads(line) for line in handle if line.strip()),
                                  args.chunk_chars, args.overlap_chars))
    if not chunks:
        raise ValueError('No document chunks')
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'kb_chunks.jsonl'
    with path.open('x', encoding='utf-8') as handle:
        for row in chunks:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_new_json(args.output / 'corpus_manifest.json', {
        'build_policy': 'full_documents_without_qa_v1', 'kb_sha256': digest_file(path),
        'source_sha256': digest_file(args.documents), 'source_name': args.documents.name,
        'chunk_chars': args.chunk_chars, 'overlap_chars': args.overlap_chars,
        'chunks': len(chunks), 'question_conditioned_selection': False,
        'scope': 'Builder does not inspect questions. User must supply an independently obtained complete document snapshot.'})
    print(f'Saved {len(chunks)} fixed-corpus chunks; rebuild embeddings and KG for this snapshot')


if __name__ == '__main__':
    main()
