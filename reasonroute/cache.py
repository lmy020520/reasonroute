"""Bind retrieval caches to ordered documents, model identity and cache bytes."""
import hashlib
import json
from pathlib import Path

from reasonroute.datasets import digest_file, write_new_json


def documents_sha256(docs):
    digest = hashlib.sha256(b'reasonroute-ordered-documents-v1\0')
    for doc in docs:
        raw = doc.encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def title_setting(value):
    return '0' if str(value).strip().lower() in ('0', 'false', 'no') else '1'


def write_embedding_metadata(path, docs, model_name, kb_sha256, doc_title_context):
    metadata = {'format': 'reasonroute-embeddings-v2', 'cache_sha256': digest_file(path),
                'kb_sha256': kb_sha256, 'documents_sha256': documents_sha256(docs),
                'embedding_model': model_name, 'rows': len(docs),
                'doc_title_context': title_setting(doc_title_context), 'normalized': True}
    write_new_json(Path(path).with_suffix('.metadata.json'), metadata)
    return metadata


def validate_embedding_metadata(path, docs, model_name, doc_title_context, kb_sha256=None):
    meta_path = Path(path).with_suffix('.metadata.json')
    if not meta_path.is_file():
        raise ValueError('Embedding metadata is required; build a new cache with tools.build_method_index')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    expected = {'format': 'reasonroute-embeddings-v2', 'documents_sha256': documents_sha256(docs),
                'embedding_model': model_name, 'rows': len(docs),
                'doc_title_context': title_setting(doc_title_context), 'normalized': True}
    if kb_sha256 is not None:
        expected['kb_sha256'] = kb_sha256
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('Embedding metadata does not match the ordered corpus, model or title setting; rebuild cache')
    if metadata.get('cache_sha256') != digest_file(path):
        raise ValueError('Embedding cache bytes do not match the frozen metadata; rebuild cache')
    return metadata
