"""Create a corpus-ordered BGE-M3 cache from an existing evidence JSONL."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from reasonroute.datasets import digest_file, write_new_json
from reasonroute.cache import write_embedding_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kb', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.metadata.json').exists():
        raise FileExistsError('Choose a new cache path')
    from evaluate_rag import SiliconFlowEmbedding, SILICONFLOW_API_KEY, load_json_data
    if not SILICONFLOW_API_KEY:
        raise ValueError('Set SILICONFLOW_API_KEY locally')
    docs = load_json_data(str(args.kb))
    if not docs:
        raise ValueError('Empty knowledge base')
    vectors = SiliconFlowEmbedding(api_key=SILICONFLOW_API_KEY).encode(docs, normalize_embeddings=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, embeddings=vectors)
    write_embedding_metadata(args.output, docs, 'BAAI/bge-m3', digest_file(args.kb),
                             __import__('os').getenv('DOC_TITLE_CONTEXT', '0'))


if __name__ == '__main__':
    main()
