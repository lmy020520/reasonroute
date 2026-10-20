"""Strict local data contracts and reproducibility identifiers."""
import hashlib
import json
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    keys = [normalize(row.get('question')) for row in rows]
    if not keys or any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError(f'Empty or duplicate normalized questions: {path.name}')
    return rows


def normalize(question):
    return ' '.join(str(question or '').strip().lower().split())


def disjoint(*groups):
    seen = set()
    for rows in groups:
        keys = {normalize(row['question']) for row in rows}
        if seen & keys:
            raise ValueError('Normalized question overlap between data partitions')
        seen.update(keys)


def question_fingerprints(rows):
    return sorted(hashlib.sha256(normalize(row['question']).encode()).hexdigest() for row in rows)


def verify_model_data(artifact, rows, qa_sha256, kind):
    if artifact.get('protocol_version') != 'independent-v4':
        raise ValueError('Legacy model protocol is not valid for independent predicted routing')
    fingerprints = artifact.get('training_question_sha256')
    if not fingerprints or len(fingerprints) != artifact.get('train_n'):
        raise ValueError('Model is missing complete training-question fingerprints')
    if set(fingerprints) & set(question_fingerprints(rows)):
        raise ValueError('Evaluation question overlaps model training data')
    if kind == 'query_classifier':
        if artifact.get('test_sha256') != qa_sha256:
            raise ValueError('Classifier was frozen against a different evaluation dataset version')
    elif qa_sha256 not in artifact.get('excluded_evaluation_sha256', []):
        raise ValueError('Gate training did not explicitly exclude this evaluation dataset version')


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def answer_text(row):
    value = row.get('answers') or row.get('answer') or row.get('ground_truth')
    if isinstance(value, list):
        return '; '.join(str(x) for x in value if x)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Missing reference answer: {row.get("id", "unknown")}')
    return value.strip()


def load_artifact(path, kind):
    import joblib
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get('kind') != kind:
        raise ValueError(f'Expected a {kind} artifact from train_method_models')
    return artifact
