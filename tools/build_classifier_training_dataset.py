"""Build an independent 1,000-example query-type classifier dataset.

The dataset is collected from public training splits and is kept disjoint from
the current RAG evaluation snapshots.  Labels are assigned from question text
and supporting-evidence structure, not directly from source dataset names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from reasonroute.annotation import annotate, classify, POLICY, policy_sha256
from reasonroute.datasets import read_rows, digest_file
DEFAULT_OUTPUT = PROJECT_ROOT / "data/public_eval/classifier_train_1000"
LABELS = ("fact", "explanation", "reasoning")


def norm_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def support_count(value: Any) -> int:
    if isinstance(value, dict):
        titles = value.get("title")
        if titles is None:
            titles = value.get("titles")
        if titles is None:
            titles = []
        titles = jsonable(titles)
        if isinstance(titles, str):
            titles = [titles]
        return len({norm_question(title) for title in titles if title})
    if isinstance(value, (list, tuple)):
        values = jsonable(value)
        return len({norm_question(item) for item in values if item})
    if hasattr(value, "tolist"):
        return support_count(value.tolist())
    return 0


def add_candidate(
    output: List[dict],
    *,
    source: str,
    item_id: Any,
    question: Any,
    answer: Any = "",
    evidence_count: int = 0,
) -> None:
    question_text = str(question or "").strip()
    if not question_text:
        return
    output.append(
        {
            "id": str(item_id or hashlib.sha1(question_text.encode("utf-8")).hexdigest()[:16]),
            "question": question_text,
            "answer": answer if answer is not None else "",
            "source_dataset": source,
            "source_split": "train",
            "supporting_evidence_count": int(evidence_count),
        }
    )


def load_parquet(path: Path, source: str) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    import pandas as pd

    import pyarrow.parquet as pq
    columns = [name for name in ('id', 'question', 'answer', 'annotations', 'supporting_facts', 'evidences')
               if name in pq.ParquetFile(path).schema_arrow.names]
    frame = pd.read_parquet(path, columns=columns)
    output: List[dict] = []
    for _, row in frame.iterrows():
        record = jsonable(row.to_dict())
        question = record.get("question", "")
        if isinstance(question, dict):
            question = question.get("text", "")
        elif hasattr(question, "item"):
            try:
                question = question.item()
            except Exception:
                pass
        support = record.get("supporting_facts")
        if support is None:
            support = record.get("evidences")
        answer = record.get("answer", "")
        if not answer and isinstance(record.get('annotations'), dict):
            annotations = record['annotations']
            short = [text for group in annotations.get('short_answers', [])
                     for text in group.get('text', []) if text]
            answer = short
            if not answer:
                # NQ's documented enum is -1=NONE, 0=NO, 1=YES.
                answer = ["yes" if v == 1 else "no" for v in annotations.get('yes_no_answer', []) if v in (0, 1)]
        evidence_count = support_count(support)
        if isinstance(record.get('annotations'), dict) and answer:
            evidence_count = 1
        add_candidate(
            output,
            source=source,
            item_id=record.get("id"),
            question=question,
            answer=answer,
            evidence_count=evidence_count,
        )
    return output


def load_iirc(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    output: List[dict] = []
    for article in raw:
        for item in article.get("questions", []):
            answer = item.get("answer", {})
            if isinstance(answer, dict):
                spans = answer.get("answer_spans") or []
                answer = [str(span.get("text", "")) for span in spans if span.get('text')] or answer.get('answer_value') or answer.get('answer') or ''
            add_candidate(
                output,
                source="iirc",
                item_id=item.get("qid"),
                question=item.get("question"),
                answer=answer,
                evidence_count=len({c.get('passage', 'main') for c in item.get('context', []) if c.get('text')}),
            )
    return output


def load_musique(path: Path) -> List[dict]:
    output: List[dict] = []
    for item in read_jsonl(path):
        paragraphs = item.get("paragraphs") or []
        evidence_count = sum(1 for paragraph in paragraphs if paragraph.get("is_supporting"))
        add_candidate(
            output,
            source="musique",
            item_id=item.get("id"),
            question=item.get("question"),
            answer=item.get("answer"),
            evidence_count=evidence_count,
        )
    return output


COMPARISON_RE = re.compile(
    r"\b(older|younger|earlier|later|higher|lower|larger|smaller|more|less|"
    r"same|different|both|either|respectively|before|after|versus|vs\.)\b"
)
NESTED_RE = re.compile(
    r"\b(the (?:person|author|director|film|movie|city|country|team|company|"
    r"school|university|song|book|show|organization)\s+"
    r"(?:who|that|which|where|whose)|"
    r"(?:director|author|spouse|father|mother|capital|birthplace) of)\b"
)
EXPLANATION_RE = re.compile(
    r"^(why|how does|how do|how did|how can|what does|what is the meaning of)\b|"
    r"\b(explain|meaning|cause|purpose|process|outcome|impact|significance|"
    r"effect|role|description|included|consist)\b"
)


def assign_label(item: dict) -> tuple[str, str]:
    label, reasons = classify(item)
    return label, ';'.join(reasons)


def load_exclusion_questions(paths: Sequence[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            question = norm_question(row.get("question"))
            if question:
                excluded.add(question)
    return excluded


def stable_shuffle(items: Iterable[dict], seed: int) -> List[dict]:
    output = list(items)
    random.Random(seed).shuffle(output)
    return output


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a leakage-safe 1,000-example classifier dataset.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument('--raw-root', type=Path, default=PROJECT_ROOT / 'data/public_eval/raw')
    parser.add_argument('--exclude', type=Path, nargs='+', required=True,
                        help='All frozen evaluation sets and any existing Gate development set')
    args = parser.parse_args()

    if args.total != 1000:
        raise ValueError("This reproducibility build is fixed to total=1000.")

    exclusion_paths = args.exclude
    excluded = load_exclusion_questions(exclusion_paths)
    train_sources: List[dict] = []
    train_sources.extend(load_parquet(args.raw_root / 'nq/train-00000-of-00287.parquet', 'naturalquestions'))
    train_sources.extend(load_parquet(args.raw_root / 'hotpot_qa/train-00000-of-00002.parquet', 'hotpotqa'))
    train_sources.extend(load_parquet(args.raw_root / '2wikimultihopqa/train-00000-of-00002.parquet', '2wikimultihopqa'))
    train_sources.extend(load_iirc(args.raw_root / 'iirc/train.json'))
    train_sources.extend(load_musique(args.raw_root / 'musique/musique_ans_v1.0_train.jsonl'))

    unique: Dict[str, dict] = {}
    excluded_count = 0
    for item in train_sources:
        key = norm_question(item["question"])
        if not key:
            continue
        if key in excluded:
            excluded_count += 1
            continue
        unique.setdefault(key, item)

    labeled: List[dict] = []
    for item in unique.values():
        if classify(item)[0] == 'invalid':
            continue
        labeled.append(annotate(item))

    target = {"fact": 334, "explanation": 333, "reasoning": 333}
    selected: List[dict] = []
    available = Counter(row["question_type"] for row in labeled)
    for label in LABELS:
        candidates = stable_shuffle((row for row in labeled if row["question_type"] == label), args.seed)
        if len(candidates) < target[label]:
            raise RuntimeError(f"Not enough {label} candidates: need {target[label]}, found {len(candidates)}")
        selected.extend(candidates[: target[label]])
    selected = stable_shuffle(selected, args.seed)

    # Stratified 80/20 split. The 1,000-row file remains available for auditing.
    train_rows: List[dict] = []
    valid_rows: List[dict] = []
    split_cuts = {label: int(len([row for row in selected if row["question_type"] == label]) * 0.8) for label in LABELS}
    target_train_size = round(len(selected) * 0.8)
    while sum(split_cuts.values()) < target_train_size:
        label = max(
            LABELS,
            key=lambda name: (
                len([row for row in selected if row["question_type"] == name]) * 0.8 - split_cuts[name],
                -LABELS.index(name),
            ),
        )
        split_cuts[label] += 1
    for label in LABELS:
        rows = [row for row in selected if row["question_type"] == label]
        cut = split_cuts[label]
        train_rows.extend(rows[:cut])
        valid_rows.extend(rows[cut:])
    train_rows = stable_shuffle(train_rows, args.seed)
    valid_rows = stable_shuffle(valid_rows, args.seed + 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "classifier_train_1000.jsonl", selected)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)

    manifest = {
        "dataset_name": "independent_query_classifier_train_1000",
        "total": len(selected),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "seed": args.seed,
        "label_counts": dict(Counter(row["question_type"] for row in selected)),
        "train_label_counts": dict(Counter(row["question_type"] for row in train_rows)),
        "valid_label_counts": dict(Counter(row["question_type"] for row in valid_rows)),
        "source_counts": dict(Counter(row["source_dataset"] for row in selected)),
        "candidate_counts_before_sampling": dict(available),
        "public_training_sources": [
            "NaturalQuestions train split",
            "HotpotQA train split",
            "2WikiMultihopQA train split",
            "IIRC train split",
            "MuSiQue-Ans train split",
        ],
        "excluded_evaluation_rows": excluded_count,
        "excluded_question_count": len(excluded),
        "exclusion_sets": [{"name": p.name, "sha256": digest_file(p)} for p in exclusion_paths],
        "label_policy": POLICY,
        "label_policy_sha256": policy_sha256(),
        "annotation_inputs": ['question', 'reference_answer', 'support_structure'],
        "source_dataset_used_for_label": False,
        "leakage_policy": "Normalized question text is unique and disjoint from every explicitly supplied exclusion set.",
        "note": "Labels are deterministic evidence-need annotations and should receive a small manual audit before being cited as human annotations.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved: {output_dir / 'classifier_train_1000.jsonl'}")
    print(f"Saved: {output_dir / 'train.jsonl'}")
    print(f"Saved: {output_dir / 'valid.jsonl'}")
    print(f"Saved: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
