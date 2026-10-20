"""Build an independent candidate set for CompactGate supervision.

The CompactGate is only applied to reasoning queries.  Its final binary target
must be created later from paired low-cost/high-cost branch outcomes, so this
script deliberately writes candidates with ``gate_label: null`` rather than
inventing outcome labels from source metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_classifier_training_dataset import (
    assign_label,
    load_iirc,
    load_musique,
    load_parquet,
    norm_question,
    read_jsonl,
)
from reasonroute.annotation import annotate, POLICY, policy_sha256
from reasonroute.datasets import read_rows


DEFAULT_OUTPUT = PROJECT_ROOT / "data/public_eval/compact_gate_train_800"


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_questions(paths: Iterable[Path]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            question = norm_question(row.get("question"))
            if question:
                seen.add(question)
    return seen


def stable_shuffle(rows: Sequence[dict], seed: int) -> list[dict]:
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an independent 800-query CompactGate candidate set."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument('--raw-root', type=Path, default=PROJECT_ROOT / 'data/public_eval/raw')
    parser.add_argument('--exclude', type=Path, nargs='+', required=True,
                        help='Independent classifier set and all evaluation sets; missing files are errors')
    args = parser.parse_args()

    if args.total != 800:
        raise ValueError("This reproducibility build is fixed to 800 candidates.")

    exclusion_paths = args.exclude
    excluded_questions = collect_questions(exclusion_paths)
    raw_root = args.raw_root
    source_rows: list[dict] = []
    source_rows.extend(load_parquet(raw_root / "nq/train-00000-of-00287.parquet", "naturalquestions"))
    source_rows.extend(load_parquet(raw_root / "hotpot_qa/train-00000-of-00002.parquet", "hotpotqa"))
    source_rows.extend(load_parquet(raw_root / "2wikimultihopqa/train-00000-of-00002.parquet", "2wikimultihopqa"))
    source_rows.extend(load_iirc(raw_root / "iirc/train.json"))
    source_rows.extend(load_musique(raw_root / "musique/musique_ans_v1.0_train.jsonl"))

    unique: dict[str, dict] = {}
    excluded_count = 0
    for item in source_rows:
        key = norm_question(item.get("question"))
        if not key:
            continue
        if key in excluded_questions:
            excluded_count += 1
            continue
        unique.setdefault(key, item)

    candidates: list[dict] = []
    for item in unique.values():
        if not str(item.get("answer") or "").strip():
            continue
        query_type, label_reason = assign_label(item)
        if query_type != "reasoning":
            continue
        question = item["question"].strip()
        candidates.append(
            {
                **annotate(item),
                "id": f"compact_gate_train__{item['source_dataset']}__{item['id']}",
                "source_id": str(item["id"]),
                "question": question,
                "answer": item.get("answer", ""),
                "source_dataset": item["source_dataset"],
                "source_split": "train",
                "supporting_evidence_count": int(item.get("supporting_evidence_count", 0)),
                "candidate_question_type": "reasoning",
                "candidate_reason": label_reason,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "gate_label": None,
                "gate_label_definition": "pending_paired_low_vs_high_branch_evaluation",
            }
        )

    candidates = stable_shuffle(candidates, args.seed)
    if len(candidates) < args.total:
        raise RuntimeError(
            f"Only {len(candidates)} unique non-overlapping reasoning candidates are available; need {args.total}."
        )
    selected = candidates[: args.total]

    selected_keys = {norm_question(row["question"]) for row in selected}
    overlap = selected_keys & excluded_questions
    if overlap:
        raise RuntimeError(f"Leakage check failed: {len(overlap)} normalized questions overlap excluded sets.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "compact_gate_candidates_800.jsonl"
    if dataset_path.exists() or (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Choose a new candidate directory to preserve previous datasets")
    write_jsonl(dataset_path, selected)
    manifest = {
        "dataset_name": "independent_compact_gate_candidates_800",
        "total": len(selected),
        "seed": args.seed,
        "source_counts": dict(Counter(row["source_dataset"] for row in selected)),
        "source_split": "public_train_only",
        "candidate_policy": (
            "reasoning candidates selected by deterministic evidence-need rules; "
            "these are not CompactGate labels"
        ),
        "label_policy": POLICY,
        "label_policy_sha256": policy_sha256(),
        "gate_label_status": "pending",
        "required_next_step": (
            "Run both frozen low-cost and high-cost reasoning branches on every candidate. "
            "Set gate_label=1 only when high branch correctness is 1 and low branch correctness is 0; "
            "otherwise set gate_label=0."
        ),
        "exclusion_sets": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in exclusion_paths],
        "deduplication": "normalized exact question text: trim, lowercase, collapse whitespace; punctuation retained",
        "excluded_source_rows": excluded_count,
        "excluded_normalized_question_count": len(excluded_questions),
        "verified_overlap_with_exclusion_sets": 0,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved: {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
