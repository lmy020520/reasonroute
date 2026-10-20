"""
Reclassify public-eval QA samples by question/evidence need, without binding
question_type to source dataset names.

Outputs are written under normalized/reclassified by default:
- qa_dev_reclassified.jsonl
- qa_test_reclassified.jsonl
- qa_dev_reclassified_balanced.jsonl (if enough samples)
- qa_test_reclassified_balanced.jsonl (if enough samples)
- reclassification_report.json

The original qa_dev.jsonl and qa_test.jsonl files are not modified.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any, Dict, Iterable, List, Tuple

QUESTION_TYPES = ("fact", "explanation", "reasoning")

EXPLANATION_PATTERNS = [
    r"\bwhy\b",
    r"\bhow and why\b",
    r"\bhow does\b",
    r"\bhow do\b",
    r"\bhow did\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\bwhat is the difference\b",
    r"\bdifference between\b",
    r"\bmeaning of\b",
    r"\bdefinition of\b",
    r"\bpurpose of\b",
    r"\bgoal of\b",
    r"\bprocess to\b",
    r"\brole of\b",
    r"\bwhat causes\b",
]

COMPARISON_PATTERNS = [
    r"\bolder\b",
    r"\byounger\b",
    r"\bearlier\b",
    r"\blater\b",
    r"\bmore recently\b",
    r"\bhigher\b",
    r"\blower\b",
    r"\blarger\b",
    r"\bsmaller\b",
    r"\blonger\b",
    r"\bshorter\b",
    r"\bsame\b",
    r"\bdifferent\b",
    r"\bboth\b",
    r"\bdo both\b",
    r"\bare both\b",
    r"\bwhich .* (older|younger|earlier|later|higher|lower|larger|smaller|more|less)\b",
]

CHAIN_PATTERNS = [
    r"\bwho .* (that|who|which)\b",
    r"\bwhich .* (that|who|which|whose)\b",
    r"\bwhere .* (that|who|which|whose)\b",
    r"\bwhen .* (that|who|which|whose)\b",
    r"\bthe (father|mother|director|author|performer|manufacturer|headquarters|judge|person|film|song|album|city|country|team|company|school|university) of\b",
    r"\bof the .* (who|that|which|where)\b",
]

DIRECT_FACT_STARTS = (
    "who ",
    "when ",
    "where ",
    "what ",
    "which ",
    "how many ",
    "how much ",
    "in what ",
    "from what ",
)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def flatten_answers(raw: Any) -> List[str]:
    values: List[str] = []

    def visit(x: Any) -> None:
        if isinstance(x, list):
            for item in x:
                visit(item)
            return
        if x is None:
            return
        text = str(x).strip()
        if text:
            values.append(text)

    visit(raw)
    unique: List[str] = []
    seen = set()
    for answer in values:
        key = answer.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(answer)
    return unique


def token_len(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def has_pattern(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def support_stats(item: dict) -> Dict[str, int]:
    ev = item.get("supporting_evidence") or []
    ev_doc_ids = {e.get("doc_id") for e in ev if e.get("doc_id")}
    ctx_doc_ids = {d for d in item.get("context_doc_ids") or [] if d}
    chunk_ids = {
        cid
        for e in ev
        for cid in (e.get("chunk_ids") or [])
        if cid
    }
    try:
        num_hops = int(item.get("num_hops") or 0)
    except (TypeError, ValueError):
        num_hops = 0
    support_doc_count = len(ev_doc_ids or ctx_doc_ids)
    return {
        "support_doc_count": support_doc_count,
        "context_doc_count": len(ctx_doc_ids),
        "support_chunk_count": len(chunk_ids),
        "evidence_item_count": len(ev),
        "num_hops": num_hops or max(1, support_doc_count),
    }


def classify_item(item: dict, answers: List[str], stats: Dict[str, int]) -> Tuple[str, List[str]]:
    question = (item.get("question") or "").strip()
    q = question.casefold()
    if not question:
        return "invalid", ["empty_question"]
    if not answers:
        return "invalid", ["empty_answers"]

    max_answer_tokens = max(token_len(a) for a in answers)
    word_count = token_len(question)
    support_docs = stats["support_doc_count"]
    hops = stats["num_hops"]

    explanation_cue = has_pattern(q, EXPLANATION_PATTERNS)
    comparison_cue = has_pattern(q, COMPARISON_PATTERNS)
    chain_cue = has_pattern(q, CHAIN_PATTERNS) or (
        word_count >= 18 and support_docs >= 2 and q.count(" of ") >= 2
    )
    direct_fact = q.startswith(DIRECT_FACT_STARTS)

    reasons = []
    if explanation_cue:
        reasons.append("explanation_cue")
    if comparison_cue:
        reasons.append("comparison_cue")
    if chain_cue:
        reasons.append("compositional_chain_cue")
    if support_docs >= 2:
        reasons.append("multi_doc_evidence")
    if hops >= 2:
        reasons.append("num_hops_ge_2")
    if max_answer_tokens > 12:
        reasons.append("long_or_descriptive_answer")

    # Reasoning: comparison/composition, or clearly 3+ evidence steps.
    if comparison_cue or hops >= 3 or support_docs >= 3 or chain_cue:
        return "reasoning", reasons or ["reasoning_signal"]

    # Explanation: explicit explanatory intent or descriptive/contextual answer.
    if explanation_cue or max_answer_tokens > 12 or q.startswith("how "):
        return "explanation", reasons or ["explanatory_or_descriptive_question"]

    # Fact: direct lookup intent with concise answer. It may still use public
    # multi-doc evidence, but the query intent itself is fact seeking.
    if direct_fact and max_answer_tokens <= 8:
        return "fact", reasons or ["direct_fact_short_answer"]
    if support_docs <= 1 and max_answer_tokens <= 10:
        return "fact", reasons or ["single_context_short_answer"]

    return "explanation", reasons or ["contextual_description_fallback"]


def reclassify_split(rows: List[dict]) -> Tuple[List[dict], List[dict], dict]:
    from reasonroute.annotation import annotate
    output = []
    invalid = []
    counters = defaultdict(Counter)
    examples = defaultdict(list)

    for row in rows:
        answers = flatten_answers(row.get("answers"))
        stats = support_stats(row)
        new_type, reasons = classify_item(row, answers, stats)
        old_type = row.get("question_type")

        new_row = dict(row)
        new_row["answers"] = answers
        new_row["original_question_type"] = old_type
        new_row["question_type"] = new_type
        new_row["reclassification"] = {
            "method": "dataset_agnostic_rule_v1",
            "reasons": reasons,
            "support_stats": stats,
        }

        counters["new_question_type"][new_type] += 1
        counters["old_question_type"][old_type] += 1
        counters["dataset"][row.get("dataset", "unknown")] += 1
        counters["new_type_by_dataset"][f"{new_type}::{row.get('dataset', 'unknown')}"] += 1
        counters["old_to_new"][f"{old_type}->{new_type}"] += 1

        if new_type == "invalid":
            invalid.append(new_row)
            if len(examples["invalid"]) < 20:
                examples["invalid"].append({
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "old_type": old_type,
                    "question": row.get("question"),
                    "raw_answers": row.get("answers"),
                    "reasons": reasons,
                })
            continue

        output.append(annotate(new_row))
        if old_type != new_type and len(examples["changed"]) < 40:
            examples["changed"].append({
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "old_type": old_type,
                "new_type": new_type,
                "question": row.get("question"),
                "answers": answers[:3],
                "reasons": reasons,
            })

    report = {
        "summary": {k: dict(v) for k, v in counters.items()},
        "valid_count": len(output),
        "invalid_count": len(invalid),
        "examples": examples,
    }
    return output, invalid, report


def round_robin_by_dataset(items: List[dict], limit: int) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        grouped[item.get("dataset", "unknown")].append(item)

    datasets = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    selected: List[dict] = []
    cursor = {name: 0 for name in datasets}
    while len(selected) < limit:
        progressed = False
        for name in datasets:
            idx = cursor[name]
            if idx >= len(grouped[name]):
                continue
            selected.append(grouped[name][idx])
            cursor[name] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def select_balanced(rows: List[dict], per_type: int) -> Tuple[List[dict], dict]:
    buckets: Dict[str, List[dict]] = {q_type: [] for q_type in QUESTION_TYPES}
    for row in rows:
        q_type = row.get("question_type")
        if q_type in buckets:
            buckets[q_type].append(row)
    available = {q_type: len(items) for q_type, items in buckets.items()}
    selected_per_type = min(per_type, *(available.values())) if per_type > 0 else min(available.values())
    selected = []
    selected_by_type_dataset = {}
    for q_type in QUESTION_TYPES:
        chosen = round_robin_by_dataset(buckets[q_type], selected_per_type)
        selected.extend(chosen)
        selected_by_type_dataset[q_type] = dict(Counter(row.get("dataset", "unknown") for row in chosen))
    return selected, {
        "available": available,
        "selected_per_type": selected_per_type,
        "selected_total": len(selected),
        "selected_by_type_dataset": selected_by_type_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--balanced-per-type", type=int, default=500)
    args = parser.parse_args()

    normalized_dir = args.normalized_dir
    output_dir = args.output_dir or normalized_dir / "reclassified"
    output_dir.mkdir(parents=True, exist_ok=False)

    full_report = {
        "method": "dataset_agnostic_rule_v1",
        "note": "Source dataset name is not used to assign question_type.",
        "splits": {},
        "outputs": {},
    }

    for split in ("dev", "test"):
        rows = list(read_jsonl(normalized_dir / f"qa_{split}.jsonl"))
        reclassified, invalid, report = reclassify_split(rows)
        balanced, balanced_report = select_balanced(reclassified, args.balanced_per_type)

        reclassified_path = output_dir / f"qa_{split}_reclassified.jsonl"
        invalid_path = output_dir / f"qa_{split}_invalid.jsonl"
        balanced_path = output_dir / f"qa_{split}_reclassified_balanced{balanced_report['selected_total']}.jsonl"
        write_jsonl(reclassified_path, reclassified)
        write_jsonl(invalid_path, invalid)
        write_jsonl(balanced_path, balanced)

        report["balanced"] = balanced_report
        full_report["splits"][split] = report
        full_report["outputs"][f"{split}_reclassified"] = str(reclassified_path)
        full_report["outputs"][f"{split}_invalid"] = str(invalid_path)
        full_report["outputs"][f"{split}_balanced"] = str(balanced_path)

    report_path = output_dir / "reclassification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    full_report["outputs"]["report"] = str(report_path)

    print(f"Report: {report_path}")
    for split, split_report in full_report["splits"].items():
        print(f"\n[{split}] valid={split_report['valid_count']} invalid={split_report['invalid_count']}")
        print("new_question_type:", split_report["summary"]["new_question_type"])
        print("new_type_by_dataset:", split_report["summary"]["new_type_by_dataset"])
        print("old_to_new:", split_report["summary"]["old_to_new"])
        print("balanced:", split_report["balanced"])


if __name__ == "__main__":
    main()
