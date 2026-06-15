"""HotpotQA benchmark runner for APEX-RAG.

Evaluates multi-hop question answering using the distractor or fullwiki setting.
Reports Answer EM/F1, Supporting-Fact F1 (document-level), and Joint F1.

Usage:
    from benchmarks.hotpotqa import run, print_results
    result = run(max_examples=200)
    print_results(result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks._utils import (
    avg,
    best_exact_match,
    best_token_f1,
    make_item,
    retrieved_ids,
    safe_run_pipeline,
)


@dataclass
class HotpotQAResult:
    """Aggregated evaluation scores for a HotpotQA run."""

    benchmark: str = "HotpotQA"
    config: str = "distractor"
    split: str = "validation"
    num_examples: int = 0
    answer_em: float = 0.0
    answer_f1: float = 0.0
    sf_f1: float = 0.0
    joint_f1: float = 0.0
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    by_level: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: int = 0
    elapsed_seconds: float = 0.0


def _context_to_items(context: dict, query: str) -> list[dict[str, str]]:
    """Convert HotpotQA context paragraphs to pipeline evidence items."""
    items = []
    for title, sentences in zip(context["title"], context["sentences"], strict=False):
        content = " ".join(sentences).strip()
        if content:
            items.append(
                make_item(
                    content=content,
                    source_id=title,
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    expert="search",
                    query=query,
                )
            )
    return items


def _sf_f1(gold_titles: set[str], bundle_ids: list[str]) -> float:
    """Document-level supporting-fact F1 between gold titles and retrieved IDs."""
    predicted = set(bundle_ids)
    tp = len(predicted & gold_titles)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold_titles) if gold_titles else 1.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run(
    max_examples: int = 200,
    split: str = "validation",
    config: str = "distractor",
) -> HotpotQAResult:
    """Load HotpotQA and run APEX-RAG on up to max_examples questions.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Install the 'datasets' library: pip install datasets") from exc

    ds = load_dataset("hotpotqa/hotpot_qa", config, split=split)
    examples = list(ds.select(range(min(max_examples, len(ds)))))

    answer_ems: list[float] = []
    answer_f1s: list[float] = []
    sf_f1s: list[float] = []
    joint_f1s: list[float] = []
    failures = 0

    # Per-type and per-level buckets: {key: {answer_f1: [], sf_f1: [], joint_f1: []}}
    type_buckets: dict[str, dict[str, list[float]]] = {}
    level_buckets: dict[str, dict[str, list[float]]] = {}

    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        query = ex["question"]
        gold_answer = ex["answer"]
        q_type = ex.get("type", "unknown")
        q_level = ex.get("level", "unknown")
        gold_sf_titles: set[str] = set(ex["supporting_facts"]["title"])

        items = _context_to_items(ex["context"], query)
        result, err = safe_run_pipeline(query, items, query_id=f"hotpot-{i}")

        if result is None:
            failures += 1
            continue

        prediction = result.answer.text
        a_em = best_exact_match(prediction, [gold_answer])
        a_f1 = best_token_f1(prediction, [gold_answer])
        b_ids = retrieved_ids(result)
        s_f1 = _sf_f1(gold_sf_titles, b_ids)
        j_f1 = a_f1 * s_f1

        answer_ems.append(a_em)
        answer_f1s.append(a_f1)
        sf_f1s.append(s_f1)
        joint_f1s.append(j_f1)

        for bucket, key in [(type_buckets, q_type), (level_buckets, q_level)]:
            if key not in bucket:
                bucket[key] = {"answer_f1": [], "sf_f1": [], "joint_f1": []}
            bucket[key]["answer_f1"].append(a_f1)
            bucket[key]["sf_f1"].append(s_f1)
            bucket[key]["joint_f1"].append(j_f1)

    by_type = {k: {m: avg(v) for m, v in vs.items()} for k, vs in type_buckets.items()}
    by_level = {k: {m: avg(v) for m, v in vs.items()} for k, vs in level_buckets.items()}

    return HotpotQAResult(
        config=config,
        split=split,
        num_examples=len(examples),
        answer_em=avg(answer_ems),
        answer_f1=avg(answer_f1s),
        sf_f1=avg(sf_f1s),
        joint_f1=avg(joint_f1s),
        by_type=by_type,
        by_level=by_level,
        failures=failures,
        elapsed_seconds=time.perf_counter() - t0,
    )


def print_results(r: HotpotQAResult) -> None:
    """Print HotpotQA results in a readable table."""
    print(f"\n{'=' * 55}")
    print(f"  HotpotQA [{r.config}] — {r.split} split")
    print(f"{'=' * 55}")
    print(f"  Examples evaluated : {r.num_examples - r.failures}/{r.num_examples}")
    print(f"  Failures           : {r.failures}")
    print(f"  Answer EM          : {r.answer_em:.3f}")
    print(f"  Answer F1          : {r.answer_f1:.3f}")
    print(f"  Supporting-Fact F1 : {r.sf_f1:.3f}")
    print(f"  Joint F1           : {r.joint_f1:.3f}")
    if r.by_type:
        print("\n  By question type:")
        for qtype, scores in r.by_type.items():
            print(
                f"    {qtype:12s}  answer_f1={scores['answer_f1']:.3f}  "
                f"sf_f1={scores['sf_f1']:.3f}  joint_f1={scores['joint_f1']:.3f}"
            )
    if r.by_level:
        print("\n  By difficulty level:")
        for lvl, scores in r.by_level.items():
            print(
                f"    {lvl:8s}  answer_f1={scores['answer_f1']:.3f}  "
                f"joint_f1={scores['joint_f1']:.3f}"
            )
    print(f"\n  Elapsed: {r.elapsed_seconds:.1f}s")
    print(f"{'=' * 55}\n")
