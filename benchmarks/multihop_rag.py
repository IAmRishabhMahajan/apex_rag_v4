"""MultiHop-RAG benchmark runner for APEX-RAG.

Evaluates multi-hop retrieval and QA on a dataset purpose-built for RAG systems.
Reports MRR, Recall@K, Precision@K for retrieval and Answer EM/F1 for QA.
Null-query handling rate is measured separately.

Usage:
    from benchmarks.multihop_rag import run, print_results
    result = run(max_examples=200)
    print_results(result)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from benchmarks._utils import (
    avg,
    best_exact_match,
    best_token_f1,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    make_item,
    retrieved_ids,
    safe_run_pipeline,
)

# Null queries have no answer in the corpus — the pipeline should surface limitations.
_NULL_TYPE = "null"


@dataclass
class MultiHopRAGResult:
    """Aggregated evaluation scores for a MultiHop-RAG run."""

    benchmark: str = "MultiHop-RAG"
    num_examples: int = 0
    k: int = 5
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    answer_em: float = 0.0
    answer_f1: float = 0.0
    null_handling_rate: float = 0.0
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: int = 0
    elapsed_seconds: float = 0.0


def _evidence_to_items(evidence_list: list[dict], query: str) -> list[dict[str, str]]:
    """Convert MultiHop-RAG evidence_list entries to pipeline evidence items."""
    items = []
    for ev in evidence_list:
        facts = ev.get("facts", [])
        content = " ".join(facts) if facts else ev.get("title", "")
        source_id = ev.get("source", ev.get("title", f"src-{len(items)}"))
        if content.strip():
            items.append(
                make_item(
                    content=content,
                    source_id=str(source_id),
                    title=ev.get("title", ""),
                    url=str(source_id) if source_id.startswith("http") else "",
                    expert="search",
                    query=query,
                )
            )
    return items


def run(max_examples: int = 200, k: int = 5) -> MultiHopRAGResult:
    """Load MultiHop-RAG and run APEX-RAG on up to max_examples queries.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Install the 'datasets' library: pip install datasets") from exc

    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train", streaming=True)
    examples = list(itertools.islice(ds, max_examples))

    mrrs: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    ems: list[float] = []
    f1s: list[float] = []
    null_correct = 0
    null_total = 0
    failures = 0

    type_buckets: dict[str, dict[str, list[float]]] = {}

    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        query: str = ex["query"]
        gold_answer: str = ex.get("answer", "")
        q_type: str = ex.get("question_type", "unknown")
        evidence_list: list[dict] = ex.get("evidence_list", [])

        # Gold retrieval: all source IDs from evidence_list
        gold_ids: set[str] = {str(ev.get("source", ev.get("title", ""))) for ev in evidence_list}

        items = _evidence_to_items(evidence_list, query)
        result, err = safe_run_pipeline(query, items, query_id=f"multihop-{i}")

        if result is None:
            failures += 1
            continue

        b_ids = retrieved_ids(result)

        # Retrieval metrics
        mrrs.append(compute_mrr(b_ids, gold_ids, k=k))
        recalls.append(compute_recall_at_k(b_ids, gold_ids, k=k))
        precisions.append(compute_precision_at_k(b_ids, gold_ids, k=k))

        # Null-query handling: pipeline should surface has_limitations
        if q_type == _NULL_TYPE:
            null_total += 1
            if result.answer.has_limitations:
                null_correct += 1

        # QA metrics (skip for null queries — there is no gold answer)
        if q_type != _NULL_TYPE and gold_answer:
            prediction = result.answer.text
            em = best_exact_match(prediction, [gold_answer])
            f1 = best_token_f1(prediction, [gold_answer])
            ems.append(em)
            f1s.append(f1)

            # Per-type buckets
            if q_type not in type_buckets:
                type_buckets[q_type] = {"answer_f1": [], "mrr": [], "recall": []}
            type_buckets[q_type]["answer_f1"].append(f1)
            type_buckets[q_type]["mrr"].append(compute_mrr(b_ids, gold_ids, k=k))
            type_buckets[q_type]["recall"].append(compute_recall_at_k(b_ids, gold_ids, k=k))

    by_type = {k: {m: avg(v) for m, v in vs.items()} for k, vs in type_buckets.items()}

    return MultiHopRAGResult(
        num_examples=len(examples),
        k=k,
        mrr=avg(mrrs),
        recall_at_k=avg(recalls),
        precision_at_k=avg(precisions),
        answer_em=avg(ems),
        answer_f1=avg(f1s),
        null_handling_rate=null_correct / null_total if null_total else 0.0,
        by_type=by_type,
        failures=failures,
        elapsed_seconds=time.perf_counter() - t0,
    )


def print_results(r: MultiHopRAGResult) -> None:
    """Print MultiHop-RAG results in a readable table."""
    print(f"\n{'=' * 55}")
    print(f"  MultiHop-RAG (k={r.k})")
    print(f"{'=' * 55}")
    print(f"  Examples evaluated : {r.num_examples - r.failures}/{r.num_examples}")
    print(f"  Failures           : {r.failures}")
    print(f"  MRR@{r.k}            : {r.mrr:.3f}")
    print(f"  Recall@{r.k}         : {r.recall_at_k:.3f}")
    print(f"  Precision@{r.k}      : {r.precision_at_k:.3f}")
    print(f"  Answer EM          : {r.answer_em:.3f}")
    print(f"  Answer F1          : {r.answer_f1:.3f}")
    print(f"  Null handling rate : {r.null_handling_rate:.3f}")
    if r.by_type:
        print("\n  By question type:")
        for qtype, scores in r.by_type.items():
            print(
                f"    {qtype:12s}  answer_f1={scores['answer_f1']:.3f}  "
                f"mrr={scores['mrr']:.3f}  recall={scores['recall']:.3f}"
            )
    print(f"\n  Elapsed: {r.elapsed_seconds:.1f}s")
    print(f"{'=' * 55}\n")
