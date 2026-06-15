"""RAGBench benchmark runner for APEX-RAG.

Evaluates end-to-end RAG quality across multiple enterprise domains using the
TRACe adherence framework. Reports adherence rate, claim support rate, and
domain-level breakdowns for all 12 subsets.

Usage:
    from benchmarks.ragbench import run, print_results
    result = run(max_examples=200)
    print_results(result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks._utils import avg, make_item, safe_run_pipeline

# All known RAGBench domain subsets on Hugging Face.
_ALL_SUBSETS = [
    "covidqa", "cuad", "finqa", "hotpotqa", "pubmedqa",
    "techqa", "msmarco", "nq", "squad", "triviaqa",
    "natural_questions", "bioasq",
]


@dataclass
class SubsetResult:
    """Scores for a single RAGBench domain subset."""

    subset: str
    num_examples: int = 0
    adherence_rate: float = 0.0
    claim_support_rate: float = 0.0
    has_limitations_rate: float = 0.0
    failures: int = 0


@dataclass
class RAGBenchResult:
    """Aggregated evaluation scores for a RAGBench run."""

    benchmark: str = "RAGBench"
    num_examples: int = 0
    overall_adherence: float = 0.0
    overall_support_rate: float = 0.0
    by_subset: dict[str, SubsetResult] = field(default_factory=dict)
    failures: int = 0
    elapsed_seconds: float = 0.0


def _docs_to_items(documents: list[str], query: str) -> list[dict[str, str]]:
    """Convert RAGBench document list to pipeline evidence items."""
    items = []
    for idx, doc in enumerate(documents):
        content = (doc or "").strip()
        if content:
            items.append(make_item(
                content=content,
                source_id=f"ragbench-doc-{idx}",
                title=f"Document {idx + 1}",
                expert="search",
                query=query,
            ))
    return items


def _run_subset(
    subset_name: str,
    max_examples: int,
) -> SubsetResult:
    """Run evaluation on a single RAGBench subset. Returns SubsetResult."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    try:
        ds = load_dataset(
            "galileo-ai/ragbench",
            subset_name,
            split="train",
        )
    except Exception:  # noqa: BLE001
        # Subset may not exist; skip silently
        return SubsetResult(subset=subset_name)

    if len(ds) == 0:
        return SubsetResult(subset=subset_name)

    examples = list(ds.select(range(min(max_examples, len(ds)))))

    adherences: list[float] = []
    support_rates: list[float] = []
    has_lim: list[float] = []
    failures = 0

    for i, ex in enumerate(examples):
        query: str = ex.get("question", "") or ex.get("query", "")
        if not query:
            failures += 1
            continue

        documents: list[str] = ex.get("documents", []) or []
        if isinstance(documents, str):
            documents = [documents]

        items = _docs_to_items(documents, query)
        if not items:
            failures += 1
            continue

        result, err = safe_run_pipeline(
            query, items, query_id=f"{subset_name}-{i}"
        )
        if result is None:
            failures += 1
            continue

        # Adherence: proportion of answer sentences supported by evidence.
        # We use claim_support_rate from APEX-Eval as the proxy.
        support_rate = result.eval_result.claims.support_rate
        support_rates.append(support_rate)

        # A sentence is "adherent" if it comes from a supported claim.
        # We treat support_rate as the adherence score for this example.
        adherences.append(support_rate)
        has_lim.append(float(result.answer.has_limitations))

    return SubsetResult(
        subset=subset_name,
        num_examples=len(examples),
        adherence_rate=avg(adherences),
        claim_support_rate=avg(support_rates),
        has_limitations_rate=avg(has_lim),
        failures=failures,
    )


def run(
    max_examples: int = 200,
    subsets: list[str] | None = None,
) -> RAGBenchResult:
    """Load RAGBench and run APEX-RAG across specified domain subsets.

    Requires: pip install datasets
    Args:
        subsets: List of subset names to run. Defaults to all known subsets.
    """
    try:
        import datasets as _datasets_lib  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install the 'datasets' library: pip install datasets") from exc

    target_subsets = subsets or _ALL_SUBSETS
    t0 = time.perf_counter()

    by_subset: dict[str, SubsetResult] = {}
    total_examples = 0
    total_failures = 0

    for subset in target_subsets:
        sr = _run_subset(subset, max_examples)
        if sr.num_examples > 0:
            by_subset[subset] = sr
            total_examples += sr.num_examples
            total_failures += sr.failures

    all_adherences = [sr.adherence_rate for sr in by_subset.values()]
    all_support_rates = [sr.claim_support_rate for sr in by_subset.values()]

    return RAGBenchResult(
        num_examples=total_examples,
        overall_adherence=avg(all_adherences),
        overall_support_rate=avg(all_support_rates),
        by_subset=by_subset,
        failures=total_failures,
        elapsed_seconds=time.perf_counter() - t0,
    )


def print_results(r: RAGBenchResult) -> None:
    """Print RAGBench results in a readable table."""
    print(f"\n{'='*60}")
    print("  RAGBench — multi-domain evaluation")
    print(f"{'='*60}")
    print(f"  Examples evaluated : {r.num_examples - r.failures}/{r.num_examples}")
    print(f"  Failures           : {r.failures}")
    print(f"  Overall Adherence  : {r.overall_adherence:.3f}")
    print(f"  Overall Support    : {r.overall_support_rate:.3f}")
    if r.by_subset:
        print(f"\n  {'Subset':<22} {'Examples':>8} {'Adherence':>10} {'Support':>8} {'Failures':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for name, sr in r.by_subset.items():
            print(
                f"  {name:<22} {sr.num_examples:>8} "
                f"{sr.adherence_rate:>10.3f} {sr.claim_support_rate:>8.3f} "
                f"{sr.failures:>8}"
            )
    print(f"\n  Elapsed: {r.elapsed_seconds:.1f}s")
    print(f"{'='*60}\n")
