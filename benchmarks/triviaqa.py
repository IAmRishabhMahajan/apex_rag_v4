"""TriviaQA benchmark runner for APEX-RAG.

Evaluates open-domain reading comprehension on 650k+ question-answer-evidence triples.
Reports EM and F1 with alias-based matching and answer normalization.
Supports both the 'rc' (reading comprehension) and 'unfiltered' (open-domain) configs.

Usage:
    from benchmarks.triviaqa import run, print_results
    result = run(max_examples=200)
    print_results(result)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from benchmarks._utils import (
    avg,
    best_exact_match,
    best_token_f1,
    compute_recall_at_k,
    make_item,
    retrieved_ids,
    safe_run_pipeline,
)

# Max tokens taken from each wiki/web passage to keep evidence manageable.
_MAX_PASSAGE_CHARS = 1500


@dataclass
class TriviaQAResult:
    """Aggregated evaluation scores for a TriviaQA run."""

    benchmark: str = "TriviaQA"
    config: str = "rc"
    split: str = "validation"
    num_examples: int = 0
    em: float = 0.0
    f1: float = 0.0
    retrieval_recall: float = 0.0
    failures: int = 0
    elapsed_seconds: float = 0.0


def _get_all_aliases(answer: dict) -> list[str]:
    """Return all acceptable answer forms including normalized aliases."""
    aliases: list[str] = []
    if answer.get("value"):
        aliases.append(answer["value"])
    aliases.extend(answer.get("aliases", []))
    aliases.extend(answer.get("normalized_aliases", []))
    if answer.get("normalized_value"):
        aliases.append(answer["normalized_value"])
    return list(dict.fromkeys(a for a in aliases if a))  # deduplicate, preserve order


def _pages_to_items(pages: dict, query: str) -> list[dict[str, str]]:
    """Convert TriviaQA entity_pages or search_results to evidence items."""
    items = []
    contexts = pages.get("wiki_context") or pages.get("search_context") or []
    titles = pages.get("title") or [""] * len(contexts)
    urls = pages.get("url") or [""] * len(contexts)
    filenames = pages.get("filename") or [""] * len(contexts)

    for ctx, title, url, fname in zip(contexts, titles, urls, filenames):
        content = (ctx or "")[:_MAX_PASSAGE_CHARS].strip()
        if not content:
            continue
        source_id = url or fname or title or f"src-{len(items)}"
        items.append(make_item(
            content=content,
            source_id=str(source_id),
            title=str(title),
            url=str(url),
            expert="search",
            query=query,
        ))
    return items


def run(
    max_examples: int = 200,
    split: str = "validation",
    config: str = "rc",
) -> TriviaQAResult:
    """Load TriviaQA and run APEX-RAG on up to max_examples questions.

    Requires: pip install datasets
    Args:
        config: 'rc' uses provided Wikipedia/web context; 'unfiltered' is open-domain.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Install the 'datasets' library: pip install datasets") from exc

    # streaming=True fetches only as many Parquet shards as needed for max_examples.
    ds = load_dataset("mandarjoshi/trivia_qa", config, split=split, streaming=True)
    examples = list(itertools.islice(ds, max_examples))

    ems: list[float] = []
    f1s: list[float] = []
    retrieval_recalls: list[float] = []
    failures = 0

    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        query: str = ex["question"]
        answer: dict = ex.get("answer", {})
        aliases = _get_all_aliases(answer)

        if not aliases:
            failures += 1
            continue

        # Build evidence items from entity pages (RC) or search results (unfiltered)
        pages = ex.get("entity_pages") or ex.get("search_results") or {}
        items = _pages_to_items(pages, query)

        if not items:
            failures += 1
            continue

        result, err = safe_run_pipeline(query, items, query_id=f"trivia-{i}")
        if result is None:
            failures += 1
            continue

        prediction = result.answer.text
        ems.append(best_exact_match(prediction, aliases))
        f1s.append(best_token_f1(prediction, aliases))

        # Retrieval recall: does the bundle contain docs that hold the answer?
        # Gold = all source IDs that were supplied as evidence (we assume all are relevant)
        gold_ids = {item["source_id"] for item in items}
        b_ids = retrieved_ids(result)
        retrieval_recalls.append(compute_recall_at_k(b_ids, gold_ids, k=5))

    return TriviaQAResult(
        config=config,
        split=split,
        num_examples=len(examples),
        em=avg(ems),
        f1=avg(f1s),
        retrieval_recall=avg(retrieval_recalls),
        failures=failures,
        elapsed_seconds=time.perf_counter() - t0,
    )


def print_results(r: TriviaQAResult) -> None:
    """Print TriviaQA results in a readable table."""
    print(f"\n{'='*55}")
    print(f"  TriviaQA [{r.config}] — {r.split} split")
    print(f"{'='*55}")
    print(f"  Examples evaluated : {r.num_examples - r.failures}/{r.num_examples}")
    print(f"  Failures           : {r.failures}")
    print(f"  EM (alias-matched) : {r.em:.3f}")
    print(f"  F1 (alias-matched) : {r.f1:.3f}")
    print(f"  Retrieval Recall@5 : {r.retrieval_recall:.3f}")
    print(f"\n  Elapsed: {r.elapsed_seconds:.1f}s")
    print(f"{'='*55}\n")
