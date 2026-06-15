"""Shared metric utilities used by every benchmark in this package."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from src.apex_rag.pipeline import PipelineResult, run_pipeline


# ---------------------------------------------------------------------------
# Text normalisation (standard QA evaluation protocol)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    """Return normalized tokens for a string."""
    return normalize_answer(s).split()


# ---------------------------------------------------------------------------
# Answer-level metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 when normalized strings are identical."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and a single gold answer."""
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(gold)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, golds: list[str]) -> float:
    """Maximum exact match over a list of acceptable gold answers."""
    return max((exact_match(prediction, g) for g in golds), default=0.0)


def best_token_f1(prediction: str, golds: list[str]) -> float:
    """Maximum token F1 over a list of acceptable gold answers."""
    return max((token_f1(prediction, g) for g in golds), default=0.0)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Mean reciprocal rank at k."""
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Fraction of relevant documents found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def compute_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(top_k)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def safe_run_pipeline(
    query: str,
    items: list[dict[str, str]],
    query_id: str = "q",
    **kwargs: Any,
) -> tuple[PipelineResult | None, str]:
    """Run pipeline, catching all exceptions. Returns (result, error_str)."""
    try:
        return run_pipeline(query, items, query_id=query_id, **kwargs), ""
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__ + ": " + str(exc)


def retrieved_ids(result: PipelineResult) -> list[str]:
    """Return source IDs from the pipeline result's evidence bundle in order."""
    return [item.citation.source_id for item in result.bundle.items]


def avg(values: list[float]) -> float:
    """Safe mean — returns 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def make_item(
    content: str,
    source_id: str,
    title: str = "",
    url: str = "",
    expert: str = "search",
    query: str = "",
) -> dict[str, str]:
    """Build a raw evidence dict in the format expected by run_pipeline()."""
    return {
        "content": content,
        "source_id": source_id,
        "title": title or source_id,
        "url": url,
        "retrieval_expert": expert,
        "retrieval_query": query,
    }
