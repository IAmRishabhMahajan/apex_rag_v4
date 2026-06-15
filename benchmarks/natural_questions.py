"""Natural Questions benchmark runner for APEX-RAG.

Evaluates open-domain QA on real Google search queries backed by Wikipedia.
Reports Short-Answer EM/F1 (max over annotators), yes/no accuracy,
and no-answer handling rate.

Usage:
    from benchmarks.natural_questions import run, print_results
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
    make_item,
    safe_run_pipeline,
)

# Maximum non-HTML tokens used as evidence passage per document.
_MAX_PASSAGE_TOKENS = 300


@dataclass
class NQResult:
    """Aggregated evaluation scores for a Natural Questions run."""

    benchmark: str = "Natural Questions"
    split: str = "validation"
    num_examples: int = 0
    short_em: float = 0.0
    short_f1: float = 0.0
    yes_no_accuracy: float = 0.0
    no_answer_handling_rate: float = 0.0
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: int = 0
    elapsed_seconds: float = 0.0


def _extract_passage(doc: dict) -> str:
    """Extract a readable text passage from NQ document token lists."""
    tokens_field = doc.get("tokens", {})
    token_list = tokens_field.get("token", [])
    is_html_list = tokens_field.get("is_html", [])
    text_tokens = [t for t, h in zip(token_list, is_html_list, strict=False) if not h]
    return " ".join(text_tokens[:_MAX_PASSAGE_TOKENS])


def _extract_short_answers(annotations: list[dict] | dict, doc_tokens: list[str]) -> list[str]:
    """Collect short answer texts from annotations in row-list or columnar-dict format."""
    answers: list[str] = []

    if isinstance(annotations, dict):
        # Streaming / Parquet columnar format: each key maps to a per-annotator list.
        # short_answers is a list of dicts: [{'text': [...], 'start_token': [...], ...}, ...]
        for sa_group in annotations.get("short_answers", []):
            if not isinstance(sa_group, dict):
                continue
            texts = sa_group.get("text", []) or []
            for text in texts:
                if text and str(text).strip():
                    answers.append(str(text))
            # Fallback to token spans when text is empty
            starts = sa_group.get("start_token", []) or []
            ends = sa_group.get("end_token", []) or []
            for start, end in zip(starts, ends, strict=False):
                if not any(answers) and start >= 0 and end > start:
                    answers.append(" ".join(doc_tokens[start:end]))
    else:
        # Standard row-list format from non-streaming load.
        for ann in annotations:
            for sa in ann.get("short_answers", []):
                text = sa.get("text", "")
                if text:
                    answers.append(text)
                else:
                    start = sa.get("start_token", -1)
                    end = sa.get("end_token", -1)
                    if start >= 0 and end > start:
                        answers.append(" ".join(doc_tokens[start:end]))

    return [a for a in answers if a.strip()]


def _yes_no_answers(annotations: list[dict] | dict) -> list[str]:
    """Return yes/no answer strings from annotations in row-list or columnar-dict format."""
    results = []

    if isinstance(annotations, dict):
        # Columnar format: yes_no_answer is a list of ints (-1=NONE, 0=NO, 1=YES)
        # or strings ('YES', 'NO', 'NONE').
        for yn in annotations.get("yes_no_answer", []):
            if yn in (1, "YES", "yes"):
                results.append("yes")
            elif yn in (0, "NO", "no"):
                results.append("no")
    else:
        for ann in annotations:
            yn = ann.get("yes_no_answer", "NONE")
            if yn in ("YES", "NO"):
                results.append(yn.lower())

    return results


def run(max_examples: int = 200, split: str = "validation") -> NQResult:
    """Load Natural Questions and run APEX-RAG on up to max_examples questions.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Install the 'datasets' library: pip install datasets") from exc

    # Use streaming=True so only the first max_examples are fetched — avoids
    # downloading all 287 Parquet shards of the full NQ corpus.
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split=split,
        streaming=True,
    )
    examples = list(itertools.islice(ds, max_examples))

    short_ems: list[float] = []
    short_f1s: list[float] = []
    yn_correct = 0
    yn_total = 0
    no_answer_handled = 0
    no_answer_total = 0
    failures = 0

    type_buckets: dict[str, dict[str, list[float]]] = {}

    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        query: str = ex["question"]["text"]
        doc = ex["document"]
        annotations = ex.get("annotations", [])

        # All non-HTML tokens for span extraction fallback
        tokens_field = doc.get("tokens", {})
        all_tokens: list[str] = tokens_field.get("token", [])

        short_answers = _extract_short_answers(annotations, all_tokens)
        yn_answers = _yes_no_answers(annotations)
        has_answer = bool(short_answers or yn_answers)

        if not has_answer:
            no_answer_total += 1

        passage = _extract_passage(doc)
        if not passage.strip():
            failures += 1
            if not has_answer:
                # Still count no-answer examples even without a passage
                pass
            continue

        items = [
            make_item(
                content=passage,
                source_id=doc.get("title", f"nq-{i}"),
                title=doc.get("title", ""),
                url=doc.get("url", ""),
                expert="search",
                query=query,
            )
        ]

        result, err = safe_run_pipeline(query, items, query_id=f"nq-{i}")
        if result is None:
            failures += 1
            continue

        prediction = result.answer.text
        q_type = "no_answer"

        if short_answers:
            q_type = "short"
            em = best_exact_match(prediction, short_answers)
            f1 = best_token_f1(prediction, short_answers)
            short_ems.append(em)
            short_f1s.append(f1)

        elif yn_answers:
            q_type = "yes_no"
            yn_total += 1
            normalized_pred = prediction.strip().lower()
            if any(normalized_pred.startswith(yn) for yn in yn_answers):
                yn_correct += 1

        else:
            # No-answer question: pipeline should surface limitations
            if result.answer.has_limitations:
                no_answer_handled += 1

        if q_type not in type_buckets:
            type_buckets[q_type] = {"em": [], "f1": []}
        if q_type == "short":
            type_buckets[q_type]["em"].append(em)  # type: ignore[possibly-undefined]
            type_buckets[q_type]["f1"].append(f1)  # type: ignore[possibly-undefined]

    by_type = {k: {m: avg(v) for m, v in vs.items()} for k, vs in type_buckets.items()}

    return NQResult(
        split=split,
        num_examples=len(examples),
        short_em=avg(short_ems),
        short_f1=avg(short_f1s),
        yes_no_accuracy=yn_correct / yn_total if yn_total else 0.0,
        no_answer_handling_rate=(no_answer_handled / no_answer_total if no_answer_total else 0.0),
        by_type=by_type,
        failures=failures,
        elapsed_seconds=time.perf_counter() - t0,
    )


def print_results(r: NQResult) -> None:
    """Print Natural Questions results in a readable table."""
    print(f"\n{'=' * 55}")
    print(f"  Natural Questions — {r.split} split")
    print(f"{'=' * 55}")
    print(f"  Examples evaluated : {r.num_examples - r.failures}/{r.num_examples}")
    print(f"  Failures           : {r.failures}")
    print(f"  Short-Answer EM    : {r.short_em:.3f}")
    print(f"  Short-Answer F1    : {r.short_f1:.3f}")
    print(f"  Yes/No accuracy    : {r.yes_no_accuracy:.3f}")
    print(f"  No-answer handling : {r.no_answer_handling_rate:.3f}")
    if r.by_type:
        print("\n  By answer type:")
        for qtype, scores in r.by_type.items():
            parts = "  ".join(f"{m}={v:.3f}" for m, v in scores.items())
            print(f"    {qtype:10s}  {parts}")
    print(f"\n  Elapsed: {r.elapsed_seconds:.1f}s")
    print(f"{'=' * 55}\n")
