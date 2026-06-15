"""Real unit tests for US-011 APEX-Eval Framework implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.apex_eval import (
    ClaimMetrics,
    EvidenceMetrics,
    FinalMetrics,
    QueryEvalResult,
    ReasoningMetrics,
    RetrievalMetrics,
    build_aggregate_report,
    compute_claim_metrics,
    compute_evidence_metrics,
    compute_final_metrics,
    compute_mrr,
    compute_ndcg,
    compute_precision_at_k,
    compute_reasoning_metrics,
    compute_recall_at_k,
    compute_recovery_metrics,
    compute_retrieval_metrics,
    format_report,
)
from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    EvidenceBundle,
    EvidenceItem,
)
from src.apex_rag.generation import generate_answer
from src.apex_rag.retrieval_repair import (
    FailureClass,
    RecoveryStrategy,
    RepairAttempt,
    RepairResult,
)


def _citation(source_id: str = "src-1") -> CitationMetadata:
    """Build a minimal CitationMetadata stub."""
    return CitationMetadata(
        source_id=source_id,
        title="Doc",
        url="https://example.com",
        retrieval_expert="policy",
        retrieval_query="q",
    )


def _item(content: str, source_id: str = "src-1") -> EvidenceItem:
    """Build a single EvidenceItem with given content and source ID."""
    return EvidenceItem(content=content, citation=_citation(source_id), claim_ids=())


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    """Wrap one or more EvidenceItems into an EvidenceBundle."""
    return EvidenceBundle(items=list(items), query_id="q1")


def _retrieval_metrics(**kwargs: float) -> RetrievalMetrics:
    """Build a RetrievalMetrics stub with all scores defaulting to 0.5."""
    defaults = dict(recall_at_k=0.5, precision_at_k=0.5, mrr=0.5, ndcg=0.5, k=5)
    defaults.update(kwargs)
    return RetrievalMetrics(**defaults)  # type: ignore[arg-type]


def _evidence_metrics(**kwargs: float) -> EvidenceMetrics:
    """Build an EvidenceMetrics stub with coverage, precision, and recall defaulting to 0.5."""
    defaults = dict(coverage=0.5, precision=0.5, recall=0.5)
    defaults.update(kwargs)
    return EvidenceMetrics(**defaults)  # type: ignore[arg-type]


def _claim_metrics(**kwargs: float) -> ClaimMetrics:
    """Build a ClaimMetrics stub with full support rate and 2 total claims by default."""
    defaults = dict(support_rate=1.0, unsupported_rate=0.0, total_claims=2)
    defaults.update(kwargs)
    return ClaimMetrics(**defaults)  # type: ignore[arg-type]


def _reasoning_metrics(**kwargs: float) -> ReasoningMetrics:
    """Build a ReasoningMetrics stub with consistency and completeness at 0.8 by default."""
    defaults = dict(logical_consistency=0.8, claim_completeness=0.8)
    defaults.update(kwargs)
    return ReasoningMetrics(**defaults)  # type: ignore[arg-type]


def _final_metrics(**kwargs: float) -> FinalMetrics:
    """Build a FinalMetrics stub with all four quality scores at 0.8 by default."""
    defaults = dict(faithfulness=0.8, groundedness=0.8, relevance=0.8, answer_quality=0.8)
    defaults.update(kwargs)
    return FinalMetrics(**defaults)  # type: ignore[arg-type]


def _query_result(query_id: str = "q1") -> QueryEvalResult:
    """Build a complete QueryEvalResult stub with default metric values."""
    return QueryEvalResult(
        query_id=query_id,
        query_text="What is idempotency?",
        retrieval=_retrieval_metrics(),
        evidence=_evidence_metrics(),
        claims=_claim_metrics(),
        recovery=None,
        reasoning=_reasoning_metrics(),
        final=_final_metrics(),
    )


class TestRetrievalMetrics(unittest.TestCase):
    """Tests for individual retrieval metric functions: recall, precision, MRR, nDCG.

    US-011 acceptance criteria require each metric to be independently computable
    from a set of relevant IDs and an ordered retrieved list, and for a convenience
    wrapper (compute_retrieval_metrics) to populate all four fields at once.
    """

    def test_recall_at_k_perfect_retrieval(self) -> None:
        """When all relevant items are in the top-k results, recall should be 1.0."""
        relevant = {"a", "b", "c"}
        retrieved = ["a", "b", "c", "d", "e"]
        self.assertAlmostEqual(compute_recall_at_k(relevant, retrieved, k=5), 1.0)

    def test_recall_at_k_zero_when_no_relevant_in_top_k(self) -> None:
        """When none of the relevant items appear in the retrieved list, recall should be 0.0."""
        relevant = {"x", "y"}
        retrieved = ["a", "b", "c"]
        self.assertAlmostEqual(compute_recall_at_k(relevant, retrieved, k=3), 0.0)

    def test_recall_at_k_partial(self) -> None:
        """When exactly one of three relevant items is in top-3, recall should be 1/3."""
        relevant = {"a", "b", "c"}
        retrieved = ["a", "d", "e"]
        self.assertAlmostEqual(compute_recall_at_k(relevant, retrieved, k=3), 1 / 3)

    def test_recall_at_k_empty_relevant_returns_zero(self) -> None:
        """An empty relevant set must return recall=0.0 rather than raise."""
        self.assertEqual(compute_recall_at_k(set(), ["a", "b"], k=2), 0.0)

    def test_precision_at_k_all_relevant(self) -> None:
        """When every retrieved item is relevant, precision should be 1.0."""
        relevant = {"a", "b", "c"}
        retrieved = ["a", "b", "c"]
        self.assertAlmostEqual(compute_precision_at_k(relevant, retrieved, k=3), 1.0)

    def test_precision_at_k_none_relevant(self) -> None:
        """When no retrieved item is relevant, precision should be 0.0."""
        relevant = {"x"}
        retrieved = ["a", "b", "c"]
        self.assertAlmostEqual(compute_precision_at_k(relevant, retrieved, k=3), 0.0)

    def test_mrr_first_result_relevant(self) -> None:
        """When the first retrieved item is relevant, MRR should be 1.0."""
        relevant = {"a"}
        retrieved = ["a", "b", "c"]
        self.assertAlmostEqual(compute_mrr(relevant, retrieved), 1.0)

    def test_mrr_second_result_relevant(self) -> None:
        """When only the second retrieved item is relevant, MRR should be 0.5."""
        relevant = {"b"}
        retrieved = ["a", "b", "c"]
        self.assertAlmostEqual(compute_mrr(relevant, retrieved), 0.5)

    def test_mrr_no_relevant_returns_zero(self) -> None:
        """When no retrieved item is relevant, MRR should be 0.0."""
        self.assertAlmostEqual(compute_mrr({"x"}, ["a", "b"]), 0.0)

    def test_ndcg_perfect_order(self) -> None:
        """When both relevant items appear first in order, nDCG should be 1.0."""
        relevant = {"a", "b"}
        retrieved = ["a", "b", "c"]
        score = compute_ndcg(relevant, retrieved, k=3)
        self.assertAlmostEqual(score, 1.0)

    def test_ndcg_reversed_order_lower_than_perfect(self) -> None:
        """Reversing relevant items to the end of the list must reduce nDCG below the perfect ordering."""
        relevant = {"a", "b"}
        perfect = compute_ndcg(relevant, ["a", "b", "c"], k=3)
        imperfect = compute_ndcg(relevant, ["c", "b", "a"], k=3)
        self.assertGreater(perfect, imperfect)

    def test_compute_retrieval_metrics_returns_all_fields(self) -> None:
        """compute_retrieval_metrics must populate k, recall_at_k, and ndcg correctly."""
        result = compute_retrieval_metrics({"a", "b"}, ["a", "b", "c"], k=3)
        self.assertEqual(result.k, 3)
        self.assertGreater(result.recall_at_k, 0)
        self.assertGreater(result.ndcg, 0)


class TestEvidenceMetrics(unittest.TestCase):
    """Tests for compute_evidence_metrics() — measures how well the bundle covers relevant sources.

    US-011 requires coverage, precision, and recall all in [0.0, 1.0]. An empty bundle
    must return all zeros; a perfect match must return precision=recall=1.0.
    """

    def test_empty_bundle_returns_zeros(self) -> None:
        """An empty bundle must yield coverage=precision=recall=0.0."""
        result = compute_evidence_metrics(_bundle(), {"src-1"})
        self.assertEqual(result.coverage, 0.0)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)

    def test_perfect_evidence_match(self) -> None:
        """When all retrieved sources exactly match relevant IDs, precision and recall should be 1.0."""
        bundle = _bundle(_item("text", "src-1"), _item("text2", "src-2"))
        result = compute_evidence_metrics(bundle, {"src-1", "src-2"})
        self.assertAlmostEqual(result.precision, 1.0)
        self.assertAlmostEqual(result.recall, 1.0)

    def test_partial_coverage(self) -> None:
        """When only one of two relevant sources is retrieved, recall should be 0.5."""
        bundle = _bundle(_item("text", "src-1"))
        result = compute_evidence_metrics(bundle, {"src-1", "src-2"})
        self.assertAlmostEqual(result.recall, 0.5)


class TestClaimMetrics(unittest.TestCase):
    """Tests for compute_claim_metrics() — measures claim support rates in a GeneratedAnswer.

    US-011 requires support_rate + unsupported_rate == 1.0 and total_claims to reflect
    the actual number of approved claims in the answer.
    """

    def test_all_claims_supported(self) -> None:
        """An answer where every claim is SUPPORTED must yield support_rate=1.0."""
        bundle = _bundle(_item("Python is widely used."))
        answer = generate_answer(["Python is widely used"], bundle)
        result = compute_claim_metrics(answer)
        self.assertAlmostEqual(result.support_rate, 1.0)
        self.assertAlmostEqual(result.unsupported_rate, 0.0)

    def test_claim_support_plus_unsupported_equals_one(self) -> None:
        """support_rate and unsupported_rate must always sum to exactly 1.0."""
        bundle = _bundle(_item("Python is widely used."))
        answer = generate_answer(["Python is widely used"], bundle)
        result = compute_claim_metrics(answer)
        self.assertAlmostEqual(result.support_rate + result.unsupported_rate, 1.0)

    def test_zero_claims_returns_full_support(self) -> None:
        """A ClaimMetrics with zero total_claims may be constructed with full support."""
        result = ClaimMetrics(support_rate=1.0, unsupported_rate=0.0, total_claims=0)
        self.assertEqual(result.total_claims, 0)
        self.assertEqual(result.support_rate, 1.0)


class TestRecoveryMetrics(unittest.TestCase):
    """Tests for compute_recovery_metrics() — measures the repair loop's success rate.

    US-011 requires compute_recovery_metrics(None) to return None (no repair happened),
    a failed repair to yield success_rate=0.0, and a successful repair to yield >0.
    """

    def test_none_repair_result_returns_none(self) -> None:
        """Passing None for repair result must return None (no repair was triggered)."""
        result = compute_recovery_metrics(None)
        self.assertIsNone(result)

    def test_failed_repair_has_zero_success_rate(self) -> None:
        """A RepairResult with succeeded=False must yield success_rate=0.0."""
        repair = RepairResult(
            succeeded=False,
            final_confidence=0.1,
            iterations_used=3,
            attempts=(
                RepairAttempt(
                    iteration=1,
                    failure_class=FailureClass.NO_EVIDENCE,
                    recovery_strategy=RecoveryStrategy.EXPAND_QUERY,
                    confidence_before=0.0,
                    confidence_after=0.1,
                    outcome="improved",
                ),
            ),
            final_bundle=_bundle(),
            failure_reason="Too many failures.",
        )
        result = compute_recovery_metrics(repair)
        assert result is not None
        self.assertEqual(result.success_rate, 0.0)

    def test_successful_repair_has_nonzero_success_rate(self) -> None:
        """A RepairResult with succeeded=True must yield success_rate > 0.0."""
        repair = RepairResult(
            succeeded=True,
            final_confidence=0.8,
            iterations_used=1,
            attempts=(
                RepairAttempt(
                    iteration=1,
                    failure_class=FailureClass.LOW_RELEVANCE,
                    recovery_strategy=RecoveryStrategy.REWRITE_QUERY,
                    confidence_before=0.3,
                    confidence_after=0.8,
                    outcome="improved",
                ),
            ),
            final_bundle=_bundle(_item("Good evidence.")),
            failure_reason="",
        )
        result = compute_recovery_metrics(repair)
        assert result is not None
        self.assertGreater(result.success_rate, 0.0)


class TestReasoningAndFinalMetrics(unittest.TestCase):
    """Tests for compute_reasoning_metrics() and compute_final_metrics() — end-to-end quality scores.

    US-011 requires all reasoning and final metrics to be in the valid [0.0, 1.0] range.
    """

    def test_reasoning_metrics_fully_supported(self) -> None:
        """logical_consistency must be in [0.0, 1.0] for a fully supported answer."""
        bundle = _bundle(_item("Python is used widely."))
        answer = generate_answer(["Python is used widely"], bundle)
        result = compute_reasoning_metrics(answer)
        self.assertGreaterEqual(result.logical_consistency, 0.0)
        self.assertLessEqual(result.logical_consistency, 1.0)

    def test_final_metrics_in_range(self) -> None:
        """faithfulness, groundedness, relevance, and answer_quality must all be in [0.0, 1.0]."""
        bundle = _bundle(_item("Python is used widely."))
        answer = generate_answer(["Python is used widely"], bundle)
        result = compute_final_metrics(answer, bundle, {"src-1"})
        for attr in ("faithfulness", "groundedness", "relevance", "answer_quality"):
            val = getattr(result, attr)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


class TestAggregateReport(unittest.TestCase):
    """Tests for build_aggregate_report() — averages metrics across multiple query results.

    US-011 requires total_queries to reflect the input count, averages to be correct,
    and per_query to contain one entry per QueryEvalResult.
    """

    def test_empty_results_returns_zero_report(self) -> None:
        """An empty input list must produce total_queries=0 and avg_recall_at_k=0.0."""
        report = build_aggregate_report([])
        self.assertEqual(report.total_queries, 0)
        self.assertEqual(report.avg_recall_at_k, 0.0)

    def test_single_query_aggregates_correctly(self) -> None:
        """A single query with recall=0.5 must produce avg_recall_at_k=0.5."""
        result = _query_result("q1")
        report = build_aggregate_report([result])
        self.assertEqual(report.total_queries, 1)
        self.assertAlmostEqual(report.avg_recall_at_k, 0.5)

    def test_multiple_queries_averaged(self) -> None:
        """Two queries with recall 0.5 and 1.0 must produce avg_recall_at_k=0.75."""
        r1 = _query_result("q1")
        r2 = QueryEvalResult(
            query_id="q2",
            query_text="What is Docker?",
            retrieval=_retrieval_metrics(recall_at_k=1.0),
            evidence=_evidence_metrics(),
            claims=_claim_metrics(),
            recovery=None,
            reasoning=_reasoning_metrics(),
            final=_final_metrics(),
        )
        report = build_aggregate_report([r1, r2])
        self.assertAlmostEqual(report.avg_recall_at_k, 0.75)

    def test_report_includes_per_query_results(self) -> None:
        """The per_query field must contain exactly one entry for a single-query report."""
        result = _query_result("q1")
        report = build_aggregate_report([result])
        self.assertEqual(len(report.per_query), 1)

    def test_unsupported_rate_visible_in_report(self) -> None:
        """avg_unsupported_rate must be ≥ 0.0 and accessible on the aggregate report."""
        result = _query_result("q1")
        report = build_aggregate_report([result])
        self.assertGreaterEqual(report.avg_unsupported_rate, 0.0)


class TestFormatReport(unittest.TestCase):
    """Tests for format_report() — serialises an AggregateReport to a human-readable string.

    US-011 requires the report to contain named sections for Retrieval, Evidence, Claims,
    and Recovery, numeric values for all aggregated metrics, and overall be a str instance.
    """

    def test_format_report_contains_key_sections(self) -> None:
        """The formatted report must contain section headings: APEX-Eval, Retrieval, Evidence, Claims, Recovery."""
        report = build_aggregate_report([_query_result("q1")])
        text = format_report(report)
        self.assertIn("APEX-Eval Report", text)
        self.assertIn("Retrieval", text)
        self.assertIn("Evidence", text)
        self.assertIn("Claims", text)
        self.assertIn("Recovery", text)

    def test_format_report_contains_numeric_values(self) -> None:
        """The formatted report must contain at least one decimal number (e.g. '0.')."""
        report = build_aggregate_report([_query_result("q1")])
        text = format_report(report)
        self.assertIn("0.", text)

    def test_format_report_is_string(self) -> None:
        """format_report must always return a str instance."""
        report = build_aggregate_report([_query_result("q1")])
        self.assertIsInstance(format_report(report), str)


if __name__ == "__main__":
    unittest.main()
