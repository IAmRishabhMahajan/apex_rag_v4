"""Integration tests for the full APEX-RAG v5 pipeline (run_pipeline / run_batch)."""

from __future__ import annotations

import unittest

from src.apex_rag.pipeline import PipelineResult, run_batch, run_pipeline


def _evidence(
    content: str,
    source_id: str = "src-1",
    retrieval_expert: str = "search",
) -> dict[str, str]:
    """Build a raw evidence dict in the format expected by run_pipeline."""
    return {
        "content": content,
        "source_id": source_id,
        "title": "Test Doc",
        "url": "https://example.com",
        "retrieval_expert": retrieval_expert,
        "retrieval_query": "test query",
    }


class TestRunPipeline(unittest.TestCase):
    """End-to-end integration tests for run_pipeline() — validates full pipeline output.

    These tests exercise every pipeline stage (query intelligence, retrieval planning,
    evidence fusion, scoring, repair, reasoning, generation, validation, risk verification,
    and APEX-Eval) via the single public entry point, verifying that all PipelineResult
    fields are populated and within valid ranges.
    """

    def _run(self, query: str, *contents: str) -> PipelineResult:
        """Run the pipeline with a query and one evidence dict per content string."""
        items = [_evidence(c, f"src-{i}") for i, c in enumerate(contents, 1)]
        return run_pipeline(query, items, query_id="test-q")

    def test_returns_pipeline_result(self) -> None:
        """run_pipeline must return a PipelineResult instance."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertIsInstance(result, PipelineResult)

    def test_query_profile_matches_input(self) -> None:
        """The query_profile raw_query must match the string passed to run_pipeline."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertEqual(result.query_profile.raw_query, "What is idempotency?")

    def test_bundle_contains_evidence(self) -> None:
        """The fused bundle must contain at least one item when evidence is supplied."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertGreater(len(result.bundle.items), 0)

    def test_answer_text_is_non_empty(self) -> None:
        """The generated answer text must be non-empty for a query with supporting evidence."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertTrue(len(result.answer.text) > 0)

    def test_verified_answer_has_risk_assessment(self) -> None:
        """The verified answer must always carry a risk_assessment object."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertIsNotNone(result.verified.risk_assessment)

    def test_eval_result_has_query_id(self) -> None:
        """The eval_result query_id must match the query_id argument passed to run_pipeline."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertEqual(result.eval_result.query_id, "test-q")

    def test_retrieval_plan_is_set(self) -> None:
        """The retrieval_plan field must be populated (not None) in every run."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertIsNotNone(result.retrieval_plan)

    def test_all_validations_present(self) -> None:
        """All four validation results (query, fusion, claim, generation) must be non-None."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertIsNotNone(result.query_validation)
        self.assertIsNotNone(result.fusion_validation)
        self.assertIsNotNone(result.claim_validation)
        self.assertIsNotNone(result.generation_validation)

    def test_scored_bundle_has_confidence(self) -> None:
        """The scored bundle's composite confidence must be in [0.0, 1.0]."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertGreaterEqual(result.scored.scores.confidence, 0.0)
        self.assertLessEqual(result.scored.scores.confidence, 1.0)

    def test_complex_query_uses_reasoning_path(self) -> None:
        """A multi-hop query containing 'why' and 'after' must activate used_complex_path=True."""
        result = self._run(
            "Why did the deployment fail after the release?",
            "The deployment state described a configuration error event.",
        )
        self.assertTrue(result.reasoning.used_complex_path)

    def test_simple_query_skips_reasoning_path(self) -> None:
        """A simple factual query must have used_complex_path=False."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        self.assertFalse(result.reasoning.used_complex_path)

    def test_repair_triggered_for_empty_evidence(self) -> None:
        """An empty evidence list must trigger the repair loop, setting repair to non-None."""
        result = run_pipeline("What is idempotency?", [], query_id="empty-q")
        self.assertIsNotNone(result.repair)

    def test_repair_not_triggered_for_sufficient_evidence(self) -> None:
        """A bundle that meets the confidence threshold must leave repair=None."""
        items = [_evidence(f"Strong authoritative evidence {i}.", f"src-{i}") for i in range(6)]
        result = run_pipeline(
            "What is idempotency?",
            items,
            query_id="full-q",
            relevant_source_ids={f"src-{i}" for i in range(6)},
        )
        self.assertIsNone(result.repair)

    def test_eval_metrics_in_range(self) -> None:
        """All four primary eval metrics must be in [0.0, 1.0]."""
        result = self._run("What is idempotency?", "Idempotency means same result every time.")
        ev = result.eval_result
        for val in (
            ev.retrieval.recall_at_k,
            ev.evidence.coverage,
            ev.claims.support_rate,
            ev.final.faithfulness,
        ):
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_high_risk_query_gets_disclaimer(self) -> None:
        """A medical query passed with high_risk=True must yield is_high_risk=True on the result."""
        items = [_evidence("Medication dosage depends on patient weight.", "src-1")]
        result = run_pipeline(
            "What medication dose should I take?",
            items,
            high_risk=True,
            candidate_claims=["Medication dosage varies by weight"],
        )
        self.assertTrue(result.verified.risk_assessment.is_high_risk)


class TestRunBatch(unittest.TestCase):
    """Integration tests for run_batch() — aggregates pipeline results over multiple queries.

    US-011 end-to-end: run_batch must produce an AggregateReport with total_queries matching
    the input count, per_query populated, and correct averages for an empty input.
    """

    def test_batch_aggregates_multiple_queries(self) -> None:
        """run_batch over two queries must return total_queries=2."""
        queries = [
            ("What is idempotency?", [_evidence("Idempotency means same result.")]),
            ("What is Docker?", [_evidence("Docker is a container runtime.")]),
        ]
        report = run_batch(queries)
        self.assertEqual(report.total_queries, 2)

    def test_batch_empty_queries_returns_zero_report(self) -> None:
        """run_batch over an empty list must return total_queries=0 and avg_recall_at_k=0.0."""
        report = run_batch([])
        self.assertEqual(report.total_queries, 0)
        self.assertEqual(report.avg_recall_at_k, 0.0)

    def test_batch_report_has_per_query_results(self) -> None:
        """run_batch over one query must populate per_query with exactly one entry."""
        queries = [("What is idempotency?", [_evidence("Idempotency means same result.")])]
        report = run_batch(queries)
        self.assertEqual(len(report.per_query), 1)


if __name__ == "__main__":
    unittest.main()
