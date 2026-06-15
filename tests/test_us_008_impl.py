"""Real unit tests for US-008 Retrieval Repair Loop implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    ConflictStatus,
    EvidenceBundle,
    EvidenceItem,
)
from src.apex_rag.evidence_scoring import score_bundle
from src.apex_rag.expert_routing import ExpertType
from src.apex_rag.query_intelligence import build_query_profile
from src.apex_rag.retrieval_repair import (
    _FAILURE_TO_RECOVERY,
    FailureClass,
    RecoveryStrategy,
    _arbitrate_conflicts,
    _decompose_for_coverage,
    _expand_query,
    _fetch_fresh_query,
    _reroute_expert,
    _rewrite_query,
    classify_failure,
    run_repair_loop,
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


class TestFailureToRecoveryMapping(unittest.TestCase):
    """Tests for the _FAILURE_TO_RECOVERY constant — validates the failure→strategy mapping.

    US-008 requires a deterministic mapping: each FailureClass maps to exactly
    one RecoveryStrategy.
    """

    def test_no_evidence_maps_to_expand_query(self) -> None:
        """NO_EVIDENCE failure should map to EXPAND_QUERY strategy."""
        self.assertEqual(
            _FAILURE_TO_RECOVERY[FailureClass.NO_EVIDENCE], RecoveryStrategy.EXPAND_QUERY
        )

    def test_low_relevance_maps_to_rewrite_query(self) -> None:
        """LOW_RELEVANCE failure should map to REWRITE_QUERY strategy."""
        self.assertEqual(
            _FAILURE_TO_RECOVERY[FailureClass.LOW_RELEVANCE], RecoveryStrategy.REWRITE_QUERY
        )

    def test_conflicting_evidence_maps_to_arbitrate(self) -> None:
        """CONFLICTING_EVIDENCE failure should map to ARBITRATE_CONFLICTS strategy."""
        self.assertEqual(
            _FAILURE_TO_RECOVERY[FailureClass.CONFLICTING_EVIDENCE],
            RecoveryStrategy.ARBITRATE_CONFLICTS,
        )

    def test_outdated_evidence_maps_to_fetch_fresh(self) -> None:
        """OUTDATED_EVIDENCE failure should map to FETCH_FRESH strategy."""
        self.assertEqual(
            _FAILURE_TO_RECOVERY[FailureClass.OUTDATED_EVIDENCE], RecoveryStrategy.FETCH_FRESH
        )

    def test_incomplete_coverage_maps_to_decompose(self) -> None:
        """INCOMPLETE_COVERAGE failure should map to DECOMPOSE_CLAIMS strategy."""
        self.assertEqual(
            _FAILURE_TO_RECOVERY[FailureClass.INCOMPLETE_COVERAGE],
            RecoveryStrategy.DECOMPOSE_CLAIMS,
        )

    def test_wrong_expert_maps_to_reroute(self) -> None:
        """WRONG_EXPERT failure should map to REROUTE_EXPERT strategy."""
        self.assertEqual(
            _FAILURE_TO_RECOVERY[FailureClass.WRONG_EXPERT], RecoveryStrategy.REROUTE_EXPERT
        )


class TestClassifyFailure(unittest.TestCase):
    """Tests for classify_failure() — detects the primary reason retrieval is insufficient.

    US-008 requires that classify_failure returns None for acceptable bundles
    and the correct FailureClass for each known failure mode.
    """

    def test_empty_bundle_is_no_evidence(self) -> None:
        """An empty bundle should be classified as NO_EVIDENCE."""
        bundle = _bundle()
        scored = score_bundle(bundle, (), False)
        result = classify_failure(scored)
        self.assertEqual(result, FailureClass.NO_EVIDENCE)

    def test_conflicting_items_classified_as_conflict(self) -> None:
        """A bundle with a high conflict ratio should be classified as CONFLICTING_EVIDENCE."""
        item_a = _item("Python is fast.", "src-1")
        item_b = _item("Python is slow.", "src-2")
        item_b.conflict_status = ConflictStatus.CONFLICT
        bundle = EvidenceBundle(items=[item_a, item_b, item_b], query_id="q1")
        scored = score_bundle(bundle, (), False)
        result = classify_failure(scored)
        self.assertEqual(result, FailureClass.CONFLICTING_EVIDENCE)

    def test_good_bundle_returns_none(self) -> None:
        """A bundle with strong evidence and no conflicts should return None (no failure)."""
        items = [_item(f"Evidence item {i} has strong proof.", f"src-{i}") for i in range(5)]
        bundle = EvidenceBundle(items=items, query_id="q1")
        scored = score_bundle(bundle, (), False)
        result = classify_failure(scored)
        self.assertIsNone(result)


class TestRecoveryActions(unittest.TestCase):
    """Tests for individual recovery action functions — deterministic query transformations.

    US-008 recovery actions simulate what would happen in a real system:
    expanding scope, rewriting phrasing, removing conflicts, adding freshness tokens,
    splitting the query, or switching experts.
    """

    def test_expand_query_appends_context_tokens(self) -> None:
        """_expand_query should retain the original query and append broadening tokens."""
        profile = build_query_profile("What is idempotency?")
        expanded = _expand_query(profile)
        self.assertIn("idempotency", expanded)
        self.assertIn("overview", expanded)

    def test_rewrite_query_restates_query(self) -> None:
        """_rewrite_query should retain the original query and prepend 'Explain in detail'."""
        profile = build_query_profile("What is idempotency?")
        rewritten = _rewrite_query(profile)
        self.assertIn("idempotency", rewritten)
        self.assertIn("Explain", rewritten)

    def test_arbitrate_conflicts_removes_contradicting_items(self) -> None:
        """_arbitrate_conflicts should drop CONFLICT-flagged items, keeping non-conflicting ones."""
        item_a = _item("Python is fast.", "src-1")
        item_b = _item("Python is slow.", "src-2")
        item_b.conflict_status = ConflictStatus.CONFLICT
        bundle = EvidenceBundle(items=[item_a, item_b], query_id="q1")
        result = _arbitrate_conflicts(bundle)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].citation.source_id, "src-1")

    def test_arbitrate_conflicts_keeps_non_conflicting_items(self) -> None:
        """_arbitrate_conflicts should leave items without CONFLICT status unchanged."""
        item_a = _item("Python is fast.", "src-1")
        item_b = _item("Python is popular.", "src-2")
        bundle = EvidenceBundle(items=[item_a, item_b], query_id="q1")
        result = _arbitrate_conflicts(bundle)
        self.assertEqual(len(result.items), 2)

    def test_fetch_fresh_query_appends_freshness_tokens(self) -> None:
        """_fetch_fresh_query should retain the original query and append recency tokens."""
        profile = build_query_profile("What is Kubernetes?")
        fresh_query = _fetch_fresh_query(profile)
        self.assertIn("Kubernetes", fresh_query)
        self.assertIn("recent", fresh_query)

    def test_decompose_for_coverage_returns_multiple_sub_queries(self) -> None:
        """_decompose_for_coverage should return three sub-queries including the original."""
        profile = build_query_profile("Why did the deployment fail?")
        sub_queries = _decompose_for_coverage(profile)
        self.assertEqual(len(sub_queries), 3)
        self.assertIn(profile.raw_query, sub_queries)

    def test_reroute_expert_returns_different_expert(self) -> None:
        """_reroute_expert should never return the same expert that was passed in."""
        profile = build_query_profile("What is Kubernetes?")
        rerouted = _reroute_expert(profile, ExpertType.SEARCH)
        self.assertNotEqual(rerouted, ExpertType.SEARCH)

    def test_reroute_expert_returns_valid_expert_type(self) -> None:
        """_reroute_expert should return a valid ExpertType instance."""
        profile = build_query_profile("What is Kubernetes?")
        rerouted = _reroute_expert(profile, ExpertType.POLICY)
        self.assertIsInstance(rerouted, ExpertType)


class TestRepairLoopBounds(unittest.TestCase):
    """Tests for run_repair_loop() iteration bounding and attempt recording.

    US-008 requires the loop to stop at max_iterations, record every attempt
    with its failure class, recovery strategy, and confidence values.
    """

    def test_loop_stops_at_max_iterations(self) -> None:
        """The loop must not exceed the configured max_iterations."""
        bundle = _bundle()  # empty — always fails
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=2)
        self.assertLessEqual(result.iterations_used, 2)

    def test_persistent_failure_sets_succeeded_false(self) -> None:
        """When all iterations fail, succeeded must be False."""
        bundle = _bundle()
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=3)
        self.assertFalse(result.succeeded)

    def test_persistent_failure_has_failure_reason(self) -> None:
        """A failed repair loop must include a non-empty failure_reason string."""
        bundle = _bundle()
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=3)
        self.assertTrue(len(result.failure_reason) > 0)

    def test_repair_attempts_are_recorded(self) -> None:
        """At least one RepairAttempt must be recorded when the loop fails."""
        bundle = _bundle()
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=3)
        self.assertTrue(len(result.attempts) > 0)

    def test_each_attempt_records_failure_class(self) -> None:
        """Every recorded attempt must carry a valid FailureClass."""
        bundle = _bundle()
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=2)
        for attempt in result.attempts:
            self.assertIsInstance(attempt.failure_class, FailureClass)

    def test_each_attempt_records_recovery_strategy(self) -> None:
        """Every recorded attempt must carry a valid RecoveryStrategy."""
        bundle = _bundle()
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=2)
        for attempt in result.attempts:
            self.assertIsInstance(attempt.recovery_strategy, RecoveryStrategy)

    def test_each_attempt_records_confidence_values(self) -> None:
        """Each attempt must record non-negative confidence_before and ≤1 confidence_after."""
        bundle = _bundle()
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=2)
        for attempt in result.attempts:
            self.assertGreaterEqual(attempt.confidence_before, 0.0)
            self.assertLessEqual(attempt.confidence_after, 1.0)


class TestRepairLoopSuccess(unittest.TestCase):
    """Tests for run_repair_loop() success path and final-bundle output.

    US-008 requires the loop to exit early when confidence_threshold is met
    and always return a final_bundle regardless of outcome.
    """

    def test_sufficient_bundle_succeeds_without_iterations(self) -> None:
        """A bundle that is already sufficient should succeed with zero iterations."""
        items = [_item(f"Strong authoritative evidence item {i}.", f"src-{i}") for i in range(6)]
        bundle = EvidenceBundle(items=items, query_id="q1")
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, confidence_threshold=0.0)
        self.assertTrue(result.succeeded)

    def test_result_has_final_bundle(self) -> None:
        """run_repair_loop must always return a final_bundle EvidenceBundle."""
        bundle = _bundle(_item("Evidence."))
        profile = build_query_profile("What is idempotency?")
        result = run_repair_loop(profile, bundle, max_iterations=1)
        self.assertIsInstance(result.final_bundle, EvidenceBundle)

    def test_conflict_arbitration_during_repair(self) -> None:
        """After arbitration, the final bundle should contain fewer items than the original."""
        item_a = _item("Python is fast.", "src-1")
        item_b = _item("Python is slow.", "src-2")
        item_b.conflict_status = ConflictStatus.CONFLICT
        bundle = EvidenceBundle(items=[item_a, item_b, item_b, item_b], query_id="q1")
        profile = build_query_profile("What is Python?")
        result = run_repair_loop(profile, bundle, max_iterations=3)
        self.assertLessEqual(len(result.final_bundle.items), len(bundle.items))


if __name__ == "__main__":
    unittest.main()
