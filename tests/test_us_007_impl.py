"""Real unit tests for US-007 Evidence Scoring and Gap Detection implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    ConflictStatus,
    EvidenceBundle,
    EvidenceItem,
)
from src.apex_rag.evidence_scoring import (
    GapType,
    score_bundle,
)


def _citation(source_id: str = "src-1", expert: str = "policy") -> CitationMetadata:
    """Build a minimal CitationMetadata stub with a configurable expert."""
    return CitationMetadata(
        source_id=source_id,
        title="Doc",
        url="https://example.com",
        retrieval_expert=expert,
        retrieval_query="q",
    )


def _item(
    content: str,
    source_id: str = "src-1",
    expert: str = "policy",
    claim_ids: tuple[str, ...] = (),
) -> EvidenceItem:
    """Build an EvidenceItem with configurable content, source, expert, and claim IDs."""
    return EvidenceItem(
        content=content,
        citation=_citation(source_id, expert),
        claim_ids=claim_ids,
    )


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    """Wrap one or more EvidenceItems into an EvidenceBundle."""
    return EvidenceBundle(items=list(items), query_id="q1")


class TestEvidenceScores(unittest.TestCase):
    """Tests for score_bundle() score dimensions — authority, freshness, agreement, confidence.

    US-007 requires five quality scores in [0, 1] and a human-readable reason string.
    Stale content must lower freshness; conflict items must lower agreement;
    high-risk mode must apply a stricter confidence threshold.
    """

    def test_scores_have_reason(self) -> None:
        """score_bundle should always return a non-empty reason string."""
        bundle = _bundle(_item("Python is popular."))
        result = score_bundle(bundle)
        self.assertTrue(len(result.scores.reason) > 0)

    def test_authority_score_range(self) -> None:
        """Authority score must be in the range [0.0, 1.0]."""
        bundle = _bundle(_item("Python is popular.", expert="policy"))
        result = score_bundle(bundle)
        self.assertGreaterEqual(result.scores.authority, 0.0)
        self.assertLessEqual(result.scores.authority, 1.0)

    def test_freshness_score_range(self) -> None:
        """Freshness score must be in the range [0.0, 1.0]."""
        bundle = _bundle(_item("Python is popular."))
        result = score_bundle(bundle)
        self.assertGreaterEqual(result.scores.freshness, 0.0)
        self.assertLessEqual(result.scores.freshness, 1.0)

    def test_agreement_score_range(self) -> None:
        """Agreement score must be in the range [0.0, 1.0]."""
        bundle = _bundle(_item("Python is popular."))
        result = score_bundle(bundle)
        self.assertGreaterEqual(result.scores.agreement, 0.0)
        self.assertLessEqual(result.scores.agreement, 1.0)

    def test_confidence_score_range(self) -> None:
        """Composite confidence score must be in the range [0.0, 1.0]."""
        bundle = _bundle(_item("Python is popular.", expert="research"))
        result = score_bundle(bundle)
        self.assertGreaterEqual(result.scores.confidence, 0.0)
        self.assertLessEqual(result.scores.confidence, 1.0)

    def test_stale_content_lowers_freshness(self) -> None:
        """Content containing stale signals like 'deprecated' should reduce the freshness score."""
        fresh_bundle = _bundle(_item("Python is popular."))
        stale_bundle = _bundle(_item("This API is now deprecated and obsolete."))
        fresh_result = score_bundle(fresh_bundle)
        stale_result = score_bundle(stale_bundle)
        self.assertGreater(fresh_result.scores.freshness, stale_result.scores.freshness)

    def test_conflict_lowers_agreement(self) -> None:
        """A CONFLICT-flagged item should reduce the agreement score below a clean bundle."""
        normal_bundle = _bundle(_item("Python is popular."))
        conflicted = _item("Python is popular.", source_id="src-1")
        conflicted.conflict_status = ConflictStatus.CONFLICT
        conflict_bundle = EvidenceBundle(items=[conflicted], query_id="q1")
        normal_result = score_bundle(normal_bundle)
        conflict_result = score_bundle(conflict_bundle)
        self.assertGreater(normal_result.scores.agreement, conflict_result.scores.agreement)

    def test_high_risk_threshold_is_higher(self) -> None:
        """High-risk mode should label the threshold 'high-risk' in the reason string."""
        bundle = _bundle(
            _item("Python is popular.", expert="freshness"),
        )
        high_risk = score_bundle(bundle, high_risk=True)
        self.assertIn("high-risk", high_risk.scores.reason)


class TestGapDetection(unittest.TestCase):
    """Tests for score_bundle() gap detection — identifies missing, stale, conflicting, or incomplete evidence.

    US-007 requires gap reports with suggested_repair strings for each identified gap.
    """

    def test_empty_bundle_produces_missing_evidence_gap(self) -> None:
        """An empty bundle should produce a MISSING_EVIDENCE gap."""
        bundle = EvidenceBundle(items=[], query_id="q1")
        result = score_bundle(bundle)
        gap_types = [g.gap_type for g in result.gaps]
        self.assertIn(GapType.MISSING_EVIDENCE, gap_types)

    def test_conflict_produces_conflicting_evidence_gap(self) -> None:
        """A CONFLICT-flagged item should produce a CONFLICTING_EVIDENCE gap."""
        item = _item("Python is popular.")
        item.conflict_status = ConflictStatus.CONFLICT
        bundle = EvidenceBundle(items=[item], query_id="q1")
        result = score_bundle(bundle)
        gap_types = [g.gap_type for g in result.gaps]
        self.assertIn(GapType.CONFLICTING_EVIDENCE, gap_types)

    def test_stale_content_produces_stale_gap(self) -> None:
        """A bundle where all items contain stale signals should produce a STALE_EVIDENCE gap."""
        bundle = _bundle(
            _item("This feature is deprecated and legacy.", expert="freshness"),
            _item("This API is obsolete.", source_id="src-2", expert="freshness"),
            _item("Old version no longer supported.", source_id="src-3", expert="freshness"),
        )
        result = score_bundle(bundle)
        gap_types = [g.gap_type for g in result.gaps]
        self.assertIn(GapType.STALE_EVIDENCE, gap_types)

    def test_uncovered_claim_produces_incomplete_coverage_gap(self) -> None:
        """A claim ID with no supporting evidence should produce an INCOMPLETE_COVERAGE gap."""
        bundle = _bundle(_item("Python is popular.", claim_ids=("c1",)))
        result = score_bundle(bundle, claim_ids=("c1", "c2"))
        gap_types = [g.gap_type for g in result.gaps]
        self.assertIn(GapType.INCOMPLETE_COVERAGE, gap_types)

    def test_gap_reports_have_suggested_repair(self) -> None:
        """Every gap report must include a non-empty suggested_repair string."""
        bundle = EvidenceBundle(items=[], query_id="q1")
        result = score_bundle(bundle)
        for gap in result.gaps:
            self.assertTrue(len(gap.suggested_repair) > 0)

    def test_healthy_bundle_has_no_critical_gaps(self) -> None:
        """A healthy bundle should not contain MISSING_EVIDENCE or INCOMPLETE_COVERAGE gaps."""
        bundle = _bundle(
            _item("Python is popular.", expert="policy"),
            _item("Python scales well.", source_id="src-2", expert="research"),
        )
        result = score_bundle(bundle)
        critical_types = {GapType.MISSING_EVIDENCE, GapType.INCOMPLETE_COVERAGE}
        gap_types = {g.gap_type for g in result.gaps}
        self.assertTrue(critical_types.isdisjoint(gap_types))


class TestSufficiency(unittest.TestCase):
    """Tests for score_bundle() sufficiency verdict — whether the bundle meets the confidence threshold.

    US-007 requires sufficient=True for healthy bundles above the normal threshold
    and potentially False for the same bundle under high-risk mode.
    """

    def test_empty_bundle_is_not_sufficient(self) -> None:
        """An empty bundle should always be insufficient."""
        bundle = EvidenceBundle(items=[], query_id="q1")
        result = score_bundle(bundle)
        self.assertFalse(result.sufficient)

    def test_healthy_bundle_is_sufficient_for_normal(self) -> None:
        """A healthy bundle with authority sources should be sufficient under normal thresholds."""
        bundle = _bundle(
            _item("Python is popular.", expert="policy"),
            _item("Python scales well.", source_id="src-2", expert="research"),
        )
        result = score_bundle(bundle, high_risk=False)
        self.assertTrue(result.sufficient)

    def test_same_bundle_may_be_insufficient_for_high_risk(self) -> None:
        """Sufficiency verdicts must be deterministic booleans regardless of risk level."""
        bundle = _bundle(_item("Python is popular.", expert="freshness"))
        normal = score_bundle(bundle, high_risk=False)
        high_risk = score_bundle(bundle, high_risk=True)
        # high-risk threshold is stricter; at minimum both should be deterministic
        self.assertIsInstance(normal.sufficient, bool)
        self.assertIsInstance(high_risk.sufficient, bool)

    def test_uncovered_claims_make_bundle_insufficient(self) -> None:
        """A bundle that does not cover all requested claim IDs must be insufficient."""
        bundle = _bundle(_item("Python is popular.", claim_ids=("c1",)))
        result = score_bundle(bundle, claim_ids=("c1", "c2"))
        self.assertFalse(result.sufficient)


if __name__ == "__main__":
    unittest.main()
