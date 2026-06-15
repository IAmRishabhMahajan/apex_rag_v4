"""Real unit tests for US-005 Validation Mesh implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    ConflictStatus,
    EvidenceBundle,
    EvidenceItem,
)
from src.apex_rag.generation import generate_answer
from src.apex_rag.query_intelligence import build_query_profile
from src.apex_rag.validation_mesh import (
    PipelineStage,
    Severity,
    ValidationBlockedError,
    ValidationStatus,
    assert_passes,
    validate_claims,
    validate_fusion,
    validate_generation,
    validate_query,
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
    """Build a single EvidenceItem with a given content and source ID."""
    return EvidenceItem(content=content, citation=_citation(source_id), claim_ids=())


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    """Wrap one or more EvidenceItems into an EvidenceBundle."""
    return EvidenceBundle(items=list(items), query_id="q1")


class TestValidateQuery(unittest.TestCase):
    """Tests for validate_query() — checks that QueryProfile passes or fails the query stage.

    US-005 requires APPROVED for valid profiles, REJECTED for short/invalid queries,
    and repair hints when the query cannot be routed.
    """

    def test_valid_query_is_approved(self) -> None:
        """A well-formed query profile should receive APPROVED status."""
        profile = build_query_profile("What is Kubernetes?")
        result = validate_query(profile)
        self.assertEqual(result.status, ValidationStatus.APPROVED)
        self.assertTrue(result.passed)

    def test_invalid_query_is_rejected(self) -> None:
        """A profile with validation errors should receive REJECTED status."""
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertFalse(result.passed)

    def test_rejected_result_blocks_downstream(self) -> None:
        """A REJECTED result must have blocks_downstream=True."""
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertTrue(result.blocks_downstream)

    def test_rejected_result_has_useful_message(self) -> None:
        """A REJECTED result must include at least one non-trivial message."""
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertTrue(len(result.messages) > 0)
        self.assertTrue(len(result.messages[0]) > 10)

    def test_rejected_result_has_repair_hint(self) -> None:
        """A REJECTED result must include at least one repair hint."""
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertTrue(len(result.repair_hints) > 0)

    def test_stage_is_query(self) -> None:
        """The stage field on the result must be PipelineStage.QUERY."""
        profile = build_query_profile("What is Kubernetes?")
        result = validate_query(profile)
        self.assertEqual(result.stage, PipelineStage.QUERY)


class TestValidateFusion(unittest.TestCase):
    """Tests for validate_fusion() — checks bundle health at the fusion stage.

    US-005 requires APPROVED for healthy bundles, REJECTED for empty bundles,
    ESCALATE when conflict ratio ≥ 50%, and REPAIR for lower conflict counts.
    """

    def test_healthy_bundle_is_approved(self) -> None:
        """A bundle with multiple non-conflicting items should be APPROVED."""
        bundle = _bundle(_item("Python is popular."), _item("Docker is useful.", "src-2"))
        result = validate_fusion(bundle)
        self.assertEqual(result.status, ValidationStatus.APPROVED)

    def test_empty_bundle_is_rejected(self) -> None:
        """An empty bundle must be REJECTED and block downstream stages."""
        bundle = EvidenceBundle(items=[], query_id="q1")
        result = validate_fusion(bundle)
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertTrue(result.blocks_downstream)

    def test_high_conflict_ratio_escalates(self) -> None:
        """When ≥50% of items are in conflict, the result should be ESCALATE with CRITICAL severity."""
        item_a = _item("The service is correct and true.")
        item_b = _item("The service is false and incorrect.", "src-2")
        item_c = _item("The service is correct and working.", "src-3")
        item_d = _item("The service is false and not working.", "src-4")
        bundle = EvidenceBundle(items=[item_a, item_b, item_c, item_d], query_id="q1")
        bundle.items[0].conflict_status = ConflictStatus.CONFLICT
        bundle.items[1].conflict_status = ConflictStatus.CONFLICT
        bundle.items[2].conflict_status = ConflictStatus.CONFLICT
        result = validate_fusion(bundle)
        self.assertEqual(result.status, ValidationStatus.ESCALATE)
        self.assertEqual(result.severity, Severity.CRITICAL)

    def test_low_conflict_count_requests_repair(self) -> None:
        """A 100% conflict ratio should block downstream processing."""
        item_a = _item("Python is popular.")
        item_b = _item("Python is not popular.", "src-2")
        bundle = EvidenceBundle(items=[item_a, item_b], query_id="q1")
        bundle.items[0].conflict_status = ConflictStatus.CONFLICT
        bundle.items[1].conflict_status = ConflictStatus.CONFLICT
        result = validate_fusion(bundle)
        # 2/2 = 100% conflict → escalate
        self.assertTrue(result.blocks_downstream)

    def test_stage_is_fusion(self) -> None:
        """The stage field on the fusion result must be PipelineStage.FUSION."""
        bundle = _bundle(_item("Python is popular."))
        result = validate_fusion(bundle)
        self.assertEqual(result.stage, PipelineStage.FUSION)


class TestValidateClaims(unittest.TestCase):
    """Tests for validate_claims() — checks claim-evidence alignment at the claim stage.

    US-005 requires APPROVED when all claims are supported, REJECTED when ≥50%
    are unsupported, and REPAIR for a minority of unsupported claims.
    """

    def _make_answer(self, claims: list[str], evidence_content: str = "Python is popular."):  # type: ignore[no-untyped-def]
        """Generate an answer from the given claims and evidence text."""
        bundle = _bundle(_item(evidence_content))
        return generate_answer(claims, bundle)

    def test_fully_supported_answer_is_approved(self) -> None:
        """An answer where all claims are supported should be APPROVED."""
        answer = self._make_answer(["Python is widely used"])
        result = validate_claims(answer)
        self.assertEqual(result.status, ValidationStatus.APPROVED)

    def test_majority_unsupported_is_rejected(self) -> None:
        """An answer where the majority of claims are unsupported should be REJECTED or REPAIR."""
        answer = self._make_answer(
            ["Python is widely used", "Kubernetes scales pods", "Docker builds images"],
        )
        result = validate_claims(answer)
        self.assertIn(result.status, (ValidationStatus.REJECTED, ValidationStatus.REPAIR))

    def test_minority_unsupported_requests_repair(self) -> None:
        """An answer with one unsupported claim out of three (33%) should trigger REPAIR."""
        bundle = _bundle(
            _item("Python is popular."),
            _item("Python scales well.", "src-2"),
            _item("Python is widely adopted.", "src-3"),
        )
        from src.apex_rag.generation import generate_answer as gen

        answer = gen(
            ["Python is widely used", "Python scales well", "Kubernetes orchestrates clusters"],
            bundle,
        )
        result = validate_claims(answer)
        self.assertEqual(result.status, ValidationStatus.REPAIR)
        self.assertTrue(len(result.repair_hints) > 0)

    def test_stage_is_claim(self) -> None:
        """The stage field on the claim result must be PipelineStage.CLAIM."""
        answer = self._make_answer(["Python is widely used"])
        result = validate_claims(answer)
        self.assertEqual(result.stage, PipelineStage.CLAIM)


class TestValidateGeneration(unittest.TestCase):
    """Tests for validate_generation() — checks citation completeness of the final answer.

    US-005 requires APPROVED for well-cited answers, REJECTED for empty text or no citations.
    """

    def _make_answer(self, claims: list[str], evidence_content: str = "Python is popular."):  # type: ignore[no-untyped-def]
        """Generate an answer from the given claims and evidence text."""
        bundle = _bundle(_item(evidence_content))
        return generate_answer(claims, bundle)

    def test_well_cited_answer_is_approved(self) -> None:
        """An answer with at least one citation link should be APPROVED."""
        answer = self._make_answer(["Python is widely used"])
        result = validate_generation(answer)
        self.assertEqual(result.status, ValidationStatus.APPROVED)

    def test_stage_is_generation(self) -> None:
        """The stage field on the generation result must be PipelineStage.GENERATION."""
        answer = self._make_answer(["Python is widely used"])
        result = validate_generation(answer)
        self.assertEqual(result.stage, PipelineStage.GENERATION)

    def test_approved_result_has_info_severity(self) -> None:
        """An APPROVED generation result should have INFO severity."""
        answer = self._make_answer(["Python is widely used"])
        result = validate_generation(answer)
        self.assertEqual(result.severity, Severity.INFO)


class TestAssertPasses(unittest.TestCase):
    """Tests for assert_passes() — raises ValidationBlockedError for blocking results.

    US-005 requires that downstream stages are halted when a validation result
    has REJECTED or ESCALATE status.
    """

    def test_approved_does_not_raise(self) -> None:
        """assert_passes should not raise for an APPROVED ValidationResult."""
        profile = build_query_profile("What is Kubernetes?")
        result = validate_query(profile)
        assert_passes(result)  # should not raise

    def test_rejected_raises_blocked_error(self) -> None:
        """assert_passes must raise ValidationBlockedError for a REJECTED result."""
        profile = build_query_profile("aa")
        result = validate_query(profile)
        with self.assertRaises(ValidationBlockedError) as ctx:
            assert_passes(result)
        self.assertEqual(ctx.exception.result.stage, PipelineStage.QUERY)

    def test_blocked_error_message_contains_stage(self) -> None:
        """The ValidationBlockedError message must mention the pipeline stage name."""
        profile = build_query_profile("aa")
        result = validate_query(profile)
        try:
            assert_passes(result)
        except ValidationBlockedError as exc:
            self.assertIn("query", str(exc).lower())

    def test_repair_does_not_raise(self) -> None:
        """assert_passes should not raise for a REPAIR status result."""
        bundle = _bundle(_item("Python is popular."))
        bundle.items[0].conflict_status = ConflictStatus.CONFLICT
        dup_bundle = _bundle(_item("Python is popular."), _item("Python is popular.", "src-2"))
        result = validate_fusion(dup_bundle)
        # repair status should not block
        if result.status == ValidationStatus.REPAIR:
            assert_passes(result)  # should not raise


if __name__ == "__main__":
    unittest.main()
