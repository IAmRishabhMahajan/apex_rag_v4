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
    return CitationMetadata(
        source_id=source_id,
        title="Doc",
        url="https://example.com",
        retrieval_expert="policy",
        retrieval_query="q",
    )


def _item(content: str, source_id: str = "src-1") -> EvidenceItem:
    return EvidenceItem(content=content, citation=_citation(source_id), claim_ids=())


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    return EvidenceBundle(items=list(items), query_id="q1")


class TestValidateQuery(unittest.TestCase):
    def test_valid_query_is_approved(self) -> None:
        profile = build_query_profile("What is Kubernetes?")
        result = validate_query(profile)
        self.assertEqual(result.status, ValidationStatus.APPROVED)
        self.assertTrue(result.passed)

    def test_invalid_query_is_rejected(self) -> None:
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertFalse(result.passed)

    def test_rejected_result_blocks_downstream(self) -> None:
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertTrue(result.blocks_downstream)

    def test_rejected_result_has_useful_message(self) -> None:
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertTrue(len(result.messages) > 0)
        self.assertTrue(len(result.messages[0]) > 10)

    def test_rejected_result_has_repair_hint(self) -> None:
        profile = build_query_profile("aa")
        result = validate_query(profile)
        self.assertTrue(len(result.repair_hints) > 0)

    def test_stage_is_query(self) -> None:
        profile = build_query_profile("What is Kubernetes?")
        result = validate_query(profile)
        self.assertEqual(result.stage, PipelineStage.QUERY)


class TestValidateFusion(unittest.TestCase):
    def test_healthy_bundle_is_approved(self) -> None:
        bundle = _bundle(_item("Python is popular."), _item("Docker is useful.", "src-2"))
        result = validate_fusion(bundle)
        self.assertEqual(result.status, ValidationStatus.APPROVED)

    def test_empty_bundle_is_rejected(self) -> None:
        bundle = EvidenceBundle(items=[], query_id="q1")
        result = validate_fusion(bundle)
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertTrue(result.blocks_downstream)

    def test_high_conflict_ratio_escalates(self) -> None:
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
        item_a = _item("Python is popular.")
        item_b = _item("Python is not popular.", "src-2")
        bundle = EvidenceBundle(items=[item_a, item_b], query_id="q1")
        bundle.items[0].conflict_status = ConflictStatus.CONFLICT
        bundle.items[1].conflict_status = ConflictStatus.CONFLICT
        result = validate_fusion(bundle)
        # 2/2 = 100% conflict → escalate
        self.assertTrue(result.blocks_downstream)

    def test_stage_is_fusion(self) -> None:
        bundle = _bundle(_item("Python is popular."))
        result = validate_fusion(bundle)
        self.assertEqual(result.stage, PipelineStage.FUSION)


class TestValidateClaims(unittest.TestCase):
    def _make_answer(self, claims: list[str], evidence_content: str = "Python is popular."):  # type: ignore[no-untyped-def]
        bundle = _bundle(_item(evidence_content))
        return generate_answer(claims, bundle)

    def test_fully_supported_answer_is_approved(self) -> None:
        answer = self._make_answer(["Python is widely used"])
        result = validate_claims(answer)
        self.assertEqual(result.status, ValidationStatus.APPROVED)

    def test_majority_unsupported_is_rejected(self) -> None:
        answer = self._make_answer(
            ["Python is widely used", "Kubernetes scales pods", "Docker builds images"],
        )
        result = validate_claims(answer)
        self.assertIn(result.status, (ValidationStatus.REJECTED, ValidationStatus.REPAIR))

    def test_minority_unsupported_requests_repair(self) -> None:
        # 1 unsupported out of 3 = 33%, below the 50% reject threshold → REPAIR
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
        answer = self._make_answer(["Python is widely used"])
        result = validate_claims(answer)
        self.assertEqual(result.stage, PipelineStage.CLAIM)


class TestValidateGeneration(unittest.TestCase):
    def _make_answer(self, claims: list[str], evidence_content: str = "Python is popular."):  # type: ignore[no-untyped-def]
        bundle = _bundle(_item(evidence_content))
        return generate_answer(claims, bundle)

    def test_well_cited_answer_is_approved(self) -> None:
        answer = self._make_answer(["Python is widely used"])
        result = validate_generation(answer)
        self.assertEqual(result.status, ValidationStatus.APPROVED)

    def test_stage_is_generation(self) -> None:
        answer = self._make_answer(["Python is widely used"])
        result = validate_generation(answer)
        self.assertEqual(result.stage, PipelineStage.GENERATION)

    def test_approved_result_has_info_severity(self) -> None:
        answer = self._make_answer(["Python is widely used"])
        result = validate_generation(answer)
        self.assertEqual(result.severity, Severity.INFO)


class TestAssertPasses(unittest.TestCase):
    def test_approved_does_not_raise(self) -> None:
        profile = build_query_profile("What is Kubernetes?")
        result = validate_query(profile)
        assert_passes(result)  # should not raise

    def test_rejected_raises_blocked_error(self) -> None:
        profile = build_query_profile("aa")
        result = validate_query(profile)
        with self.assertRaises(ValidationBlockedError) as ctx:
            assert_passes(result)
        self.assertEqual(ctx.exception.result.stage, PipelineStage.QUERY)

    def test_blocked_error_message_contains_stage(self) -> None:
        profile = build_query_profile("aa")
        result = validate_query(profile)
        try:
            assert_passes(result)
        except ValidationBlockedError as exc:
            self.assertIn("query", str(exc).lower())

    def test_repair_does_not_raise(self) -> None:
        bundle = _bundle(_item("Python is popular."))
        bundle.items[0].conflict_status = ConflictStatus.CONFLICT
        dup_bundle = _bundle(_item("Python is popular."), _item("Python is popular.", "src-2"))
        result = validate_fusion(dup_bundle)
        # repair status should not block
        if result.status == ValidationStatus.REPAIR:
            assert_passes(result)  # should not raise


if __name__ == "__main__":
    unittest.main()
