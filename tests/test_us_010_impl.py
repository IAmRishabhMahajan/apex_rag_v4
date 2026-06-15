"""Real unit tests for US-010 Risk Assessment, Critique, and Verification implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    EvidenceBundle,
    EvidenceItem,
)
from src.apex_rag.generation import generate_answer
from src.apex_rag.query_intelligence import build_query_profile
from src.apex_rag.risk_verification import (
    RiskCategory,
    assess_risk,
    critique_answer,
    verify_answer,
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


class TestRiskClassification(unittest.TestCase):
    """Tests for assess_risk() — classifies query profiles into risk categories.

    US-010 requires that queries containing medical, legal, financial, or compliance
    signals are elevated to is_high_risk=True with the appropriate RiskCategory.
    Normal queries remain NORMAL and non-high-risk.
    """

    def test_normal_query_is_not_high_risk(self) -> None:
        """A general knowledge query must be NORMAL risk and not high risk."""
        profile = build_query_profile("What is Kubernetes?")
        result = assess_risk(profile)
        self.assertFalse(result.is_high_risk)
        self.assertEqual(result.category, RiskCategory.NORMAL)

    def test_medical_query_is_high_risk(self) -> None:
        """A query containing medical signals like 'medication dose' must be MEDICAL high risk."""
        profile = build_query_profile("What is the correct medication dose for this symptom?")
        result = assess_risk(profile)
        self.assertTrue(result.is_high_risk)
        self.assertEqual(result.category, RiskCategory.MEDICAL)

    def test_legal_query_is_high_risk(self) -> None:
        """A query mentioning legal terms like 'sue' and 'contract' must be LEGAL high risk."""
        profile = build_query_profile("Can I sue the contractor under the contract statute?")
        result = assess_risk(profile)
        self.assertTrue(result.is_high_risk)
        self.assertEqual(result.category, RiskCategory.LEGAL)

    def test_financial_query_is_high_risk(self) -> None:
        """A query about investment decisions must be FINANCIAL high risk."""
        profile = build_query_profile("Should I invest my portfolio in dividend stocks?")
        result = assess_risk(profile)
        self.assertTrue(result.is_high_risk)
        self.assertEqual(result.category, RiskCategory.FINANCIAL)

    def test_compliance_query_is_high_risk(self) -> None:
        """A query mentioning compliance regulations like GDPR must be COMPLIANCE high risk."""
        profile = build_query_profile("What GDPR compliance rules apply to a data breach?")
        result = assess_risk(profile)
        self.assertTrue(result.is_high_risk)
        self.assertEqual(result.category, RiskCategory.COMPLIANCE)

    def test_risk_assessment_has_reason(self) -> None:
        """A risk assessment must always include a non-empty reason string."""
        profile = build_query_profile("What is the correct medication dose?")
        result = assess_risk(profile)
        self.assertTrue(len(result.reason) > 0)

    def test_risk_signals_populated_for_high_risk(self) -> None:
        """A high-risk assessment must have at least one signal that triggered the classification."""
        profile = build_query_profile("What is the correct medication dose?")
        result = assess_risk(profile)
        self.assertTrue(len(result.signals) > 0)


class TestCritiqueAnswer(unittest.TestCase):
    """Tests for critique_answer() — checks each answer sentence for hedging and lack of support.

    US-010 requires that hedged sentences ('it is widely believed', 'many think') and
    sentences with no evidence overlap are removed. The critique result must list
    removed sentences and describe issues for audit trails.
    """

    def _answer(self, evidence: str, claims: list[str]) -> object:
        """Generate an answer from the given evidence text and candidate claims."""
        bundle = _bundle(_item(evidence))
        return generate_answer(claims, bundle)

    def test_well_supported_answer_passes_critique(self) -> None:
        """An answer where every sentence is evidence-backed should pass with no removals."""
        bundle = _bundle(_item("Python is popular."))
        answer = generate_answer(["Python is widely used"], bundle)
        result = critique_answer(answer, bundle)
        self.assertTrue(result.passed)
        self.assertEqual(result.removed_sentences, ())

    def test_hedged_sentence_is_removed(self) -> None:
        """A sentence containing hedging language like 'widely believed' must be in removed_sentences."""
        bundle = _bundle(
            _item("Python is widely used."),
            _item("It is widely believed Python scales well.", source_id="src-2"),
        )
        from src.apex_rag.generation import GeneratedAnswer

        mock_answer = GeneratedAnswer(
            text=("Python is widely used. It is widely believed that Python scales well."),
            approved_claims=(),
            citation_links=(),
            has_limitations=False,
            limitation_note="",
        )
        result = critique_answer(mock_answer, bundle)
        self.assertFalse(result.passed)
        self.assertTrue(len(result.removed_sentences) > 0)
        self.assertTrue(any("widely believed" in s for s in result.removed_sentences))

    def test_unsupported_sentence_is_removed(self) -> None:
        """A sentence with no word overlap with any evidence must be placed in removed_sentences."""
        from src.apex_rag.generation import GeneratedAnswer

        bundle = _bundle(_item("Python is popular."))
        mock_answer = GeneratedAnswer(
            text="Python is popular. Kubernetes orchestrates large clusters.",
            approved_claims=(),
            citation_links=(),
            has_limitations=False,
            limitation_note="",
        )
        result = critique_answer(mock_answer, bundle)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("Kubernetes" in s for s in result.removed_sentences),
        )

    def test_critique_issues_describe_what_was_removed(self) -> None:
        """When sentences are removed, issues must be non-empty to explain what was removed."""
        from src.apex_rag.generation import GeneratedAnswer

        bundle = _bundle(_item("Python is popular."))
        mock_answer = GeneratedAnswer(
            text="Python is popular. It is widely believed that Rust is faster.",
            approved_claims=(),
            citation_links=(),
            has_limitations=False,
            limitation_note="",
        )
        result = critique_answer(mock_answer, bundle)
        self.assertTrue(len(result.issues) > 0)


class TestVerifyAnswer(unittest.TestCase):
    """Tests for verify_answer() — applies critique conditionally based on risk level.

    US-010 requires that normal-risk queries skip critique (returning a passed result),
    while high-risk queries always go through critique and receive a disclaimer.
    Unsupported or hedged sentences are stripped from the final verified text.
    """

    def test_normal_risk_skips_critique(self) -> None:
        """A normal-risk query should produce a verified answer with critique.passed=True."""
        profile = build_query_profile("What is Kubernetes?")
        bundle = _bundle(_item("Kubernetes is a container orchestration platform."))
        answer = generate_answer(["Kubernetes is widely used"], bundle)
        result = verify_answer(answer, bundle, profile)
        self.assertFalse(result.requires_escalation)
        self.assertTrue(result.critique.passed)

    def test_high_risk_answer_goes_through_critique(self) -> None:
        """A high-risk query must produce a VerifiedAnswer with is_high_risk=True on its assessment."""
        profile = build_query_profile("What medication dose should I take?")
        bundle = _bundle(_item("Medication dosage depends on patient weight."))
        answer = generate_answer(["Medication dosage varies"], bundle)
        result = verify_answer(answer, bundle, profile)
        self.assertTrue(result.risk_assessment.is_high_risk)

    def test_high_risk_answer_includes_disclaimer(self) -> None:
        """A high-risk answer must have a non-empty disclaimer in its critique result."""
        profile = build_query_profile("What medication dose should I take?")
        bundle = _bundle(_item("Medication dosage depends on patient weight."))
        answer = generate_answer(["Medication dosage varies"], bundle)
        result = verify_answer(answer, bundle, profile)
        self.assertTrue(len(result.critique.disclaimer) > 0)

    def test_normal_risk_answer_text_unchanged(self) -> None:
        """A normal-risk answer must have text identical to the original generated answer."""
        profile = build_query_profile("What is idempotency?")
        bundle = _bundle(_item("Idempotency means the same result every time."))
        answer = generate_answer(["Idempotency ensures consistent results"], bundle)
        result = verify_answer(answer, bundle, profile)
        self.assertEqual(result.text, answer.text)

    def test_unsupported_high_risk_sentences_removed(self) -> None:
        """Hedged sentences in a high-risk answer must be stripped from the verified text."""
        from src.apex_rag.generation import GeneratedAnswer

        profile = build_query_profile("What medication dose should I take?")
        bundle = _bundle(_item("Medication dosage depends on patient weight."))
        mock_answer = GeneratedAnswer(
            text=(
                "Medication dosage depends on patient weight. "
                "It is widely believed that higher doses are always better."
            ),
            approved_claims=(),
            citation_links=(),
            has_limitations=False,
            limitation_note="",
        )
        result = verify_answer(mock_answer, bundle, profile)
        self.assertNotIn("widely believed", result.text)

    def test_verified_answer_has_risk_assessment(self) -> None:
        """A verified answer must expose a risk_assessment with the correct RiskCategory."""
        profile = build_query_profile("What is the correct medication dose?")
        bundle = _bundle(_item("Medication dosage depends on weight."))
        answer = generate_answer(["Medication dosage varies"], bundle)
        result = verify_answer(answer, bundle, profile)
        self.assertIsNotNone(result.risk_assessment)
        self.assertEqual(result.risk_assessment.category, RiskCategory.MEDICAL)


if __name__ == "__main__":
    unittest.main()
