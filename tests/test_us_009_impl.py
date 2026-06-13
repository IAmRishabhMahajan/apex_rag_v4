"""Real unit tests for US-009 Grounded Reasoning and Generation implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    ConflictStatus,
    EvidenceBundle,
    EvidenceItem,
)
from src.apex_rag.generation import (
    ClaimStatus,
    GroundingError,
    approve_claims,
    generate_answer,
)


def _citation(source_id: str = "src-1", expert: str = "policy") -> CitationMetadata:
    return CitationMetadata(
        source_id=source_id,
        title=f"Document {source_id}",
        url=f"https://example.com/{source_id}",
        retrieval_expert=expert,
        retrieval_query="test",
    )


def _item(
    content: str,
    source_id: str = "src-1",
    conflict: ConflictStatus = ConflictStatus.NONE,
) -> EvidenceItem:
    item = EvidenceItem(content=content, citation=_citation(source_id), claim_ids=())
    item.conflict_status = conflict
    return item


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    return EvidenceBundle(items=list(items), query_id="q1")


class TestClaimApproval(unittest.TestCase):
    def test_supported_claim_when_evidence_matches(self) -> None:
        bundle = _bundle(_item("Python is a popular programming language."))
        approved = approve_claims(["Python is widely used"], bundle)
        self.assertEqual(approved[0].status, ClaimStatus.SUPPORTED)

    def test_unsupported_claim_when_no_evidence_matches(self) -> None:
        bundle = _bundle(_item("Docker is a containerisation platform."))
        approved = approve_claims(["Kubernetes scales workloads"], bundle)
        self.assertEqual(approved[0].status, ClaimStatus.UNSUPPORTED)

    def test_citation_links_populated_for_supported_claim(self) -> None:
        bundle = _bundle(_item("Python is popular.", source_id="doc-7"))
        approved = approve_claims(["Python is widely used"], bundle)
        self.assertTrue(len(approved[0].citation_links) > 0)
        self.assertEqual(approved[0].citation_links[0].evidence_source_id, "doc-7")

    def test_unsupported_claim_has_empty_citations(self) -> None:
        bundle = _bundle(_item("Docker is a container tool."))
        approved = approve_claims(["Kubernetes orchestrates pods"], bundle)
        self.assertEqual(approved[0].citation_links, ())

    def test_conflicted_evidence_excluded_from_support(self) -> None:
        bundle = _bundle(_item("Python is popular.", conflict=ConflictStatus.CONFLICT))
        approved = approve_claims(["Python is widely used"], bundle)
        self.assertEqual(approved[0].status, ClaimStatus.UNSUPPORTED)

    def test_multiple_claims_independently_assessed(self) -> None:
        bundle = _bundle(
            _item("Python is popular."),
            _item("Docker simplifies deployment.", source_id="src-2"),
        )
        approved = approve_claims(
            ["Python is widely used", "Kubernetes orchestrates clusters"],
            bundle,
        )
        self.assertEqual(approved[0].status, ClaimStatus.SUPPORTED)
        self.assertEqual(approved[1].status, ClaimStatus.UNSUPPORTED)


class TestGenerateAnswer(unittest.TestCase):
    def test_answer_text_from_supported_claims_only(self) -> None:
        bundle = _bundle(_item("Python is a popular language."))
        answer = generate_answer(["Python is widely used"], bundle)
        self.assertIn("Python", answer.text)

    def test_citations_present_for_supported_claims(self) -> None:
        bundle = _bundle(_item("Python is popular.", source_id="doc-1"))
        answer = generate_answer(["Python is widely used"], bundle)
        self.assertTrue(len(answer.citation_links) > 0)

    def test_citations_map_to_source_evidence(self) -> None:
        bundle = _bundle(_item("Python is popular.", source_id="doc-42"))
        answer = generate_answer(["Python is widely used"], bundle)
        source_ids = {link.evidence_source_id for link in answer.citation_links}
        self.assertIn("doc-42", source_ids)

    def test_unsupported_claims_excluded_from_answer_text(self) -> None:
        bundle = _bundle(_item("Python is popular."))
        answer = generate_answer(
            ["Python is widely used", "Kubernetes orchestrates clusters"],
            bundle,
        )
        self.assertNotIn("Kubernetes orchestrates clusters", answer.text)
        self.assertGreater(answer.unsupported_claim_count, 0)

    def test_limitations_surfaced_when_unsupported_claims(self) -> None:
        bundle = _bundle(_item("Python is popular."))
        answer = generate_answer(
            ["Python is widely used", "Kubernetes orchestrates clusters"],
            bundle,
        )
        self.assertTrue(answer.has_limitations)
        self.assertIn("excluded", answer.limitation_note.lower())

    def test_no_limitations_when_all_supported(self) -> None:
        bundle = _bundle(_item("Python is a popular language."))
        answer = generate_answer(["Python is widely used"], bundle)
        self.assertFalse(answer.has_limitations)
        self.assertEqual(answer.limitation_note, "")

    def test_conflict_note_when_bundle_has_conflicts(self) -> None:
        good = _item("Python is widely used in data science.", source_id="src-2")
        bad = _item("Python is a popular language.", source_id="src-1")
        bundle = EvidenceBundle(items=[bad, good], query_id="q1")
        bundle.items[0].conflict_status = ConflictStatus.CONFLICT
        answer = generate_answer(["Python is used in data science"], bundle)
        self.assertTrue(answer.has_limitations)
        self.assertIn("conflicting", answer.limitation_note.lower())

    def test_all_claims_supported_property(self) -> None:
        bundle = _bundle(_item("Python is popular."))
        answer = generate_answer(["Python is widely used"], bundle)
        self.assertTrue(answer.all_claims_supported)

    def test_no_duplicate_citation_links(self) -> None:
        bundle = _bundle(_item("Python is a popular language.", source_id="src-1"))
        answer = generate_answer(["Python is widely used", "Python is popular"], bundle)
        pairs = [(lk.claim_text, lk.evidence_source_id) for lk in answer.citation_links]
        self.assertEqual(len(pairs), len(set(pairs)))


class TestGroundingErrors(unittest.TestCase):
    def test_empty_bundle_raises_grounding_error(self) -> None:
        bundle = EvidenceBundle(items=[], query_id="q1")
        with self.assertRaises(GroundingError) as ctx:
            generate_answer(["Some claim"], bundle)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_no_matching_claims_raises_grounding_error(self) -> None:
        bundle = _bundle(_item("Docker is a container tool."))
        with self.assertRaises(GroundingError) as ctx:
            generate_answer(["Kubernetes orchestrates clusters"], bundle)
        self.assertIn("fails closed", str(ctx.exception).lower())

    def test_grounding_error_has_reason_attribute(self) -> None:
        bundle = EvidenceBundle(items=[], query_id="q1")
        try:
            generate_answer(["claim"], bundle)
        except GroundingError as exc:
            self.assertTrue(len(exc.reason) > 0)


if __name__ == "__main__":
    unittest.main()
