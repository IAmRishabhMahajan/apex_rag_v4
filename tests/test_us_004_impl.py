"""Real unit tests for US-004 Evidence Fusion implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    ConflictStatus,
    EvidenceItem,
    EvidenceValidationError,
    fuse_evidence,
)


def _citation(
    source_id: str = "src-1",
    expert: str = "policy",
    query: str = "test query",
) -> CitationMetadata:
    return CitationMetadata(
        source_id=source_id,
        title="Test Document",
        url=f"https://example.com/{source_id}",
        retrieval_expert=expert,
        retrieval_query=query,
    )


def _item(
    content: str = "The service is healthy.",
    source_id: str = "src-1",
    claim_ids: tuple[str, ...] = ("claim-1",),
) -> EvidenceItem:
    return EvidenceItem(content=content, citation=_citation(source_id), claim_ids=claim_ids)


class TestEvidenceItemValidation(unittest.TestCase):
    def test_valid_item_has_no_errors(self) -> None:
        self.assertEqual(_item().validate(), [])

    def test_empty_content_fails(self) -> None:
        item = EvidenceItem(content="  ", citation=_citation(), claim_ids=())
        errors = item.validate()
        self.assertTrue(len(errors) > 0)
        self.assertIn("content", errors[0].lower())

    def test_missing_source_id_fails(self) -> None:
        item = EvidenceItem(
            content="Some content",
            citation=_citation(source_id=""),
            claim_ids=(),
        )
        errors = item.validate()
        self.assertTrue(any("source_id" in e for e in errors))

    def test_missing_expert_fails(self) -> None:
        item = EvidenceItem(
            content="Some content",
            citation=CitationMetadata(
                source_id="s1",
                title="T",
                url="u",
                retrieval_expert="",
                retrieval_query="q",
            ),
            claim_ids=(),
        )
        errors = item.validate()
        self.assertTrue(any("retrieval_expert" in e for e in errors))


class TestDeduplication(unittest.TestCase):
    def test_identical_content_marked_duplicate(self) -> None:
        item_a = _item(content="The service is healthy.", source_id="src-1")
        item_b = _item(content="The service is healthy.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        statuses = [i.conflict_status for i in bundle.items]
        self.assertIn(ConflictStatus.DUPLICATE, statuses)

    def test_unique_items_not_marked_duplicate(self) -> None:
        item_a = _item(content="Service A is healthy.")
        item_b = _item(content="Service B has an outage.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        duplicates = [i for i in bundle.items if i.conflict_status == ConflictStatus.DUPLICATE]
        self.assertEqual(duplicates, [])

    def test_duplicate_count_property(self) -> None:
        item_a = _item(content="Repeated text.", source_id="src-1")
        item_b = _item(content="Repeated text.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(bundle.duplicate_count, 1)

    def test_whitespace_normalisation_deduplicated(self) -> None:
        item_a = _item(content="The service is healthy.")
        item_b = _item(content="  The service is healthy.  ", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(bundle.duplicate_count, 1)


class TestConflictDetection(unittest.TestCase):
    def test_conflicting_items_flagged(self) -> None:
        item_a = _item(content="The feature is correct and enabled.")
        item_b = _item(content="The feature is incorrect and not enabled.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        conflicts = [i for i in bundle.items if i.conflict_status == ConflictStatus.CONFLICT]
        self.assertEqual(len(conflicts), 2)

    def test_non_conflicting_items_not_flagged(self) -> None:
        item_a = _item(content="Python is a programming language.")
        item_b = _item(content="Docker is a containerisation tool.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(bundle.conflict_count, 0)

    def test_conflict_count_property(self) -> None:
        item_a = _item(content="The service is true and correct.")
        item_b = _item(content="The service is false and incorrect.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertGreater(bundle.conflict_count, 0)


class TestProvenancePreservation(unittest.TestCase):
    def test_source_id_preserved(self) -> None:
        bundle = fuse_evidence([_item(source_id="doc-42")], query_id="q1")
        self.assertEqual(bundle.items[0].citation.source_id, "doc-42")

    def test_retrieval_expert_preserved(self) -> None:
        item = EvidenceItem(
            content="Evidence text.",
            citation=_citation(expert="research"),
            claim_ids=("c1",),
        )
        bundle = fuse_evidence([item], query_id="q1")
        self.assertEqual(bundle.items[0].citation.retrieval_expert, "research")

    def test_retrieval_query_preserved(self) -> None:
        item = EvidenceItem(
            content="Evidence text.",
            citation=_citation(query="original search query"),
            claim_ids=("c1",),
        )
        bundle = fuse_evidence([item], query_id="q1")
        self.assertEqual(bundle.items[0].citation.retrieval_query, "original search query")


class TestClaimGrouping(unittest.TestCase):
    def test_by_claim_returns_matching_items(self) -> None:
        item_a = _item(claim_ids=("claim-1",))
        item_b = _item(content="Different evidence.", source_id="src-2", claim_ids=("claim-2",))
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(len(bundle.by_claim("claim-1")), 1)
        self.assertEqual(len(bundle.by_claim("claim-2")), 1)

    def test_by_claim_empty_when_no_match(self) -> None:
        bundle = fuse_evidence([_item(claim_ids=("claim-1",))], query_id="q1")
        self.assertEqual(bundle.by_claim("claim-99"), [])

    def test_item_can_belong_to_multiple_claims(self) -> None:
        item = _item(claim_ids=("c1", "c2"))
        bundle = fuse_evidence([item], query_id="q1")
        self.assertEqual(len(bundle.by_claim("c1")), 1)
        self.assertEqual(len(bundle.by_claim("c2")), 1)


class TestValidationErrors(unittest.TestCase):
    def test_invalid_item_raises(self) -> None:
        bad_item = EvidenceItem(content="", citation=_citation(), claim_ids=())
        with self.assertRaises(EvidenceValidationError) as ctx:
            fuse_evidence([bad_item], query_id="q1")
        self.assertIsInstance(ctx.exception, EvidenceValidationError)
        self.assertTrue(len(ctx.exception.errors) > 0)


if __name__ == "__main__":
    unittest.main()
