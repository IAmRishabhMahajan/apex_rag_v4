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
    """Build a CitationMetadata stub with configurable source_id, expert, and query."""
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
    """Build an EvidenceItem with configurable content, source, and claim IDs."""
    return EvidenceItem(content=content, citation=_citation(source_id), claim_ids=claim_ids)


class TestEvidenceItemValidation(unittest.TestCase):
    """Tests for EvidenceItem.validate() — enforces content and citation field constraints.

    US-004 requires that every evidence item has non-empty content, a source_id,
    and a retrieval_expert before it can enter the fusion pipeline.
    """

    def test_valid_item_has_no_errors(self) -> None:
        """A correctly constructed EvidenceItem should pass validation with no errors."""
        self.assertEqual(_item().validate(), [])

    def test_empty_content_fails(self) -> None:
        """Whitespace-only content should produce a validation error mentioning 'content'."""
        item = EvidenceItem(content="  ", citation=_citation(), claim_ids=())
        errors = item.validate()
        self.assertTrue(len(errors) > 0)
        self.assertIn("content", errors[0].lower())

    def test_missing_source_id_fails(self) -> None:
        """An empty source_id should produce a validation error mentioning 'source_id'."""
        item = EvidenceItem(
            content="Some content",
            citation=_citation(source_id=""),
            claim_ids=(),
        )
        errors = item.validate()
        self.assertTrue(any("source_id" in e for e in errors))

    def test_missing_expert_fails(self) -> None:
        """An empty retrieval_expert should produce a validation error mentioning 'retrieval_expert'."""
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
    """Tests for fuse_evidence() deduplication — identical content gets DUPLICATE status.

    US-004 requires SHA-256 content hashing so that exact or whitespace-normalised
    duplicates are detected regardless of source_id.
    """

    def test_identical_content_marked_duplicate(self) -> None:
        """Two items with the same content should result in one DUPLICATE-flagged item."""
        item_a = _item(content="The service is healthy.", source_id="src-1")
        item_b = _item(content="The service is healthy.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        statuses = [i.conflict_status for i in bundle.items]
        self.assertIn(ConflictStatus.DUPLICATE, statuses)

    def test_unique_items_not_marked_duplicate(self) -> None:
        """Items with distinct content should not be flagged as DUPLICATE."""
        item_a = _item(content="Service A is healthy.")
        item_b = _item(content="Service B has an outage.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        duplicates = [i for i in bundle.items if i.conflict_status == ConflictStatus.DUPLICATE]
        self.assertEqual(duplicates, [])

    def test_duplicate_count_property(self) -> None:
        """bundle.duplicate_count should equal the number of DUPLICATE items."""
        item_a = _item(content="Repeated text.", source_id="src-1")
        item_b = _item(content="Repeated text.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(bundle.duplicate_count, 1)

    def test_whitespace_normalisation_deduplicated(self) -> None:
        """Leading/trailing whitespace should not prevent duplicate detection."""
        item_a = _item(content="The service is healthy.")
        item_b = _item(content="  The service is healthy.  ", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(bundle.duplicate_count, 1)


class TestConflictDetection(unittest.TestCase):
    """Tests for fuse_evidence() conflict detection — negation-based heuristic flags contradictions.

    US-004 uses a negation-signal heuristic: if one text has negation words
    (not, no, false) and another has affirmative words (is, true, correct),
    both are flagged as CONFLICT.
    """

    def test_conflicting_items_flagged(self) -> None:
        """Items containing opposite truth signals should both get CONFLICT status."""
        item_a = _item(content="The feature is correct and enabled.")
        item_b = _item(content="The feature is incorrect and not enabled.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        conflicts = [i for i in bundle.items if i.conflict_status == ConflictStatus.CONFLICT]
        self.assertEqual(len(conflicts), 2)

    def test_non_conflicting_items_not_flagged(self) -> None:
        """Items about unrelated topics should not trigger conflict detection."""
        item_a = _item(content="Python is a programming language.")
        item_b = _item(content="Docker is a containerisation tool.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(bundle.conflict_count, 0)

    def test_conflict_count_property(self) -> None:
        """bundle.conflict_count should be greater than zero when conflicts are detected."""
        item_a = _item(content="The service is true and correct.")
        item_b = _item(content="The service is false and incorrect.", source_id="src-2")
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertGreater(bundle.conflict_count, 0)


class TestProvenancePreservation(unittest.TestCase):
    """Tests that fuse_evidence() preserves full citation metadata through the pipeline.

    US-004 acceptance criterion: source_id, retrieval_expert, and retrieval_query
    must pass through fusion unchanged.
    """

    def test_source_id_preserved(self) -> None:
        """The source_id from CitationMetadata must be unchanged after fusion."""
        bundle = fuse_evidence([_item(source_id="doc-42")], query_id="q1")
        self.assertEqual(bundle.items[0].citation.source_id, "doc-42")

    def test_retrieval_expert_preserved(self) -> None:
        """The retrieval_expert from CitationMetadata must be unchanged after fusion."""
        item = EvidenceItem(
            content="Evidence text.",
            citation=_citation(expert="research"),
            claim_ids=("c1",),
        )
        bundle = fuse_evidence([item], query_id="q1")
        self.assertEqual(bundle.items[0].citation.retrieval_expert, "research")

    def test_retrieval_query_preserved(self) -> None:
        """The retrieval_query from CitationMetadata must be unchanged after fusion."""
        item = EvidenceItem(
            content="Evidence text.",
            citation=_citation(query="original search query"),
            claim_ids=("c1",),
        )
        bundle = fuse_evidence([item], query_id="q1")
        self.assertEqual(bundle.items[0].citation.retrieval_query, "original search query")


class TestClaimGrouping(unittest.TestCase):
    """Tests for EvidenceBundle.by_claim() — filters items by claim ID.

    US-004 requires that items can be retrieved per claim to support
    downstream claim-evidence matching (US-009).
    """

    def test_by_claim_returns_matching_items(self) -> None:
        """by_claim should return only items whose claim_ids contain the given ID."""
        item_a = _item(claim_ids=("claim-1",))
        item_b = _item(content="Different evidence.", source_id="src-2", claim_ids=("claim-2",))
        bundle = fuse_evidence([item_a, item_b], query_id="q1")
        self.assertEqual(len(bundle.by_claim("claim-1")), 1)
        self.assertEqual(len(bundle.by_claim("claim-2")), 1)

    def test_by_claim_empty_when_no_match(self) -> None:
        """by_claim should return an empty list when no item has the requested claim ID."""
        bundle = fuse_evidence([_item(claim_ids=("claim-1",))], query_id="q1")
        self.assertEqual(bundle.by_claim("claim-99"), [])

    def test_item_can_belong_to_multiple_claims(self) -> None:
        """An item with multiple claim_ids should be returned by by_claim for each of them."""
        item = _item(claim_ids=("c1", "c2"))
        bundle = fuse_evidence([item], query_id="q1")
        self.assertEqual(len(bundle.by_claim("c1")), 1)
        self.assertEqual(len(bundle.by_claim("c2")), 1)


class TestValidationErrors(unittest.TestCase):
    """Tests for EvidenceValidationError — raised by fuse_evidence() for invalid items.

    US-004 requires that bad items are rejected before they enter the bundle.
    """

    def test_invalid_item_raises(self) -> None:
        """Passing an item with empty content to fuse_evidence should raise EvidenceValidationError."""
        bad_item = EvidenceItem(content="", citation=_citation(), claim_ids=())
        with self.assertRaises(EvidenceValidationError) as ctx:
            fuse_evidence([bad_item], query_id="q1")
        self.assertIsInstance(ctx.exception, EvidenceValidationError)
        self.assertTrue(len(ctx.exception.errors) > 0)


if __name__ == "__main__":
    unittest.main()
