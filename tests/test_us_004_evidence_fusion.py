"""Executable checks for US-004 Evidence Fusion."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS004EvidenceFusion(unittest.TestCase):
    """Keep evidence fusion grounded in traceability and provenance."""

    def test_story_contract(self) -> None:
        """Verify US-004 story file has required sections, terms, and minimum bullet counts."""
        expectation = StoryExpectation(
            story_id="US-004",
            slug="evidence-fusion",
            title="Evidence Fusion",
            required_terms=(
                "EvidenceItem",
                "EvidenceBundle",
                "Deduplicate",
                "Conflicting evidence",
                "source",
                "provenance",
            ),
            dependency_terms=("US-003 Expert Retrieval Routing",),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
