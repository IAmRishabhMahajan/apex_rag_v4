"""Executable checks for US-007 Evidence Scoring and Gap Detection."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS007EvidenceScoringGapDetection(unittest.TestCase):
    """Keep evidence scoring connected to repairable gap reports."""

    def test_story_contract(self) -> None:
        """Verify US-007 story file has required sections, terms, and minimum bullet counts."""
        expectation = StoryExpectation(
            story_id="US-007",
            slug="evidence-scoring-gap-detection",
            title="Evidence Scoring and Gap Detection",
            required_terms=(
                "authority",
                "freshness",
                "agreement",
                "completeness",
                "confidence",
                "Gap reports",
            ),
            dependency_terms=("US-004 Evidence Fusion", "US-005 Validation Mesh"),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
