"""Executable checks for US-006 Complex Query Reasoning Path."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS006ComplexQueryReasoningPath(unittest.TestCase):
    """Keep the complex path optional, claim-based, and source-preserving."""

    def test_story_contract(self) -> None:
        expectation = StoryExpectation(
            story_id="US-006",
            slug="complex-query-reasoning-path",
            title="Complex Query Reasoning Path",
            required_terms=(
                "complexity assessment gate",
                "Claim",
                "claim graph",
                "Compress",
                "Simple queries bypass",
                "source links",
            ),
            dependency_terms=(
                "US-002 Adaptive Retrieval Planning",
                "US-004 Evidence Fusion",
                "US-005 Validation Mesh",
            ),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
