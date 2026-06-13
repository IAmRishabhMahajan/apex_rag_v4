"""Executable checks for US-009 Grounded Reasoning and Generation."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS009GroundedReasoningGeneration(unittest.TestCase):
    """Keep final generation constrained by approved evidence."""

    def test_story_contract(self) -> None:
        expectation = StoryExpectation(
            story_id="US-009",
            slug="grounded-reasoning-generation",
            title="Grounded Reasoning and Generation",
            required_terms=(
                "approved claims",
                "evidence links",
                "citation mappings",
                "Unsupported claims",
                "confidence calibration",
                "fails closed",
            ),
            dependency_terms=(
                "US-004 Evidence Fusion",
                "US-005 Validation Mesh",
                "US-007 Evidence Scoring and Gap Detection",
            ),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
