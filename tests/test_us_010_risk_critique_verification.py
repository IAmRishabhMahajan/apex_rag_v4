"""Executable checks for US-010 Risk Assessment, Critique, and Verification."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS010RiskCritiqueVerification(unittest.TestCase):
    """Keep high-risk answer safeguards explicit and testable."""

    def test_story_contract(self) -> None:
        """Verify US-010 story file has required sections, terms, and minimum bullet counts."""
        expectation = StoryExpectation(
            story_id="US-010",
            slug="risk-critique-verification",
            title="Risk Assessment, Critique, and Verification",
            required_terms=(
                "risk assessment gate",
                "medical",
                "legal",
                "financial",
                "answer critic",
                "Verify each sentence",
            ),
            dependency_terms=("US-005 Validation Mesh", "US-009 Grounded Reasoning and Generation"),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
