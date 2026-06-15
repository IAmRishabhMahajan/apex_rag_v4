"""Executable checks for US-011 APEX-Eval Framework."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS011ApexEvalFramework(unittest.TestCase):
    """Keep evaluation coverage broad enough for the full RAG pipeline."""

    def test_story_contract(self) -> None:
        """Verify US-011 story file has required sections, terms, and minimum bullet counts."""
        expectation = StoryExpectation(
            story_id="US-011",
            slug="apex-eval-framework",
            title="APEX-Eval Framework",
            required_terms=(
                "Recall@K",
                "Precision@K",
                "unsupported claim rate",
                "recovery success rate",
                "faithfulness",
                "human-readable evaluation reports",
            ),
            dependency_terms=(
                "US-004 Evidence Fusion",
                "US-008 Retrieval Repair Loop",
                "US-009 Grounded Reasoning and Generation",
            ),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
