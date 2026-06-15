"""Executable checks for US-003 Expert Retrieval Routing."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS003ExpertRetrievalRouting(unittest.TestCase):
    """Keep expert routing explicit enough for future integrations."""

    def test_story_contract(self) -> None:
        """Verify US-003 story file has required sections, terms, and minimum bullet counts."""
        expectation = StoryExpectation(
            story_id="US-003",
            slug="expert-retrieval-routing",
            title="Expert Retrieval Routing",
            required_terms=(
                "policy expert",
                "research expert",
                "analytics expert",
                "graph expert",
                "freshness expert",
                "search expert",
            ),
            dependency_terms=("US-002 Adaptive Retrieval Planning",),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
