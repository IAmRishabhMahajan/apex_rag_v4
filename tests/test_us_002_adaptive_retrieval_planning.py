"""Executable checks for US-002 Adaptive Retrieval Planning."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS002AdaptiveRetrievalPlanning(unittest.TestCase):
    """Keep retrieval planning tied to adaptive strategy selection."""

    def test_story_contract(self) -> None:
        expectation = StoryExpectation(
            story_id="US-002",
            slug="adaptive-retrieval-planning",
            title="Adaptive Retrieval Planning",
            required_terms=(
                "RetrievalPlan",
                "standard",
                "multi-hop",
                "graph",
                "freshness",
                "fallback",
            ),
            dependency_terms=("US-001 Query Intelligence",),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
