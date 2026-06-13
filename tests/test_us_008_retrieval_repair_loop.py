"""Executable checks for US-008 Retrieval Repair Loop."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS008RetrievalRepairLoop(unittest.TestCase):
    """Keep retrieval repair bounded and tied to classified failures."""

    def test_story_contract(self) -> None:
        expectation = StoryExpectation(
            story_id="US-008",
            slug="retrieval-repair-loop",
            title="Retrieval Repair Loop",
            required_terms=(
                "no evidence",
                "low relevance",
                "conflicting evidence",
                "outdated evidence",
                "wrong expert",
                "max iterations",
            ),
            dependency_terms=(
                "US-003 Expert Retrieval Routing",
                "US-006 Complex Query Reasoning Path",
                "US-007 Evidence Scoring and Gap Detection",
            ),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
