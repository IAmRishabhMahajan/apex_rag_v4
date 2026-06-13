"""Executable checks for US-001 Query Intelligence."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS001QueryIntelligence(unittest.TestCase):
    """Keep the query intelligence story specific enough to guide implementation."""

    def test_story_contract(self) -> None:
        expectation = StoryExpectation(
            story_id="US-001",
            slug="query-intelligence",
            title="Query Intelligence",
            required_terms=(
                "QueryProfile",
                "intent",
                "entities",
                "constraints",
                "query expansions",
                "ambiguous",
            ),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
