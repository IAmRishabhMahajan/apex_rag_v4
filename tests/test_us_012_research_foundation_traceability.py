"""Executable checks for US-012 Research Foundation Traceability."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS012ResearchFoundationTraceability(unittest.TestCase):
    """Keep research influences visible without overstating implementation."""

    def test_story_contract(self) -> None:
        """Verify US-012 story file has required sections, terms, and minimum bullet counts."""
        expectation = StoryExpectation(
            story_id="US-012",
            slug="research-foundation-traceability",
            title="Research Foundation Traceability",
            required_terms=(
                "CRAG",
                "Self-RAG",
                "GraphRAG",
                "RAGAS",
                "Deferred ideas",
                "not implemented",
            ),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
